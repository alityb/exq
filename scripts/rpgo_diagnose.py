#!/usr/bin/env python3
"""R-PGO Diagnostic: analyze a model and predict optimization benefit in <3 seconds.

Usage:
    python scripts/rpgo_diagnose.py --profile profiles/olmoe.json
    python scripts/rpgo_diagnose.py --profile profiles/qwen3-30b-a3b.json

This is the compile-time diagnostic that tells you WHETHER to deploy R-PGO
before running any expensive evaluation. No other tool gives you this.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="R-PGO: Compile-time deployment diagnostic"
    )
    parser.add_argument("--profile", required=True, help="Routing profile path")
    parser.add_argument("--emit-kernels", action="store_true", help="Also emit Triton kernels")
    parser.add_argument("--output-dir", default="compiled/", help="Kernel output directory")
    args = parser.parse_args()

    from rpgo._core import (
        CompilerPipeline,
        RoutingProfile,
        py_build_routing_graph,
        py_graph_summary,
    )
    from rpgo.eval.coverage import CoverageAnalyzer

    # ── Load and compile ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    profile = RoutingProfile.load(args.profile)
    graph = py_build_routing_graph(profile)

    layer_indices = profile.moe_layer_indices()
    first_layer = profile.get_layer(layer_indices[0])
    n_experts = first_layer.n_experts
    top_k = first_layer.top_k

    pipe = CompilerPipeline()
    pipe.run_auto(graph, n_experts, top_k)
    compile_time = time.perf_counter() - t0

    # ── Gather metrics ───────────────────────────────────────────────────
    summary = py_graph_summary(graph)
    quant_plan = pipe.get_quant_plan()
    prefetch_schedule = pipe.get_prefetch_schedule()
    prefetch_count = pipe.get_prefetch_entry_count()

    coverage = CoverageAnalyzer(graph, prefetch_schedule)
    coverage_report = coverage.coverage_report()

    # Count quant tiers
    from collections import Counter
    prec_counts = Counter(quant_plan.values())
    total_experts = sum(prec_counts.values())

    # Compute activation-weighted higher-prec mass
    with open(args.profile, encoding="utf-8") as f:
        pdata = json.load(f)

    total_freq = 0.0
    higher_prec_freq = 0.0
    for (layer_idx, expert_idx), prec in quant_plan.items():
        stats = pdata["layers"].get(str(layer_idx), {}).get("expert_stats", [])
        if expert_idx < len(stats):
            freq = stats[expert_idx]["activation_freq"]
            total_freq += freq
            if prec != "INT4":
                higher_prec_freq += freq

    quant_diff = higher_prec_freq / total_freq if total_freq > 0 else 0.0

    # ── Print diagnostic ─────────────────────────────────────────────────
    print()
    print("=" * 66)
    print(f"  R-PGO Deployment Diagnostic")
    print(f"  Model: {profile.model_id}")
    print(f"  Compiled in {compile_time:.2f}s ({summary['total_nodes']} nodes, "
          f"{summary['total_edges']:,} edges)")
    print("=" * 66)
    print()

    # Architecture
    print(f"  Architecture:    {n_experts} experts/layer, top-{top_k}, "
          f"{len(layer_indices)} MoE layers")
    print(f"  Routing entropy: {summary['avg_entropy']:.3f} nats "
          f"(max possible: {_max_entropy(n_experts):.3f})")
    print(f"  Normalized:      {summary['avg_entropy'] / _max_entropy(n_experts):.1%} of maximum")
    print()

    # Quantization decision
    print("  ── Quantization Plan (Pass B) ──")
    bf16 = prec_counts.get("BF16", 0)
    int8 = prec_counts.get("INT8", 0)
    int4 = prec_counts.get("INT4", 0)
    print(f"  BF16 (hot, GPU-resident):  {bf16:>5} experts ({bf16/total_experts:.1%})")
    print(f"  INT8 (warm, prefetchable): {int8:>5} experts ({int8/total_experts:.1%})")
    print(f"  INT4 (cold, compressed):   {int4:>5} experts ({int4/total_experts:.1%})")
    print(f"  Quant differentiation:     {quant_diff:.1%} of activation mass at higher precision")
    print()

    # Prefetch decision
    print("  ── Prefetch Schedule (Pass C) ──")
    print(f"  Static entries:     {prefetch_count}")
    print(f"  Coverage ratio:     {coverage_report['coverage_ratio']:.1%}")
    print(f"  Interpretation:     {coverage_report['interpretation']}")
    print()

    # Prediction
    print("  ── Compile-Time Prediction ──")
    print()

    quant_benefit = _predict_quant_benefit(quant_diff, summary["avg_entropy"], n_experts)
    prefetch_benefit = _predict_prefetch_benefit(coverage_report["coverage_ratio"], prefetch_count)

    if quant_benefit == "HIGH":
        print("  QUANTIZATION:  HIGH benefit expected")
        print(f"    → {quant_diff:.1%} of activation mass protected at higher precision")
        print(f"    → Expected recovery: 50-70% of uniform INT4 degradation")
        print(f"    → RECOMMENDATION: Deploy R-PGO mixed precision")
    elif quant_benefit == "MODERATE":
        print("  QUANTIZATION:  MODERATE benefit expected")
        print(f"    → {quant_diff:.1%} of activation mass differentiated")
        print(f"    → Expected recovery: 20-50% of uniform INT4 degradation")
        print(f"    → RECOMMENDATION: Test on representative data before deploying")
    else:
        print("  QUANTIZATION:  MINIMAL benefit expected")
        print(f"    → Only {quant_diff:.1%} differentiation (routing is nearly uniform)")
        print(f"    → RECOMMENDATION: Use standard uniform quantization (GPTQ/AWQ)")

    print()

    if prefetch_benefit == "HIGH":
        print("  PREFETCHING:   HIGH benefit expected")
        print(f"    → {coverage_report['coverage_ratio']:.1%} of activations anticipated")
        print(f"    → RECOMMENDATION: Deploy static prefetch schedule")
    elif prefetch_benefit == "MODERATE":
        print("  PREFETCHING:   MODERATE benefit expected")
        print(f"    → {coverage_report['coverage_ratio']:.1%} coverage")
        print(f"    → RECOMMENDATION: Deploy prefetch, expect partial latency hide")
    else:
        print("  PREFETCHING:   MINIMAL benefit expected")
        print(f"    → {coverage_report['coverage_ratio']:.1%} coverage (routing too distributed)")
        print(f"    → RECOMMENDATION: Runtime adaptive prefetching may be more effective")

    print()
    print("=" * 66)
    print()

    # ── Optional: emit Triton kernels ────────────────────────────────────
    if args.emit_kernels:
        from rpgo.codegen import emit_prefetch_kernels

        # Build artifact dict from pipeline output
        artifact = {
            "model_id": profile.model_id,
            "quant_assignments": {f"{k[0]}:{k[1]}": v for k, v in quant_plan.items()},
            "layout_placements": {f"{k[0]}:{k[1]}": v for k, v in pipe.get_layout_plan().items()},
            "specialization_decisions": pipe.get_specialization_plan(),
            "prefetch_entry_count": prefetch_count,
        }

        kernel_path = emit_prefetch_kernels(
            artifact_path=None,
            output_dir=args.output_dir,
            profile_meta={"compile_time_sec": compile_time},
        ) if False else None  # Need artifact as file

        # Write artifact to temp, then emit
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(artifact, f, indent=2)
            tmp_path = f.name

        kernel_path = emit_prefetch_kernels(
            tmp_path, args.output_dir,
            profile_meta={"compile_time_sec": compile_time},
        )
        print(f"  Triton kernels emitted to: {kernel_path}")
        print(f"  Manifest: {Path(args.output_dir) / 'rpgo_manifest.json'}")
        Path(tmp_path).unlink()


def _max_entropy(n_experts: int) -> float:
    import math
    return math.log(n_experts) if n_experts > 1 else 1.0


def _predict_quant_benefit(quant_diff: float, entropy: float, n_experts: int) -> str:
    """Predict quantization benefit from compile-time metrics."""
    import math
    normalized_entropy = entropy / math.log(n_experts) if n_experts > 1 else 1.0

    if quant_diff > 0.40:
        return "HIGH"
    elif quant_diff > 0.10 or normalized_entropy < 0.85:
        return "MODERATE"
    else:
        return "MINIMAL"


def _predict_prefetch_benefit(coverage: float, n_entries: int) -> str:
    """Predict prefetch benefit from coverage metric."""
    if coverage > 0.60:
        return "HIGH"
    elif coverage > 0.20 or n_entries > 500:
        return "MODERATE"
    else:
        return "MINIMAL"


if __name__ == "__main__":
    main()
