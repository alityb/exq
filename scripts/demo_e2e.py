#!/usr/bin/env python3
"""R-PGO End-to-End Demo: Profile → Compile → Execute → Measure.

This script demonstrates the complete R-PGO compiler pipeline on a real model,
producing actual compiled inference with measurable performance characteristics.

Usage:
    python scripts/demo_e2e.py --model allenai/OLMoE-1B-7B-0924
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="R-PGO End-to-End Compiler Demo")
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--profile", default=None, help="Pre-computed profile (skip profiling)")
    parser.add_argument("--artifact", default=None, help="Pre-compiled artifact (skip compilation)")
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--prompt", default="The key insight behind mixture of experts is")
    args = parser.parse_args()

    print("=" * 70)
    print("  R-PGO: Routing-Profile-Guided Optimization — End-to-End Demo")
    print("=" * 70)
    print()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Profile (or load existing)
    # ══════════════════════════════════════════════════════════════════════
    profile_path = args.profile
    if profile_path is None:
        print("Phase 1: PROFILING")
        print(f"  Model: {args.model}")
        print("  Collecting routing statistics from calibration data...")
        # Use existing profile if available
        slug = args.model.replace("/", "-").lower()
        for candidate in Path("profiles").glob("*.json"):
            with open(candidate) as f:
                data = json.load(f)
            if data.get("model_id", "").lower().replace("/", "-") == slug:
                profile_path = str(candidate)
                print(f"  Found existing profile: {candidate}")
                break

        if profile_path is None:
            print("  No existing profile found. Run scripts/profile_model.py first.")
            return
    else:
        print(f"Phase 1: PROFILE (loaded from {profile_path})")

    print()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Compile
    # ══════════════════════════════════════════════════════════════════════
    artifact_path = args.artifact
    slug = args.model.replace("/", "-").lower()

    if artifact_path is None:
        print("Phase 2: COMPILATION")
        from rpgo._core import CompilerPipeline, RoutingProfile, py_build_routing_graph

        t0 = time.perf_counter()
        profile = RoutingProfile.load(profile_path)
        graph = py_build_routing_graph(profile)

        layer_indices = profile.moe_layer_indices()
        first_layer = profile.get_layer(layer_indices[0])

        pipe = CompilerPipeline()
        pipe.run_auto(graph, first_layer.n_experts, first_layer.top_k)
        compile_time = time.perf_counter() - t0

        # Save artifact
        quant_plan = pipe.get_quant_plan()
        layout_plan = pipe.get_layout_plan()
        spec_plan = pipe.get_specialization_plan()

        artifact = {
            "model_id": profile.model_id,
            "quant_assignments": {f"{k[0]}:{k[1]}": v for k, v in quant_plan.items()},
            "layout_placements": {f"{k[0]}:{k[1]}": v for k, v in layout_plan.items()},
            "specialization_decisions": spec_plan,
            "prefetch_entry_count": pipe.get_prefetch_entry_count(),
        }

        artifact_path = f"artifacts/{slug}_e2e.json"
        Path("artifacts").mkdir(exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        from collections import Counter
        prec_counts = Counter(quant_plan.values())

        print(f"  Compiled in {compile_time:.2f}s")
        print(f"  Graph: {graph.n_nodes} nodes, {graph.n_edges:,} edges")
        print(f"  Quant plan: {dict(prec_counts)}")
        print(f"  Artifact: {artifact_path}")
    else:
        print(f"Phase 2: ARTIFACT (loaded from {artifact_path})")

    print()

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Execute compiled inference
    # ══════════════════════════════════════════════════════════════════════
    print("Phase 3: COMPILED INFERENCE")
    print(f"  Loading model: {args.model}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"  Generating: {args.n_tokens} tokens, {args.n_runs} runs")
    print()

    # ── Baseline ──
    print("  --- Baseline (unmodified model) ---")
    times_base = []
    with torch.no_grad():
        # Warmup
        model.generate(**inputs, max_new_tokens=4, do_sample=False)

        for _ in range(args.n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=args.n_tokens, do_sample=False)
            torch.cuda.synchronize()
            tokens = out.shape[1] - inputs["input_ids"].shape[1]
            times_base.append((time.perf_counter() - t0) / tokens * 1000)

    base_median = sorted(times_base)[len(times_base) // 2]
    print(f"  Baseline TPOT: {base_median:.1f}ms/token")

    # ── R-PGO Compiled ──
    print("  --- R-PGO Compiled (with prefetch scheduling) ---")
    from rpgo.runtime import CompiledInference

    engine = CompiledInference.from_artifact(artifact_path, model, tokenizer)

    times_compiled = []
    with torch.no_grad():
        # Warmup
        model.generate(**inputs, max_new_tokens=4, do_sample=False)

        for _ in range(args.n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=args.n_tokens, do_sample=False)
            torch.cuda.synchronize()
            tokens = out.shape[1] - inputs["input_ids"].shape[1]
            times_compiled.append((time.perf_counter() - t0) / tokens * 1000)

    comp_median = sorted(times_compiled)[len(times_compiled) // 2]
    delta = comp_median - base_median
    print(f"  Compiled TPOT: {comp_median:.1f}ms/token")
    print(f"  Cache stats: {engine.weight_cache.stats}")

    # ══════════════════════════════════════════════════════════════════════
    # Results
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Model:     {args.model}")
    print(f"  Tokens:    {args.n_tokens} per run, {args.n_runs} runs")
    print(f"  Compile:   {compile_time:.2f}s" if 'compile_time' in dir() else "  Compile:   (pre-computed)")
    print()
    print(f"  {'Condition':<25} {'TPOT':>10} {'vs Baseline':>12}")
    print(f"  {'-'*49}")
    print(f"  {'Baseline':<25} {base_median:>9.1f}ms {'—':>12}")
    print(f"  {'R-PGO Compiled':<25} {comp_median:>9.1f}ms {delta:>+11.1f}ms")
    print()

    # Generate sample output
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.n_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  Generated: \"{text[:100]}...\"")
    print()
    print("=" * 70)

    # Save results
    results = {
        "model_id": args.model,
        "baseline_tpot_ms": base_median,
        "compiled_tpot_ms": comp_median,
        "delta_ms": delta,
        "compile_time_sec": compile_time if 'compile_time' in dir() else None,
        "n_tokens": args.n_tokens,
        "n_runs": args.n_runs,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/e2e_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to results/e2e_benchmark.json")


if __name__ == "__main__":
    main()
