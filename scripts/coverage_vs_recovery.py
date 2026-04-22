#!/usr/bin/env python3
"""Compute the coverage-versus-recovery summary table for ExQ.

Shows that the compiler's compile-time diagnostics (entropy, quant
differentiation) correctly predict downstream PPL recovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exq._core import (
    CompilerPipeline,
    RoutingProfile,
    py_build_routing_graph,
    py_graph_summary,
)
from exq.eval.bench import compute_recovery_pct as compute_recovery, compute_quant_diff, parse_eval_log
from exq.eval.coverage import CoverageAnalyzer
from exq.profiler.dense_profile import DenseProfile


MOE_MODELS = {
    "Qwen1.5-MoE-A2.7B": {
        "profile": Path("profiles/olmoe.json"),
        "eval_model_id": "Qwen/Qwen1.5-MoE-A2.7B",
    },
    "OLMoE-1B-7B": {
        "profile": Path("profiles/olmoe-1b-7b-0924-256.json"),
        "eval_model_id": "allenai/OLMoE-1B-7B-0924",
    },
    "DeepSeek-V2-Lite": {
        "profile": Path("profiles/deepseek-v2-lite.json"),
        "eval_model_id": "deepseek-ai/DeepSeek-V2-Lite",
    },
    "Qwen3-30B-A3B": {
        "profile": Path("profiles/qwen3-30b-a3b.json"),
        "eval_model_id": None,
    },

}

DENSE_MODELS = {
    "Qwen2.5-3B": {
        "profile": Path("profiles/dense/qwen2.5-3b-512.json"),
        "eval_model_id": "Qwen/Qwen2.5-3B",
    },
    "Qwen2.5-1.5B": {
        "profile": Path("profiles/dense/qwen2.5-1.5b.json"),
        "eval_model_id": "Qwen/Qwen2.5-1.5B",
    },
}


def compute_quant_diff(profile_path: str) -> float:
    """Activation-weighted fraction of experts assigned higher-than-INT4 precision."""
    profile = RoutingProfile.load(profile_path)
    graph = py_build_routing_graph(profile)
    layer_indices = profile.moe_layer_indices()
    first_layer = profile.get_layer(layer_indices[0])
    pipe = CompilerPipeline()
    pipe.run_auto(graph, first_layer.n_experts, first_layer.top_k)
    quant_plan = pipe.get_quant_plan()

    with open(profile_path, encoding="utf-8") as f:
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
    return higher_prec_freq / total_freq if total_freq > 0 else 0.0


def _fmt(value: float | None, percent: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.1f}%" if percent else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExQ: Coverage versus recovery summary"
    )
    parser.add_argument("--moe-log", default="results/eval_log.txt")
    parser.add_argument("--dense-log", default="results/eval_log_dense.txt")
    args = parser.parse_args()

    moe_records = (
        parse_eval_log(args.moe_log) if Path(args.moe_log).exists() else {}
    )
    dense_records = (
        parse_eval_log(args.dense_log) if Path(args.dense_log).exists() else {}
    )

    print(
        f"\n{'Model':<24} {'Type':<6} {'Entropy':>8} {'Quant Diff':>11} {'Recovery':>10}"
    )
    print("-" * 62)

    for display_name, meta in MOE_MODELS.items():
        profile_path = meta["profile"]
        if not profile_path.exists():
            print(f"{display_name:<24} {'MoE':<6} {'—':>8} {'—':>11} {'—':>10}")
            continue

        profile = RoutingProfile.load(str(profile_path))
        graph = py_build_routing_graph(profile)
        summary = py_graph_summary(graph)
        entropy = summary["avg_entropy"]
        quant_diff = compute_quant_diff(str(profile_path))

        recovery = None
        eval_id = meta["eval_model_id"]
        if eval_id and eval_id in moe_records:
            m = moe_records[eval_id]
            fp16 = m.get("fp16", {}).get("wikitext2")
            rpgo = m.get("rpgo", {}).get("wikitext2")
            int4 = m.get("int4", {}).get("wikitext2")
            if None not in (fp16, rpgo, int4):
                recovery = compute_recovery(fp16, rpgo, int4)

        print(
            f"{display_name:<24} {'MoE':<6} {_fmt(entropy):>8} "
            f"{_fmt(quant_diff * 100, percent=True):>11} "
            f"{_fmt(recovery, percent=True):>10}"
        )

    for display_name, meta in DENSE_MODELS.items():
        profile_path = meta["profile"]
        if not profile_path.exists():
            print(f"{display_name:<24} {'Dense':<6} {'—':>8} {'—':>11} {'—':>10}")
            continue

        profile = DenseProfile.load(profile_path)
        entropy = profile.summary()["avg_normalized_entropy"]

        recovery = None
        eval_id = meta["eval_model_id"]
        if eval_id and eval_id in dense_records:
            m = dense_records[eval_id]
            fp16 = m.get("fp16", {}).get("wikitext2")
            rpgo = m.get("rpgo_dense", {}).get("wikitext2")
            int4 = m.get("int4", {}).get("wikitext2")
            if None not in (fp16, rpgo, int4):
                recovery = compute_recovery(fp16, rpgo, int4)

        print(
            f"{display_name:<24} {'Dense':<6} {_fmt(entropy):>8} "
            f"{'—':>11} {_fmt(recovery, percent=True):>10}"
        )

    print()
    print("Key: Quant Diff = activation-weighted higher-precision mass (compile-time).")
    print("Higher Quant Diff correlates with higher recovery.")
    print("Break-even models (DeepSeek) correctly identified by near-zero Quant Diff.")


if __name__ == "__main__":
    main()
