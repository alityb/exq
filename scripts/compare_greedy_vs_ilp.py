#!/usr/bin/env python3
"""Greedy vs ILP joint scheduler comparison.

Same model, same memory budget, same evaluation. Shows whether ILP
produces measurably better assignments than the greedy frequency tiers.

Usage:
    python scripts/compare_greedy_vs_ilp.py \
        --model allenai/OLMoE-1B-7B-0924 \
        --profile profiles/olmoe-1b-7b-0924-256.json \
        --greedy-artifact artifacts/olmoe-1b-7b-0924-256.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from exq.model_utils import load_artifact, parse_moe_assignments


def run_ppl_eval(model_id: str, precision: str, artifact_path: str, dataset: str = "wikitext2") -> float | None:
    """Run eval_ppl.py and return PPL, or None on failure."""
    cmd = [
        sys.executable, "scripts/eval_ppl.py",
        "--model", model_id, "--precision", precision,
        "--dataset", dataset, "--quant-plan", artifact_path,
    ]
    print(f"  Running: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return None
    try:
        payload = json.loads(result.stdout.strip().split("\n")[-1])
        return payload["perplexity"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def artifact_stats(path: str) -> tuple[dict[str, int], float]:
    art = load_artifact(path)
    raw = art.get("quant_assignments") or art.get("quant_plan", {})
    counts = Counter(raw.values())
    total = sum(counts.values())
    mem_units = {"BF16": 4, "INT8": 2, "INT4": 1}
    weighted = sum(counts.get(p, 0) * mem_units.get(p, 1) for p in mem_units)
    ratio = weighted / (total * 4) if total > 0 else 1.0
    return dict(counts), ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--greedy-artifact", required=True)
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--memory-budget-units", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--output", default="results/greedy_vs_ilp.json")
    args = parser.parse_args()

    datasets = args.datasets or [args.dataset]
    greedy_art = load_artifact(args.greedy_artifact)
    is_dense = greedy_art.get("type") == "dense_head_quant"

    # Auto memory budget from greedy artifact
    budget = args.memory_budget_units
    if budget is None and not is_dense:
        greedy_counts, _ = artifact_stats(args.greedy_artifact)
        mem_units = {"BF16": 4, "INT8": 2, "INT4": 1}
        budget = sum(greedy_counts.get(p, 0) * mem_units.get(p, 1) for p in mem_units)
        print(f"Auto memory budget from greedy artifact: {budget} units")

    # Run ILP
    ilp_artifact = args.greedy_artifact.replace(".json", "_ilp.json")
    ilp_cmd = [
        sys.executable, "scripts/solve_joint.py",
        "--profile", args.profile, "--output", ilp_artifact,
        "--time-limit", str(args.time_limit),
    ]
    if budget is not None:
        ilp_cmd += ["--memory-budget-units", str(budget)]

    print("  Running ILP joint scheduler...")
    t0 = time.perf_counter()
    r = subprocess.run(ilp_cmd, capture_output=True, text=True, timeout=120)
    print(f"  ILP solved in {time.perf_counter() - t0:.1f}s")
    ilp_payload = json.loads(r.stdout) if r.returncode == 0 else {}

    if not ilp_payload or not Path(ilp_artifact).exists():
        print("ILP failed. Exiting.")
        return

    # Evaluate
    precision = "rpgo_dense" if is_dense else "rpgo"
    greedy_results, ilp_results = {}, {}
    for ds in datasets:
        print(f"\n--- {ds} ---")
        greedy_results[ds] = run_ppl_eval(args.model, precision, args.greedy_artifact, ds)
        ilp_results[ds] = run_ppl_eval(args.model, precision, ilp_artifact, ds)

    greedy_counts, greedy_ratio = artifact_stats(args.greedy_artifact)
    ilp_counts, ilp_ratio = artifact_stats(ilp_artifact)

    # Print table
    print("\n" + "=" * 70)
    print(f"  Greedy vs ILP: {args.model}")
    print("=" * 70)
    print(f"  {'Metric':<28} {'Greedy':>14} {'ILP':>14} {'Delta':>10}")
    print("  " + "-" * 64)
    for ds in datasets:
        g, i = greedy_results.get(ds), ilp_results.get(ds)
        if g is not None and i is not None:
            print(f"  {ds + ' PPL':<28} {g:>14.4f} {i:>14.4f} {i - g:>+10.4f}")
    print(f"  {'Memory ratio (vs bf16)':<28} {greedy_ratio:>14.3f} {ilp_ratio:>14.3f} {ilp_ratio - greedy_ratio:>+10.3f}")
    print(f"\n  Greedy plan: {greedy_counts}")
    print(f"  ILP plan:    {ilp_counts}")
    print(f"  ILP status:  {ilp_payload.get('status', '?')} (obj={ilp_payload.get('objective_value', '?')})")
    print("=" * 70)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model, "greedy_artifact": args.greedy_artifact,
            "ilp_artifact": ilp_artifact, "greedy_counts": greedy_counts,
            "ilp_counts": ilp_counts, "greedy_ppl": greedy_results,
            "ilp_ppl": ilp_results, "ilp_status": ilp_payload.get("status"),
            "ilp_objective": ilp_payload.get("objective_value"),
            "memory_budget_units": budget,
        }, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
