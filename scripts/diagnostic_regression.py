#!/usr/bin/env python3
"""Calibrated diagnostic regression: quant_diff predicts recovery.

Reads eval logs and fits a linear model: quant_diff -> recovery, where
recovery is the fraction of INT4 degradation that ExQ eliminates.

Usage:
    python scripts/diagnostic_regression.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from exq.eval.bench import compute_recovery, parse_eval_log
from exq.model_utils import load_artifact

ARTIFACT_MAP = {
    "deepseek-ai/DeepSeek-V2-Lite": "artifacts/deepseek-v2-lite.json",
    "allenai/OLMoE-1B-7B-0924": "artifacts/olmoe-1b-7b-0924-256.json",
    "Qwen/Qwen1.5-MoE-A2.7B": "artifacts/qwen15moe_plan.json",
    "zai-org/GLM-4.7-Flash": "artifacts/glm-4-7-flash-reduced8.json",
    "Qwen/Qwen2.5-3B": "artifacts/dense/qwen2.5-3b-512.json",
    "Qwen/Qwen2.5-1.5B": "artifacts/dense/qwen2.5-1.5b.json",
}


def compute_quant_diff(artifact_path: str) -> float | None:
    """Fraction of assignments at higher than INT4 precision."""
    if not artifact_path or not Path(artifact_path).exists():
        return None
    art = load_artifact(artifact_path)
    raw = art.get("quant_assignments") or art.get("quant_plan", {})
    if not raw:
        return None
    total = len(raw)
    higher = sum(1 for v in raw.values() if v in ("BF16", "INT8"))
    return higher / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-log", default="results/eval_log.txt")
    parser.add_argument("--eval-log-dense", default="results/eval_log_dense.txt")
    parser.add_argument("--output", default="results/diagnostic_regression.json")
    args = parser.parse_args()

    moe_data = parse_eval_log(args.eval_log)
    dense_data = parse_eval_log(args.eval_log_dense)

    rows = []
    for source_label, data, rpgo_key in [("MoE", moe_data, "rpgo"), ("Dense", dense_data, "rpgo_dense")]:
        for model_id, precisions in data.items():
            fp16, rpgo, int4 = precisions.get("fp16", {}), precisions.get(rpgo_key, {}), precisions.get("int4", {})
            qd = compute_quant_diff(ARTIFACT_MAP.get(model_id, ""))
            for ds in sorted(set(fp16) & set(rpgo) & set(int4)):
                rec = compute_recovery(fp16[ds], rpgo[ds], int4[ds])
                if rec is None:
                    continue
                rows.append({
                    "model": model_id.split("/")[-1], "model_id": model_id,
                    "type": source_label, "dataset": ds,
                    "fp16_ppl": fp16[ds], "rpgo_ppl": rpgo[ds], "int4_ppl": int4[ds],
                    "recovery": rec, "quant_diff": qd,
                })

    if not rows:
        print("No evaluation data found.")
        return

    # Data table
    print("\n" + "=" * 90)
    print("  ExQ Diagnostic Regression: Calibration Data")
    print("=" * 90 + "\n")
    print(f"  {'Model':<24} {'Type':<6} {'DS':<10} {'fp16':>8} {'rpgo':>8} {'int4':>8} {'Recovery':>9} {'QDiff':>7}")
    print("  " + "-" * 84)
    for r in sorted(rows, key=lambda x: (x["type"], x["model"], x["dataset"])):
        qd = f"{r['quant_diff']:.3f}" if r["quant_diff"] is not None else "  N/A"
        print(f"  {r['model']:<24} {r['type']:<6} {r['dataset']:<10} "
              f"{r['fp16_ppl']:>8.3f} {r['rpgo_ppl']:>8.3f} {r['int4_ppl']:>8.3f} "
              f"{r['recovery']:>8.1%} {qd:>7}")

    # Aggregate per model
    model_agg: dict[str, dict] = {}
    for r in rows:
        key = r["model"]
        if key not in model_agg:
            model_agg[key] = {"type": r["type"], "quant_diff": r["quant_diff"],
                              "recoveries": [], "model_id": r["model_id"]}
        model_agg[key]["recoveries"].append(r["recovery"])
    for v in model_agg.values():
        v["mean_recovery"] = float(np.mean(v["recoveries"]))

    print(f"\n  {'Model':<24} {'Type':<6} {'Mean Recovery':>13} {'QDiff':>7}")
    print("  " + "-" * 52)
    for k, v in sorted(model_agg.items()):
        qd = f"{v['quant_diff']:.3f}" if v["quant_diff"] is not None else "  N/A"
        print(f"  {k:<24} {v['type']:<6} {v['mean_recovery']:>12.1%} {qd:>7}")

    # MoE regression
    moe_pts = [(v["quant_diff"], v["mean_recovery"]) for v in model_agg.values()
               if v["type"] == "MoE" and v["quant_diff"] is not None]

    print("\n" + "=" * 70)
    print("  Regression: quant_diff -> recovery (MoE models)")
    print("=" * 70)

    reg = {}
    if len(moe_pts) >= 2:
        x, y = np.array([p[0] for p in moe_pts]), np.array([p[1] for p in moe_pts])
        n = len(x)
        r_corr = float(np.corrcoef(x, y)[0, 1])
        xm, ym = x.mean(), y.mean()
        ss_xx = float(np.sum((x - xm) ** 2))
        if ss_xx > 0:
            a = float(np.sum((x - xm) * (y - ym)) / ss_xx)
            b = float(ym - a * xm)
            ss_res = float(np.sum((y - (a * x + b)) ** 2))
            ss_tot = float(np.sum((y - ym) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            reg = {"n": int(n), "slope": a, "intercept": b, "r": r_corr, "r2": r2}

            print(f"\n  n={n}  r={r_corr:.3f}  R2={r2:.3f}")
            print(f"  recovery = {a:.3f} * quant_diff + {b:.3f}")
            print(f"\n  {'Model':<24} {'QDiff':>7} {'Predicted':>10} {'Actual':>8}")
            print("  " + "-" * 50)
            for (xi, yi) in zip(x, y):
                print(f"  {'':24} {xi:>7.3f} {a*xi+b:>10.3f} {yi:>8.3f}")
            if n <= 4:
                print(f"\n  n={n}: interpret with caution.")
    else:
        print(f"  Only {len(moe_pts)} MoE data points, need >= 2 for regression.")

    # Dense summary
    dense_pts = [v for v in model_agg.values() if v["type"] == "Dense"]
    if dense_pts:
        print("\n  Dense models:")
        for v in dense_pts:
            print(f"    {v['model_id'].split('/')[-1]}: recovery={v['mean_recovery']:.1%}")

    # Cross-type
    type_stats = defaultdict(list)
    for v in model_agg.values():
        type_stats[v["type"]].append(v["mean_recovery"])
    print(f"\n  Cross-type: ", end="")
    for t, recs in sorted(type_stats.items()):
        print(f"{t} n={len(recs)} mean={np.mean(recs):.1%}  ", end="")
    print("\n" + "=" * 70)

    # Save
    out = {"rows": rows, "model_aggregates": {
        k: {"type": v["type"], "quant_diff": v["quant_diff"], "mean_recovery": v["mean_recovery"]}
        for k, v in model_agg.items()
    }}
    if reg:
        out["moe_regression"] = reg
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
