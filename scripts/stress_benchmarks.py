#!/usr/bin/env python3
"""Stress-test ExQ benchmark stability and artifact determinism."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test latency and artifact stability")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-sizes", default="1,4")
    args = parser.parse_args()

    results = []
    for i in range(args.repeats):
        cmd = [
            sys.executable,
            "scripts/bench_latency.py",
            "--model",
            args.model,
            "--profile",
            args.profile,
            "--artifact",
            args.artifact,
            "--batch-sizes",
            args.batch_sizes,
            "--n-tokens",
            "16",
            "--n-runs",
            "5",
            "--warmup",
            "2",
        ]
        subprocess.run(cmd, check=True)
        payload = json.loads(Path("results/latency_benchmark.json").read_text(encoding="utf-8"))
        results.append(payload)

    summary = {}
    for batch_size, _ in results[0]["batch_results"].items():
        vals = [r["batch_results"][batch_size]["runtime_predictor"]["p50_ms"] - r["batch_results"][batch_size]["baseline"]["p50_ms"] for r in results]
        summary[batch_size] = {
            "mean_overhead_ms": statistics.mean(vals),
            "stdev_overhead_ms": statistics.pstdev(vals),
            "min_overhead_ms": min(vals),
            "max_overhead_ms": max(vals),
        }

    output = {
        "model": args.model,
        "repeats": args.repeats,
        "summary": summary,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/stress_benchmarks.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
