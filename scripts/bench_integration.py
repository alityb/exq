#!/usr/bin/env python3
"""Integration benchmark: fp16 vs INT4 vs ExQ patched.

Usage:
    python scripts/bench_integration.py \
        --model Qwen/Qwen2.5-3B \
        --artifact artifacts/dense/qwen2.5-3b-512.json \
        --n-tokens 64 --n-runs 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from exq.eval.bench import measure_tpot
from exq.model_utils import fix_tokenizer, load_model_and_tokenizer
from exq.runtime.transformers_integration import load_exq_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--n-tokens", type=int, default=64)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--prompt", default="The following is a detailed analysis of")
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--skip-int4", action="store_true")
    parser.add_argument("--output", default="results/integration_benchmark.json")
    args = parser.parse_args()

    results_all: dict[str, dict] = {}
    measure_kwargs = dict(prompt=args.prompt, n_tokens=args.n_tokens, n_runs=args.n_runs, warmup=args.warmup)

    if not args.skip_fp16:
        print("=== Condition A: fp16 baseline ===")
        model, tokenizer = load_model_and_tokenizer(args.model)
        r = measure_tpot(model, tokenizer, **measure_kwargs)
        results_all["fp16"] = r
        print(f"  TPOT p50: {r['tpot_p50']:.2f}ms  memory: {r['memory_gb']:.2f}GB")
        del model
        torch.cuda.empty_cache()

    if not args.skip_int4:
        print("=== Condition B: uniform INT4 (bitsandbytes) ===")
        model, tokenizer = load_model_and_tokenizer(args.model, load_in_4bit=True)
        r = measure_tpot(model, tokenizer, **measure_kwargs)
        results_all["int4"] = r
        print(f"  TPOT p50: {r['tpot_p50']:.2f}ms  memory: {r['memory_gb']:.2f}GB")
        del model
        torch.cuda.empty_cache()

    print("=== Condition C: ExQ patched ===")
    model_exq, tokenizer_exq = load_exq_model(args.model, args.artifact, torch_dtype=torch.float16)
    r = measure_tpot(model_exq, tokenizer_exq, **measure_kwargs)
    results_all["exq"] = r
    print(f"  TPOT p50: {r['tpot_p50']:.2f}ms  memory: {r['memory_gb']:.2f}GB")
    del model_exq
    torch.cuda.empty_cache()

    # Summary
    print()
    print("=" * 74)
    print(f"  Integration Benchmark: {args.model}")
    print(f"  Runs: {args.n_runs} | Tokens: {args.n_tokens}")
    print("=" * 74)
    print(f"  {'Condition':<22} {'TPOT P50':>10} {'+-CI95':>8} {'P99':>10} {'Mem GB':>8}")
    print("  " + "-" * 60)
    for name, key in [("fp16 baseline", "fp16"), ("uniform INT4", "int4"), ("ExQ patched", "exq")]:
        if key not in results_all:
            continue
        r = results_all[key]
        print(f"  {name:<22} {r['tpot_p50']:>9.2f}ms {r['tpot_ci95']:>7.2f}ms "
              f"{r['tpot_p99']:>9.2f}ms {r['memory_gb']:>7.2f}GB")

    if "int4" in results_all:
        d = results_all["exq"]["tpot_p50"] - results_all["int4"]["tpot_p50"]
        print(f"\n  ExQ vs INT4 TPOT: {d:+.2f}ms ({'faster' if d < 0 else 'slower'})")
    print("=" * 74)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"model": args.model, "artifact": args.artifact, **results_all}, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
