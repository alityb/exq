#!/usr/bin/env python3
"""
Task 4 benchmark: Triton baseline vs ExQ frequency-aware kernel.

Tests three conditions on OLMoE-1B-7B expert dispatch:
  A) Triton baseline  — sorted dispatch, uniform tile sizes (BLOCK_M=64)
  B) ExQ Triton     — sorted dispatch, per-expert tile sizes from profile

Reports:
  - P50/P95/P99 latency with 95% CI
  - Speedup of B over A
  - Go/no-go decision: ≥5% → continue to mixed precision (Task 5)

Usage:
  python scripts/bench_exq_triton.py
  python scripts/bench_exq_triton.py --artifact artifacts/olmoe-1b-7b-0924-256.json
  python scripts/bench_exq_triton.py --batch 1 --seqlen 1   # decode
  python scripts/bench_exq_triton.py --batch 8 --seqlen 128 # prefill
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from exq.kernels.moe_grouped_gemm import moe_grouped_gemm, unsort_output
from exq.kernels.moe_exq_kernel import rpgo_moe_forward
from exq.kernels.exq_artifact import load_exq_artifact, print_profile_summary


def benchmark_fn(fn, n_warmup: int = 20, n_runs: int = 100) -> dict:
    """Warm-up then time `fn`, returning latency statistics in ms."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    n = len(times)
    ci95 = 1.96 * statistics.stdev(times) / (n ** 0.5)
    return {
        "p50":  times[n // 2],
        "p75":  times[int(0.75 * (n - 1))],
        "p95":  times[int(0.95 * (n - 1))],
        "p99":  times[int(0.99 * (n - 1))],
        "mean": statistics.mean(times),
        "std":  statistics.stdev(times),
        "ci95": ci95,
        "n_runs": n,
    }


def verify_correctness(out_a: torch.Tensor, out_b: torch.Tensor,
                        label: str, atol: float = 0.05) -> bool:
    """Compare two output tensors; print result."""
    max_diff = (out_a.float() - out_b.float()).abs().max().item()
    mean_diff = (out_a.float() - out_b.float()).abs().mean().item()
    ok = max_diff < atol
    mark = "PASS ✓" if ok else "FAIL ✗"
    print(f"  {label}: max_diff={max_diff:.5f}  mean_diff={mean_diff:.6f}  {mark}")
    return ok


def print_table(results: dict, baseline_key: str = "triton_baseline") -> None:
    base_p50 = results[baseline_key]["p50"]
    labels = {
        "triton_baseline": "Triton baseline (uniform tiles)",
        "rpgo_triton":     "ExQ Triton (freq-aware tiles)",
    }
    print(f"\n{'Kernel':<34} {'P50':>8} {'±CI95':>7} {'P95':>8} {'P99':>8} {'vs baseline':>12}")
    print("-" * 82)
    for key, r in results.items():
        label = labels.get(key, key)
        if key == baseline_key:
            vs_str = "—"
        else:
            delta_pct = (base_p50 - r["p50"]) / base_p50 * 100
            vs_str = f"{delta_pct:+.1f}%"
        print(f"  {label:<32} {r['p50']:>7.3f}ms "
              f"{r['ci95']:>6.3f}ms "
              f"{r['p95']:>7.3f}ms "
              f"{r['p99']:>7.3f}ms "
              f"{vs_str:>12}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ExQ Triton kernel benchmark")
    parser.add_argument("--artifact", default="artifacts/olmoe-1b-7b-0924-256.json")
    parser.add_argument("--profile",  default="profiles/olmoe-1b-7b-0924-256.json")
    parser.add_argument("--batch",    type=int, default=8)
    parser.add_argument("--seqlen",   type=int, default=64)
    parser.add_argument("--n-runs",   type=int, default=100)
    parser.add_argument("--hidden",   type=int, default=2048)
    parser.add_argument("--inter",    type=int, default=1024)
    parser.add_argument("--n-experts",type=int, default=64)
    parser.add_argument("--top-k",    type=int, default=2)
    parser.add_argument("--layer-idx",type=int, default=0,
                        help="Which layer's profile to use for tile config")
    parser.add_argument("--save",     default="results/triton_benchmark.json")
    args = parser.parse_args()

    HIDDEN    = args.hidden
    INTER     = args.inter
    N_EXPERTS = args.n_experts
    TOP_K     = args.top_k
    N_TOKENS  = args.batch * args.seqlen

    print("=" * 70)
    print("ExQ Triton Kernel Benchmark")
    print("=" * 70)
    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"Tokens:    {N_TOKENS}  (batch={args.batch}, seqlen={args.seqlen})")
    print(f"Dims:      hidden={HIDDEN}, inter={INTER}")
    print(f"Experts:   {N_EXPERTS}, top-k={TOP_K}")
    print(f"Artifact:  {args.artifact}")
    print(f"Profile:   {args.profile}")
    print()

    # ── Load ExQ profile ──────────────────────────────────────────────────
    if not Path(args.artifact).exists():
        print(f"ERROR: artifact not found: {args.artifact}")
        sys.exit(1)

    profile = load_exq_artifact(args.artifact, args.profile)
    print_profile_summary(profile)
    print()

    # ── Test tensors ────────────────────────────────────────────────────────
    torch.manual_seed(42)
    hidden   = torch.randn(N_TOKENS, HIDDEN,         dtype=torch.float16, device="cuda")
    weights  = torch.randn(N_EXPERTS, INTER, HIDDEN, dtype=torch.float16, device="cuda")
    r_idx    = torch.randint(0, N_EXPERTS, (N_TOKENS, TOP_K), device="cuda")
    r_logits = torch.randn(N_TOKENS, TOP_K, device="cuda")
    r_wts    = torch.softmax(r_logits, dim=-1).half()

    # ── Correctness check ────────────────────────────────────────────────────
    print("Correctness verification:")
    out_baseline = moe_grouped_gemm(
        hidden, weights, r_idx, N_EXPERTS,
        block_m=64, block_n=64, block_k=32,
    )
    out_rpgo = rpgo_moe_forward(
        hidden, weights, r_idx, profile, layer_idx=args.layer_idx,
    )
    ok = verify_correctness(out_baseline, out_rpgo,
                             "ExQ vs baseline (sorted output)")
    if not ok:
        print("\n  WARNING: outputs diverge. Check kernel implementation.")
    print()

    # ── Benchmark ────────────────────────────────────────────────────────────
    print(f"Benchmarking ({args.n_runs} runs, 20 warmup)...")
    results: dict[str, dict] = {}

    results["triton_baseline"] = benchmark_fn(
        lambda: moe_grouped_gemm(hidden, weights, r_idx, N_EXPERTS,
                                  block_m=64, block_n=64, block_k=32),
        n_runs=args.n_runs,
    )

    results["rpgo_triton"] = benchmark_fn(
        lambda: rpgo_moe_forward(hidden, weights, r_idx, profile,
                                  layer_idx=args.layer_idx),
        n_runs=args.n_runs,
    )

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\nResults: OLMoE expert GEMM, batch={args.batch}, seqlen={args.seqlen}")
    print("=" * 70)
    print_table(results)

    # ── Go/no-go decision ────────────────────────────────────────────────────
    base_p50  = results["triton_baseline"]["p50"]
    rpgo_p50  = results["rpgo_triton"]["p50"]
    delta_pct = (base_p50 - rpgo_p50) / base_p50 * 100

    print("=" * 70)
    print(f"ExQ tile sizing vs baseline: {delta_pct:+.1f}% at batch={args.batch}")
    print()

    if delta_pct >= 5.0:
        decision = ("CONTINUE → tile sizing shows ≥5% gain. "
                    "Proceed to mixed precision (Task 5).")
    elif delta_pct >= 0.0:
        decision = ("NEUTRAL → tile sizing: <5% gain. "
                    "Check if kernel-launch overhead dominates. "
                    "Mixed precision (Task 5) may still show gains via "
                    "weight bandwidth reduction.")
    else:
        decision = ("REGRESSION → ExQ is slower. "
                    "Per-expert kernel dispatch overhead exceeds tile gains. "
                    "Consider fused dispatch or larger batch sizes.")

    print(f"Decision: {decision}")
    print()

    # ── Batch sweep ──────────────────────────────────────────────────────────
    print("Batch sweep (P50 ms):")
    sweep_results = {}
    for bsz in [1, 2, 4, 8, 16]:
        nt = bsz * args.seqlen
        h  = torch.randn(nt, HIDDEN, dtype=torch.float16, device="cuda")
        ri = torch.randint(0, N_EXPERTS, (nt, TOP_K), device="cuda")

        t_base = benchmark_fn(
            lambda h=h, ri=ri: moe_grouped_gemm(
                h, weights, ri, N_EXPERTS, block_m=64, block_n=64, block_k=32),
            n_runs=50,
        )
        t_rpgo = benchmark_fn(
            lambda h=h, ri=ri: rpgo_moe_forward(
                h, weights, ri, profile, layer_idx=args.layer_idx),
            n_runs=50,
        )
        delta = (t_base["p50"] - t_rpgo["p50"]) / t_base["p50"] * 100
        sweep_results[bsz] = {"baseline": t_base, "rpgo": t_rpgo, "delta_pct": delta}
        print(f"  batch={bsz:3d} ({nt:5d} tok): "
              f"baseline={t_base['p50']:.3f}ms  "
              f"rpgo={t_rpgo['p50']:.3f}ms  "
              f"Δ={delta:+.1f}%")

    # ── Save results ─────────────────────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    output = {
        "config": vars(args),
        "correctness": {"max_diff_vs_baseline": float(
            (out_baseline.float() - out_rpgo.float()).abs().max().item()
        )},
        "results": results,
        "delta_pct_p50": float(delta_pct),
        "decision": decision,
        "batch_sweep": {
            str(bsz): {
                "baseline_p50": v["baseline"]["p50"],
                "rpgo_p50":     v["rpgo"]["p50"],
                "delta_pct":    v["delta_pct"],
            }
            for bsz, v in sweep_results.items()
        },
    }
    Path(args.save).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved: {args.save}")


if __name__ == "__main__":
    main()
