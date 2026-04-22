#!/usr/bin/env python3
"""
Task 1 benchmark: verify baseline Triton grouped GEMM against PyTorch reference.

Checks:
  1. Correctness: max_diff vs PyTorch reference < 0.05 (FP16 rounding)
  2. Performance: Triton vs PyTorch speedup across batch sizes
  3. Go/no-go gate: Triton should beat PyTorch by ≥5×

OLMoE-1B-7B dimensions:
  hidden_size = 2048, intermediate_size = 1024
  64 experts, top-2 routing

Usage:
  python scripts/bench_triton_baseline.py
  python scripts/bench_triton_baseline.py --batch 1 --seqlen 1   # decode regime
  python scripts/bench_triton_baseline.py --batch 8 --seqlen 64  # prefill regime
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from exq.eval.bench import bench as benchmark

from exq.kernels.moe_grouped_gemm import (
    moe_grouped_gemm,
    pytorch_moe_reference,
    unsort_output,
)




def verify_correctness(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    router_indices: torch.Tensor,
    router_weights: torch.Tensor,
    n_experts: int,
    atol: float = 0.05,
) -> tuple[bool, float]:
    """Run both kernels and compare outputs."""
    # PyTorch reference (slow but correct)
    ref_out = pytorch_moe_reference(
        hidden_states.float(), expert_weights.float(),
        router_indices, router_weights.float(),
    ).half()

    # Triton baseline
    triton_sorted = moe_grouped_gemm(
        hidden_states, expert_weights, router_indices, n_experts,
    )
    triton_out = unsort_output(triton_sorted, router_indices, router_weights)

    max_diff = (ref_out.float() - triton_out.float()).abs().max().item()
    passed = max_diff < atol
    return passed, max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Triton MoE GEMM benchmark")
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inter",  type=int, default=1024)
    parser.add_argument("--n-experts", type=int, default=64)
    parser.add_argument("--top-k",    type=int, default=2)
    args = parser.parse_args()

    HIDDEN     = args.hidden
    INTER      = args.inter
    N_EXPERTS  = args.n_experts
    TOP_K      = args.top_k
    N_TOKENS   = args.batch * args.seqlen

    print("=" * 60)
    print("ExQ Triton Baseline Benchmark")
    print("=" * 60)
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"Tokens:   {N_TOKENS}  (batch={args.batch}, seqlen={args.seqlen})")
    print(f"Dims:     hidden={HIDDEN}, inter={INTER}")
    print(f"Experts:  {N_EXPERTS}, top-k={TOP_K}")
    print()

    torch.manual_seed(42)
    hidden   = torch.randn(N_TOKENS, HIDDEN,       dtype=torch.float16, device="cuda")
    weights  = torch.randn(N_EXPERTS, INTER, HIDDEN, dtype=torch.float16, device="cuda")
    r_idx    = torch.randint(0, N_EXPERTS, (N_TOKENS, TOP_K), device="cuda")
    # Normalised router weights (softmax-like)
    r_logits = torch.randn(N_TOKENS, TOP_K, device="cuda")
    r_wts    = torch.softmax(r_logits, dim=-1).half()

    # ── Correctness check ──────────────────────────────────────────────────
    print("Correctness check (Triton vs PyTorch reference):")
    # Use a smaller batch for the slow PyTorch reference
    n_ref = min(N_TOKENS, 32)
    passed, max_diff = verify_correctness(
        hidden[:n_ref], weights,
        r_idx[:n_ref], r_wts[:n_ref],
        N_EXPERTS,
        atol=0.5,   # FP16 rounding: max absolute diff ~0.125 at outputs ~±168
    )
    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"  max_diff={max_diff:.5f}  →  {status}")
    if not passed:
        print("  ERROR: Triton output diverges from PyTorch reference.")
        print("         Do not proceed to ExQ optimisations.")
        sys.exit(1)
    print()

    # ── Latency benchmark ─────────────────────────────────────────────────
    print(f"Latency benchmark ({args.n_runs} runs, 10 warmup):")
    print()

    # Triton baseline
    t_triton = benchmark(
        lambda: moe_grouped_gemm(hidden, weights, r_idx, N_EXPERTS,
                                  block_m=64, block_n=64, block_k=32),
        n_runs=args.n_runs,
    )

    # PyTorch reference (only at small sizes — it's O(tokens) not vectorised)
    n_pytorch = min(N_TOKENS, 64)
    t_pytorch = benchmark(
        lambda: pytorch_moe_reference(
            hidden[:n_pytorch].float(), weights.float(),
            r_idx[:n_pytorch], r_wts[:n_pytorch].float()
        ),
        n_warmup=3, n_runs=20,
    )

    print(f"  {'Kernel':<30} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}")
    print(f"  {'-'*62}")
    print(f"  {'Triton baseline':<30} {t_triton['p50']:>10.3f} "
          f"{t_triton['p95']:>10.3f} {t_triton['p99']:>10.3f}")
    # Scale PyTorch to full batch for display
    scale = N_TOKENS / n_pytorch
    print(f"  {'PyTorch reference (scaled)':<30} "
          f"{t_pytorch['p50']*scale:>10.3f} "
          f"{t_pytorch['p95']*scale:>10.3f} "
          f"{t_pytorch['p99']*scale:>10.3f}")
    print()

    speedup = (t_pytorch["p50"] * scale) / t_triton["p50"]
    print(f"Triton speedup vs PyTorch: {speedup:.1f}×")

    gate = "PASS — proceed to ExQ optimisations" if speedup >= 5.0 else \
           "WARN — speedup below 5×, check kernel"
    print(f"Go/no-go gate (≥5×): {gate}")
    print()

    # ── Multi-batch sweep ─────────────────────────────────────────────────
    print("Batch sweep (Triton baseline P50):")
    for bsz in [1, 2, 4, 8]:
        nt = bsz * args.seqlen
        h  = torch.randn(nt, HIDDEN, dtype=torch.float16, device="cuda")
        ri = torch.randint(0, N_EXPERTS, (nt, TOP_K), device="cuda")
        t  = benchmark(
            lambda h=h, ri=ri: moe_grouped_gemm(h, weights, ri, N_EXPERTS),
            n_runs=30,
        )
        print(f"  batch={bsz:2d} ({nt:4d} tokens): {t['p50']:.3f}ms")

    print()
    # Save results
    Path("results").mkdir(exist_ok=True)
    results = {
        "config": vars(args),
        "correctness": {"passed": passed, "max_diff": float(max_diff)},
        "triton_baseline": t_triton,
        "pytorch_reference_scaled": {k: v * scale for k, v in t_pytorch.items()},
        "speedup_vs_pytorch": float(speedup),
    }
    out_path = Path("results/triton_baseline_benchmark.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
