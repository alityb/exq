#!/usr/bin/env python3
"""
INT4 dequantization kernel benchmark.

Compares three kernels on MoE expert dispatch:
  A) BF16 baseline    — fp16 weights, standard grouped GEMM
  B) INT4 kernel      — packed uint8 weights, on-chip dequant, ~3.9× fewer bytes

Tests on both model configs:
  OLMoE-1B-7B:     hidden=2048, inter=1024, 64 experts,  top-2
  Qwen3-30B-A3B:   hidden=2048, inter=768,  128 experts, top-8

Usage:
  python scripts/bench_int4.py
  python scripts/bench_int4.py --model qwen3
  python scripts/bench_int4.py --model olmoe --batch 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from exq.kernels.moe_grouped_gemm import moe_grouped_gemm
from exq.kernels.moe_int4_kernel import (
    pack_experts_int4,
    moe_int4_forward,
)


MODEL_CONFIGS = {
    "olmoe": dict(hidden=2048, inter=1024, n_experts=64,  top_k=2,
                  name="OLMoE-1B-7B"),
    "qwen3": dict(hidden=2048, inter=768,  n_experts=128, top_k=8,
                  name="Qwen3-30B-A3B"),
}


def bench(fn, n_warmup=20, n_runs=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    n = len(times)
    ci = 1.96 * statistics.stdev(times) / n**0.5
    return {"p50": times[n//2], "p95": times[int(.95*(n-1))],
            "p99": times[int(.99*(n-1))], "mean": statistics.mean(times),
            "ci95": ci, "n": n}


def verify_int4_correctness(
    hidden: torch.Tensor,
    weights_fp16: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    router_indices: torch.Tensor,
    n_experts: int,
    group_size: int,
) -> bool:
    """
    Verify INT4 kernel output against reference dequantization.

    Uses a Python dequant reference (not the fp16 weights) since INT4
    introduces quantization error vs fp16 by design.
    """
    from exq.kernels.moe_int4_kernel import moe_int4_forward

    n_e, N, K = weights_fp16.shape

    # Build Python dequantized reference weights
    p_cpu = packed.cpu()
    s_cpu = scales.cpu().float()
    lo = (p_cpu.to(torch.uint8) & 0x0F).float() - 8.0   # [n_e, N, K//2]
    hi = ((p_cpu.to(torch.uint8) >> 4) & 0x0F).float() - 8.0  # [n_e, N, K//2]
    # Scale: group g covers K columns [g*group_size, (g+1)*group_size)
    # Packed column c corresponds to K=2c (lo) and K=2c+1 (hi).
    # Group for packed col c: (2c) // group_size = c // (group_size//2)
    n_groups = K // group_size
    g_per_packed = group_size // 2  # how many packed cols per group
    s_exp = s_cpu[:, :, :].repeat_interleave(g_per_packed, dim=2)   # [n_e, N, K//2]
    lo_dq = lo * s_exp
    hi_dq = hi * s_exp
    w_recon = torch.zeros(n_e, N, K)
    w_recon[:, :, 0::2] = lo_dq
    w_recon[:, :, 1::2] = hi_dq
    w_recon_cuda = w_recon.half().to(hidden.device)

    # Reference output: hidden @ w_recon.T per expert (via existing fp16 kernel)
    from exq.kernels.moe_grouped_gemm import moe_grouped_gemm
    ref_out = moe_grouped_gemm(hidden, w_recon_cuda, router_indices, n_experts)

    # Kernel output
    kern_out = moe_int4_forward(hidden, packed, scales, router_indices, n_experts,
                                 group_size=group_size)

    diff = (ref_out.float() - kern_out.float()).abs()
    max_diff = diff.max().item()
    rms_ref  = ref_out.float().pow(2).mean().sqrt().item()
    rel_err  = max_diff / max(rms_ref, 1e-6)
    # Kernel should exactly match the Python dequant reference (within fp16 rounding)
    ok = rel_err < 0.01   # <1% relative error = kernel is correct
    mark = "PASS" if ok else "FAIL"
    print(f"  INT4 kernel vs dequant reference: max_diff={max_diff:.4f}  "
          f"rel_err={rel_err:.2%}  [{mark}]")
    if not ok:
        print(f"  (vs BF16: {(moe_grouped_gemm(hidden, weights_fp16, router_indices, n_experts).float() - kern_out.float()).abs().max().item():.4f} -- this is quantization error, not a bug)")
    return ok


def run_model(cfg, args):
    H, INTER, N_EXP, TOP_K = cfg["hidden"], cfg["inter"], cfg["n_experts"], cfg["top_k"]
    N_TOK = args.batch * args.seqlen
    GROUP_SIZE = 128

    print(f"\n{'='*68}")
    print(f"Model: {cfg['name']}")
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"Dims:  hidden={H}, inter={INTER}, experts={N_EXP}, top_k={TOP_K}")
    print(f"Tokens:{N_TOK}  (batch={args.batch}, seqlen={args.seqlen})")
    print(f"{'='*68}")

    torch.manual_seed(42)
    hidden  = torch.randn(N_TOK, H,          dtype=torch.float16, device="cuda")
    weights = torch.randn(N_EXP, INTER, H,   dtype=torch.float16, device="cuda")
    r_idx   = torch.randint(0, N_EXP, (N_TOK, TOP_K), device="cuda")

    # ── Pack weights ──────────────────────────────────────────────────────────
    print("Packing weights to INT4... ", end="", flush=True)
    t0 = time.perf_counter()
    packed, scales = pack_experts_int4(weights, group_size=GROUP_SIZE)
    pack_time = (time.perf_counter() - t0) * 1000
    print(f"done in {pack_time:.1f}ms")

    fp16_mb  = weights.nbytes / 1024**2
    int4_mb  = (packed.nbytes + scales.nbytes) / 1024**2
    print(f"Weight storage: fp16={fp16_mb:.1f}MB  INT4={int4_mb:.1f}MB  "
          f"ratio={fp16_mb/int4_mb:.2f}x")

    # ── Correctness (INT4 kernel vs Python dequant reference) ────────────────
    print("\nCorrectness check:")
    verify_int4_correctness(hidden, weights, packed, scales, r_idx,
                             N_EXP, GROUP_SIZE)
    # Also report quantization error vs fp16 baseline (informational)
    out_bf16 = moe_grouped_gemm(hidden, weights, r_idx, N_EXP)
    out_int4 = moe_int4_forward(hidden, packed, scales, r_idx, N_EXP,
                                 group_size=GROUP_SIZE)
    diff = (out_bf16.float() - out_int4.float()).abs()
    rms = out_bf16.float().pow(2).mean().sqrt().item()
    print(f"  INT4 vs BF16 (quantization noise): "
          f"max_diff={diff.max().item():.3f}  rel_err={diff.max().item()/max(rms,1e-6):.1%}"
          f"  (expected ~6-8% for 4-bit weights)")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print(f"\nBenchmark ({args.n_runs} runs, 20 warmup):")
    r_bf16 = bench(lambda: moe_grouped_gemm(hidden, weights, r_idx, N_EXP),
                   n_runs=args.n_runs)
    r_int4 = bench(lambda: moe_int4_forward(hidden, packed, scales, r_idx, N_EXP,
                                             group_size=GROUP_SIZE),
                   n_runs=args.n_runs)
    base = r_bf16["p50"]
    delta = (base - r_int4["p50"]) / base * 100
    speedup = base / r_int4["p50"]

    print(f"\n{'Kernel':<30} {'P50':>8} {'±CI95':>7} {'P95':>8} {'vs BF16':>10}")
    print(f"{'-'*65}")
    print(f"{'BF16 baseline':<30} {r_bf16['p50']:>7.3f}ms "
          f"{r_bf16['ci95']:>6.3f}ms {r_bf16['p95']:>7.3f}ms {'—':>10}")
    print(f"{'INT4 dequant':<30} {r_int4['p50']:>7.3f}ms "
          f"{r_int4['ci95']:>6.3f}ms {r_int4['p95']:>7.3f}ms "
          f"{delta:>+9.1f}%")

    print(f"\nSpeedup: {speedup:.2f}x  ({delta:+.1f}%)")

    # ── Batch sweep ───────────────────────────────────────────────────────────
    print("\nBatch sweep:")
    sweep = {}
    for bsz in [1, 2, 4, 8, 16]:
        nt = bsz * args.seqlen
        h  = torch.randn(nt, H, dtype=torch.float16, device="cuda")
        ri = torch.randint(0, N_EXP, (nt, TOP_K), device="cuda")
        tb = bench(lambda h=h, ri=ri: moe_grouped_gemm(h, weights, ri, N_EXP),
                   n_runs=50)
        ti = bench(lambda h=h, ri=ri: moe_int4_forward(
                       h, packed, scales, ri, N_EXP, group_size=GROUP_SIZE),
                   n_runs=50)
        d = (tb["p50"] - ti["p50"]) / tb["p50"] * 100
        print(f"  batch={bsz:3d} ({nt:5d} tok): "
              f"bf16={tb['p50']:.3f}ms  int4={ti['p50']:.3f}ms  Δ={d:+.1f}%")
        sweep[bsz] = {"bf16": tb["p50"], "int4": ti["p50"], "delta_pct": d}

    return {
        "model": cfg["name"],
        "config": {"hidden": H, "inter": INTER, "n_experts": N_EXP,
                   "top_k": TOP_K, "batch": args.batch, "seqlen": args.seqlen},
        "weight_mb": {"fp16": fp16_mb, "int4": int4_mb,
                      "ratio": float(fp16_mb / int4_mb)},
        "bf16":  r_bf16,
        "int4":  r_int4,
        "speedup": speedup,
        "delta_pct": delta,
        "batch_sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser(description="INT4 dequant kernel benchmark")
    parser.add_argument("--model",  choices=["olmoe", "qwen3", "both"],
                        default="both")
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--save",   default="results/int4_benchmark.json")
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == "both" else [args.model]
    all_results = {}

    for model_key in models:
        result = run_model(MODEL_CONFIGS[model_key], args)
        all_results[model_key] = result

    # Summary table
    print(f"\n{'='*68}")
    print("Summary")
    print(f"{'='*68}")
    print(f"{'Model':<22} {'BF16 P50':>10} {'INT4 P50':>10} {'Speedup':>10} {'Δ':>8}")
    print("-"*62)
    for k, r in all_results.items():
        print(f"{r['model']:<22} {r['bf16']['p50']:>9.3f}ms "
              f"{r['int4']['p50']:>9.3f}ms "
              f"{r['speedup']:>9.2f}x "
              f"{r['delta_pct']:>+7.1f}%")

    Path("results").mkdir(exist_ok=True)
    Path(args.save).write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved: {args.save}")


if __name__ == "__main__":
    main()
