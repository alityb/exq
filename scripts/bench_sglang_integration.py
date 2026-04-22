#!/usr/bin/env python3
"""
Benchmark: SGLang default kernel vs ExQ backend.

Tests the patched UnquantizedFusedMoEMethod.forward_cuda directly,
simulating the exact call path SGLang uses during inference.

Two conditions:
  A) SGLang default (unquantized fp16 fused_experts)
  B) ExQ patched   (INT4 packed weights, ExQ Triton kernel)

Both conditions use the same dispatch_output object (same token sort,
same topk_ids/topk_weights) so only the GEMM kernel differs.

Usage:
  python scripts/bench_sglang_integration.py
  python scripts/bench_sglang_integration.py --model olmoe
  python scripts/bench_sglang_integration.py --model qwen3 --batch 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import types
from pathlib import Path

import torch


# ── Inject minimal mock ServerArgs so SGLang's MoE kernel config works ────────
# SGLang's fused_moe.py calls get_global_server_args() to check two flags.
# We inject a minimal mock before any SGLang MoE imports happen.
import types
import sglang.srt.server_args as _sa
_sa._global_server_args = types.SimpleNamespace(
    enable_deterministic_inference=False,
    enable_fused_moe_sum_all_reduce=False,
)

# ── Model configs ─────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "olmoe": dict(
        name="OLMoE-1B-7B",
        artifact="artifacts/olmoe-1b-7b-0924-256.json",
        hidden=2048,
        inter=1024,   # intermediate_size
        n_experts=64,
        top_k=2,
    ),
    "qwen3": dict(
        name="Qwen3-30B-A3B",
        artifact="artifacts/qwen3-30b-a3b.json",
        hidden=2048,
        inter=768,
        n_experts=128,
        top_k=8,
    ),
}


# ── Timing helpers ────────────────────────────────────────────────────────────

def bench(fn, n_warmup: int = 20, n_runs: int = 100) -> dict:
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
    return {
        "p50":  times[n // 2],
        "p95":  times[int(0.95 * (n - 1))],
        "p99":  times[int(0.99 * (n - 1))],
        "mean": statistics.mean(times),
        "ci95": 1.96 * statistics.stdev(times) / n**0.5,
        "n":    n,
    }


# ── SGLang mock objects ───────────────────────────────────────────────────────
# Build the exact objects that SGLang's forward_cuda receives so we can
# benchmark the patched method without loading a full SGLang server.

def make_dispatch_output(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    """Build a StandardDispatchOutput that matches SGLang's internal format."""
    try:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=None,  # not used in forward_cuda
        )
        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )
    except ImportError:
        # Fallback: plain namespace
        topk_output = types.SimpleNamespace(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=None
        )
        return types.SimpleNamespace(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )


def make_mock_layer(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    layer_id: int,
) -> object:
    """Build a minimal mock FusedMoE layer object."""
    layer = types.SimpleNamespace(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        layer_id=layer_id,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
    )
    return layer


# ── SGLang default forward_cuda ───────────────────────────────────────────────

def sglang_default_forward(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Call SGLang's fused_experts_impl directly.

    Uses fused_experts_impl (not fused_experts) to bypass the
    server_args requirement. This is the actual GEMM kernel that
    SGLang uses in production.
    """
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_impl

    return fused_experts_impl(
        hidden_states=hidden_states,
        w1=w13_weight,
        w2=w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation="silu",
        is_gated=True,
    )
    config = MoeRunnerConfig(
        num_experts=w13_weight.shape[0],
        num_local_experts=w13_weight.shape[0],
        hidden_size=hidden_states.shape[1],
        intermediate_size_per_partition=w13_weight.shape[1] // 2,
        top_k=topk_ids.shape[1],
        activation="silu",
        is_gated=True,
        inplace=False,
    )
    return fused_experts(
        hidden_states=hidden_states,
        w1=w13_weight,
        w2=w2_weight,
        topk_output=topk_output,
        moe_runner_config=config,
    )


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_model(cfg: dict, args) -> dict:
    H         = cfg["hidden"]
    INTER     = cfg["inter"]
    N_EXP     = cfg["n_experts"]
    TOP_K     = cfg["top_k"]
    N_TOK     = args.batch * args.seqlen
    ARTIFACT  = cfg["artifact"]

    print(f"\n{'='*70}")
    print(f"Model:   {cfg['name']}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Dims:    hidden={H}, inter={INTER}, experts={N_EXP}, top_k={TOP_K}")
    print(f"Tokens:  {N_TOK}  (batch={args.batch}, seqlen={args.seqlen})")
    print(f"Artifact:{ARTIFACT}")
    print(f"{'='*70}")

    if not Path(ARTIFACT).exists():
        print(f"ERROR: artifact not found: {ARTIFACT}")
        print("Run:  python scripts/compile_model.py --profile profiles/... --run-auto")
        return {}

    torch.manual_seed(42)
    # Scale weights to realistic values (model weights are ~0.02 std, not 1.0).
    # Using unit-normal weights causes FP16 overflow in GEMM2 with K=1024
    # since partial sums reach ~60k which is near FP16 max (65504).
    # Real model weights don't have this problem.
    WEIGHT_SCALE = 0.02
    # w13_weight: gate+up fused  [n_experts, 2*inter, hidden]
    w13 = (torch.randn(N_EXP, 2*INTER, H, dtype=torch.float16, device="cuda")
           * WEIGHT_SCALE).contiguous()
    # w2_weight: down projection [n_experts, hidden, inter]
    w2  = (torch.randn(N_EXP, H, INTER,   dtype=torch.float16, device="cuda")
           * WEIGHT_SCALE).contiguous()
    hidden  = torch.randn(N_TOK, H,       dtype=torch.float16, device="cuda").contiguous()
    r_idx   = torch.randint(0, N_EXP, (N_TOK, TOP_K), device="cuda")
    r_logits = torch.randn(N_TOK, TOP_K,  device="cuda")
    r_wts   = torch.softmax(r_logits, dim=-1).half()

    dispatch_out = make_dispatch_output(hidden, r_wts, r_idx)
    layer        = make_mock_layer(w13, w2, layer_id=0)

    # ── Patch SGLang ──────────────────────────────────────────────────────────
    print("\nPatching SGLang with ExQ backend...")
    from exq.runtime.sglang_backend import patch_sglang, _PACKED_CACHE
    backend = patch_sglang(ARTIFACT)
    print(f"Artifact: {backend.n_layers} layers covered")

    # Force weight packing on first call (happens lazily, but we time it)
    t_pack = time.perf_counter()
    from exq.runtime.sglang_backend import _get_or_pack
    _ = _get_or_pack(layer)
    t_pack = (time.perf_counter() - t_pack) * 1000
    print(f"Weight packing (once): {t_pack:.1f}ms  "
          f"(fp16→INT4, cached for all subsequent calls)")

    # ── Correctness ───────────────────────────────────────────────────────────
    print("\nCorrectness check:")
    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)

        # Call the patched method correctly (instance method, self is bound)
        out_exq = method.forward_cuda(layer, dispatch_out)
        exq_hidden = out_exq.hidden_states

        # Baseline: direct fused_experts_impl call
        out_base = sglang_default_forward(hidden, w13, w2, r_wts, r_idx)

        diff = (out_base.float() - exq_hidden.float()).abs()
        rms  = out_base.float().pow(2).mean().sqrt().item()
        rel  = diff.max().item() / max(rms, 1e-6)
        # INT4 introduces ~6-8% quantization error vs fp16 — expected
        print(f"  ExQ vs BF16 baseline: max_diff={diff.max().item():.3f}  "
              f"rel={rel:.1%}  (INT4 quant error ~6-8% expected)")
    except Exception as e:
        import traceback
        print(f"  Correctness check error: {e}")
        traceback.print_exc()

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print(f"\nBenchmarking ({args.n_runs} runs, 20 warmup)...")

    # A) SGLang default
    r_default = bench(
        lambda: sglang_default_forward(hidden, w13, w2, r_wts, r_idx),
        n_runs=args.n_runs,
    )

    # B) ExQ patched forward_cuda
    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        method_obj = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        r_exq = bench(
            lambda: method_obj.forward_cuda(layer, dispatch_out),
            n_runs=args.n_runs,
        )
    except Exception as e:
        import traceback
        print(f"ExQ patched bench failed: {e}")
        traceback.print_exc()
        r_exq = None

    # ── Results table ─────────────────────────────────────────────────────────
    base_p50 = r_default["p50"]
    print(f"\n{'Kernel':<30} {'P50':>8} {'±CI95':>7} {'P95':>8} {'vs default':>12}")
    print("-" * 68)
    print(f"{'SGLang default (fp16)':<30} {r_default['p50']:>7.3f}ms "
          f"{r_default['ci95']:>6.3f}ms {r_default['p95']:>7.3f}ms {'—':>12}")

    if r_exq:
        delta = (base_p50 - r_exq["p50"]) / base_p50 * 100
        speedup = base_p50 / r_exq["p50"]
        print(f"{'ExQ INT4 (patched)':<30} {r_exq['p50']:>7.3f}ms "
              f"{r_exq['ci95']:>6.3f}ms {r_exq['p95']:>7.3f}ms "
              f"{delta:>+11.1f}%")
        print(f"\nSpeedup: {speedup:.2f}x  ({delta:+.1f}%)")
    else:
        delta = 0.0
        speedup = 1.0

    # ── Batch sweep ───────────────────────────────────────────────────────────
    print("\nBatch sweep:")
    sweep = {}
    for bsz in [1, 2, 4, 8, 16]:
        nt = bsz * args.seqlen
        h  = torch.randn(nt, H, dtype=torch.float16, device="cuda")
        ri = torch.randint(0, N_EXP, (nt, TOP_K), device="cuda")
        rl = torch.randn(nt, TOP_K, device="cuda")
        rw = torch.softmax(rl, dim=-1).half()
        do = make_dispatch_output(h, rw, ri)

        tb = bench(lambda h=h, ri=ri, rw=rw: sglang_default_forward(h, w13, w2, rw, ri),
                   n_runs=50)
        try:
            from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
            m2 = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
            ti = bench(lambda m=m2, do=do: m.forward_cuda(layer, do), n_runs=50)
            d  = (tb["p50"] - ti["p50"]) / tb["p50"] * 100
            sweep[bsz] = {"default": tb["p50"], "rpgo": ti["p50"], "delta_pct": d}
            print(f"  batch={bsz:3d} ({nt:5d} tok): "
                  f"default={tb['p50']:.3f}ms  rpgo={ti['p50']:.3f}ms  Δ={d:+.1f}%")
        except Exception as e:
            print(f"  batch={bsz:3d}: {e}")

    return {
        "model": cfg["name"],
        "config": {"hidden": H, "inter": INTER, "n_experts": N_EXP,
                   "top_k": TOP_K, "batch": args.batch, "seqlen": args.seqlen},
        "default":    r_default,
        "rpgo":       r_exq,
        "speedup":    speedup,
        "delta_pct":  delta,
        "batch_sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SGLang default vs ExQ INT4 backend benchmark"
    )
    parser.add_argument("--model",   choices=["olmoe", "qwen3", "both"],
                        default="both")
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--seqlen",  type=int, default=64)
    parser.add_argument("--n-runs",  type=int, default=100)
    parser.add_argument("--save",    default="results/sglang_benchmark.json")
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == "both" else [args.model]
    all_results = {}

    for model_key in models:
        result = run_model(MODEL_CONFIGS[model_key], args)
        if result:
            all_results[model_key] = result

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"{'Model':<22} {'Default P50':>12} {'ExQ P50':>12} "
              f"{'Speedup':>10} {'Δ':>8}")
        print("-" * 66)
        for k, r in all_results.items():
            if r.get("rpgo"):
                print(
                    f"{r['model']:<22} {r['default']['p50']:>11.3f}ms "
                    f"{r['rpgo']['p50']:>11.3f}ms "
                    f"{r['speedup']:>9.2f}x "
                    f"{r['delta_pct']:>+7.1f}%"
                )

        Path("results").mkdir(exist_ok=True)
        Path(args.save).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved: {args.save}")


if __name__ == "__main__":
    main()
