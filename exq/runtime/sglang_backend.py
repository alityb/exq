"""
ExQ SGLang integration.

Patches SGLang's UnquantizedFusedMoEMethod.forward_cuda to dispatch
through ExQ's INT4 CUDA+Triton kernel pipeline.

The patch replaces the two GEMMs (gate_up + down) for layers covered
by the artifact. Layers not in the artifact fall through to SGLang's
default kernel.

Usage:
    from exq.runtime.sglang_backend import patch_sglang
    backend = patch_sglang("artifacts/qwen3-30b-a3b.json")
    # then launch SGLang server normally
"""

from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger(__name__)

# ── Per-layer INT4 weight cache ───────────────────────────────────────────────
# Packed on first call, reused every subsequent call.
# Key: id(layer) → dict with packed weights + metadata
_exq_packed_cache: dict[int, dict] = {}


def _get_or_pack(layer) -> dict:
    """Pack w13_weight and w2_weight to INT4 on first call, then cache."""
    key = id(layer)
    if key in _exq_packed_cache:
        return _exq_packed_cache[key]

    from exq.kernels.moe_int4_kernel import pack_experts_int4

    t0 = time.perf_counter()
    w1 = layer.w13_weight.to(torch.float16)  # [n_experts, 2*inter, hidden]
    w2 = layer.w2_weight.to(torch.float16)   # [n_experts, hidden, inter]

    w1_packed, w1_scales = pack_experts_int4(w1, group_size=128)
    w2_packed, w2_scales = pack_experts_int4(w2, group_size=128)

    cache = {
        "w1_packed": w1_packed, "w1_scales": w1_scales,
        "w2_packed": w2_packed, "w2_scales": w2_scales,
        "n_experts": w1.shape[0],
        "inter2":    w1.shape[1],   # 2 * intermediate_size
        "hidden":    w1.shape[2],
        "flat_router_cache": None,  # populated on first forward
    }
    _exq_packed_cache[key] = cache
    logger.info(
        f"ExQ: packed layer {getattr(layer, 'layer_id', '?')} "
        f"in {(time.perf_counter()-t0)*1000:.1f}ms"
    )
    return cache


# ── Patched forward_cuda ──────────────────────────────────────────────────────

def _exq_forward_cuda(
    self_method,
    layer,
    dispatch_output,
    _original_forward_cuda,
    _artifact_n_layers: int,
):
    """
    Replacement for UnquantizedFusedMoEMethod.forward_cuda.
    Uses ExQ's CUDA dispatch + Triton INT4 GEMMs.
    Falls through to SGLang's default kernel for uncovered layers.
    """
    import triton
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sgl_kernel import moe_align_block_size as sgl_align, silu_and_mul
    import exq_dispatch_cuda
    from exq.kernels.moe_int4_kernel import _moe_gemm_int4_kernel

    layer_id = getattr(layer, "layer_id", None)
    if layer_id is None or layer_id >= _artifact_n_layers:
        return _original_forward_cuda(self_method, layer, dispatch_output)

    hidden_states = dispatch_output.hidden_states    # [n_tokens, hidden]
    topk_output   = dispatch_output.topk_output
    topk_weights  = topk_output.topk_weights          # [n_tokens, top_k]
    topk_ids      = topk_output.topk_ids              # [n_tokens, top_k]

    cache     = _get_or_pack(layer)
    n_experts = cache["n_experts"]
    inter2    = cache["inter2"]
    hidden    = cache["hidden"]
    n_tokens  = hidden_states.shape[0]
    top_k     = topk_ids.shape[1]
    n_active  = n_tokens * top_k
    BLOCK_M   = 16

    # ── Dispatch: sgl_align (20μs) + build_ends_from_slots (48μs) ────────────
    max_padded = n_active + (n_experts + 1) * (BLOCK_M - 1)
    sorted_ids  = torch.empty(max_padded, dtype=torch.int32, device="cuda")
    expert_blk  = torch.empty(triton.cdiv(max_padded, BLOCK_M), dtype=torch.int32, device="cuda")
    ntpp        = torch.empty(1, dtype=torch.int32, device="cuda")
    cumsum_buf  = torch.empty(n_experts + 2, dtype=torch.int32, device="cuda")

    sgl_align(topk_ids, n_experts + 1, BLOCK_M,
               sorted_ids, expert_blk, ntpp, cumsum_buf, True)

    flat_router = topk_ids.reshape(-1).int().contiguous()
    sort_order  = torch.empty(n_active, dtype=torch.int64, device="cuda")
    expert_ends = torch.zeros(n_experts + 1, dtype=torch.int32, device="cuda")
    exq_dispatch_cuda.build_ends_from_slots(
        sorted_ids, flat_router, expert_ends, sort_order, n_active, n_experts)

    # ── Gather ────────────────────────────────────────────────────────────────
    sorted_hidden = torch.empty(n_active, hidden, dtype=torch.float16, device="cuda")
    exq_dispatch_cuda.gather_hidden(hidden_states, sort_order, sorted_hidden, top_k)

    max_tok = int((expert_ends[1:] - expert_ends[:-1]).max().item())
    nm      = max((max_tok + 64 - 1) // 64, 1)
    BN, BK  = 128, 32

    # ── GEMM1: gate+up ────────────────────────────────────────────────────────
    gate_up = torch.zeros(n_active, inter2, dtype=torch.float16, device="cuda")
    _moe_gemm_int4_kernel[(n_experts, nm, (inter2 + BN - 1) // BN)](
        sorted_hidden, cache["w1_packed"], cache["w1_scales"], gate_up, expert_ends,
        N=inter2, K=hidden,
        stride_am=sorted_hidden.stride(0),
        stride_be=cache["w1_packed"].stride(0), stride_bn=cache["w1_packed"].stride(1),
        stride_se=cache["w1_scales"].stride(0), stride_sn=cache["w1_scales"].stride(1),
        stride_cm=gate_up.stride(0),
        BLOCK_M=64, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE=128,
    )

    # ── SiLU+mul ──────────────────────────────────────────────────────────────
    inter = inter2 // 2
    mid = torch.empty(n_active, inter, dtype=torch.float16, device="cuda")
    silu_and_mul(gate_up, mid)

    # ── GEMM2: down ───────────────────────────────────────────────────────────
    down_out = torch.zeros(n_active, hidden, dtype=torch.float16, device="cuda")
    _moe_gemm_int4_kernel[(n_experts, nm, (hidden + BN - 1) // BN)](
        mid, cache["w2_packed"], cache["w2_scales"], down_out, expert_ends,
        N=hidden, K=inter,
        stride_am=mid.stride(0),
        stride_be=cache["w2_packed"].stride(0), stride_bn=cache["w2_packed"].stride(1),
        stride_se=cache["w2_scales"].stride(0), stride_sn=cache["w2_scales"].stride(1),
        stride_cm=down_out.stride(0),
        BLOCK_M=64, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE=128,
    )

    # ── Combine ───────────────────────────────────────────────────────────────
    result = torch.empty(n_tokens, hidden, dtype=torch.float16, device="cuda")
    exq_dispatch_cuda.combine(down_out, sort_order, topk_weights, result)

    return StandardCombineInput(hidden_states=result)


# ── Public API ────────────────────────────────────────────────────────────────

class ExQSGLangBackend:
    """Holds artifact metadata and the per-layer weight cache."""

    def __init__(self, artifact_path: str):
        import json
        artifact = json.load(open(artifact_path, encoding="utf-8"))
        qa = artifact.get("quant_assignments", {})
        layer_ids = {int(k.split(":")[0]) for k in qa}
        self.n_layers      = max(layer_ids) + 1 if layer_ids else 0
        self.artifact_path = artifact_path
        logger.info(f"ExQ SGLang backend: {self.n_layers} layers ({len(qa)} assignments)")

    def clear_cache(self):
        _exq_packed_cache.clear()

    def cache_stats(self) -> dict:
        return {"n_layers_cached": len(_exq_packed_cache),
                "n_layers_in_artifact": self.n_layers}


def patch_sglang(artifact_path: str) -> ExQSGLangBackend:
    """
    Patch SGLang's MoE forward to use ExQ's INT4 kernel.

    Replaces UnquantizedFusedMoEMethod.forward_cuda for layers covered
    by the artifact. All other layers fall through to SGLang's default.
    """
    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    except ImportError as exc:
        raise ImportError("SGLang not installed: pip install sglang") from exc

    backend = ExQSGLangBackend(artifact_path)

    if getattr(UnquantizedFusedMoEMethod, "_exq_patched", False):
        logger.warning("ExQ: re-patching SGLang (cache cleared)")
        _exq_packed_cache.clear()

    original = UnquantizedFusedMoEMethod.forward_cuda
    n_layers = backend.n_layers

    def patched(self, layer, dispatch_output):
        return _exq_forward_cuda(self, layer, dispatch_output, original, n_layers)

    UnquantizedFusedMoEMethod.forward_cuda = patched
    UnquantizedFusedMoEMethod._exq_patched = True
    logger.info(f"ExQ: patched SGLang ({n_layers} layers covered)")
    return backend


def unpatch_sglang():
    """Restore SGLang's original forward_cuda."""
    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    except ImportError:
        return
    if not getattr(UnquantizedFusedMoEMethod, "_exq_patched", False):
        return
    import importlib
    import sglang.srt.layers.quantization.unquant as mod
    importlib.reload(mod)
    _exq_packed_cache.clear()
    logger.info("ExQ: SGLang patch removed")
