"""
ExQ SGLang integration.

Patches SGLang's UnquantizedFusedMoEMethod to dispatch expert GEMMs
through ExQ's INT4 Triton kernel using a compiled artifact.

The integration is a targeted patch of one method:
  UnquantizedFusedMoEMethod.forward_cuda

SGLang's architecture (v0.5.x):
  FusedMoE.forward_impl
    └─ dispatcher.dispatch()           # sort tokens by expert
    └─ run_moe_core()
         └─ quant_method.apply()
              └─ forward_cuda()        # ← ExQ patches here
                   └─ fused_experts()  # SGLang's Triton kernel

The patch replaces the two GEMMs (gate_up + down) with ExQ's
INT4 kernel when the layer is covered by the artifact. Layers not
in the artifact fall through to SGLang's default kernel.

Weight layout in SGLang:
  w13_weight: [n_experts, 2*intermediate, hidden]  — gate+up fused
  w2_weight:  [n_experts, hidden, intermediate]    — down projection

ExQ kernel interface:
  moe_int4_forward(hidden, packed_w1, scales_w1, router_indices, n_experts)
  → [n_tokens * top_k, intermediate]  (gate+up result, before activation)

Usage:
    from exq.runtime.sglang_backend import patch_sglang
    backend = patch_sglang("artifacts/qwen3-30b-a3b.json")
    # then launch SGLang server normally

Notes on what this is NOT:
  - It does not register a new MoeRunnerBackend enum value (closed enum)
  - It does not touch SGLang's dispatcher, token sorter, or router
  - It does not change weight loading (weights stay in SGLang's format)
    until the first forward pass, at which point ExQ packs them once
    and caches the packed tensors
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Per-layer INT4 weight cache ───────────────────────────────────────────────
# SGLang stores weights as fp16 w13_weight and w2_weight on the layer object.
# We pack them to INT4 once on first call and cache the result.
# Key: id(layer) → {"w1_packed", "w1_scales", "w2_packed", "w2_scales"}
_exq_packed_cache: dict[int, dict] = {}


def _get_or_pack(layer) -> dict:
    """Pack w13_weight and w2_weight to INT4 on first call, then cache."""
    key = id(layer)
    if key in _exq_packed_cache:
        return _exq_packed_cache[key]

    from exq.kernels.moe_int4_kernel import pack_experts_int4

    t0 = time.perf_counter()
    w1 = layer.w13_weight  # [n_experts, 2*inter, hidden]
    w2 = layer.w2_weight   # [n_experts, hidden,  inter]

    # Ensure fp16 for packing (SGLang may load in bf16)
    if w1.dtype != torch.float16:
        w1 = w1.to(torch.float16)
    if w2.dtype != torch.float16:
        w2 = w2.to(torch.float16)

    w1_packed, w1_scales = pack_experts_int4(w1, group_size=128)
    w2_packed, w2_scales = pack_experts_int4(w2, group_size=128)

    cache = {
        "w1_packed": w1_packed,
        "w1_scales": w1_scales,
        "w2_packed": w2_packed,
        "w2_scales": w2_scales,
        "n_experts": w1.shape[0],
        "inter2":    w1.shape[1],   # 2 * intermediate_size
        "hidden":    w1.shape[2],
    }
    _exq_packed_cache[key] = cache
    dt = (time.perf_counter() - t0) * 1000
    logger.info(
        f"ExQ: packed layer {getattr(layer, 'layer_id', '?')} INT4 "
        f"w1={list(w1.shape)} w2={list(w2.shape)} in {dt:.1f}ms"
    )
    return cache


# ── The patched forward_cuda ──────────────────────────────────────────────────

def _rpgo_forward_cuda(
    self_method,
    layer,
    dispatch_output,
    _original_forward_cuda,
    _artifact_n_layers: int,
):
    """
    Replacement for UnquantizedFusedMoEMethod.forward_cuda.

    Runs ExQ's INT4 kernel for layers covered by the artifact.
    Falls through to SGLang's default kernel for all other layers.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sgl_kernel import silu_and_mul

    layer_id = getattr(layer, "layer_id", None)

    # Fall through to SGLang default if layer is not in artifact
    if layer_id is None or layer_id >= _artifact_n_layers:
        return _original_forward_cuda(self_method, layer, dispatch_output)

    hidden_states = dispatch_output.hidden_states       # [n_tokens, hidden]
    topk_output   = dispatch_output.topk_output
    topk_weights  = topk_output.topk_weights            # [n_tokens, top_k]
    topk_ids      = topk_output.topk_ids                # [n_tokens, top_k]

    from exq.kernels.moe_int4_kernel import moe_int4_forward

    cache = _get_or_pack(layer)
    n_experts = cache["n_experts"]
    inter2    = cache["inter2"]     # 2 * intermediate_size

    # Compute the sorted expert assignment order (needed for both GEMMs and
    # for building the correct GEMM2 router_indices).
    flat_ids   = topk_ids.reshape(-1).long()           # [n_active]
    sort_order = torch.argsort(flat_ids, stable=True)  # [n_active]
    sorted_expert_ids = flat_ids[sort_order]           # [n_active], sorted

    # ── GEMM 1: gate + up projection (w13) ───────────────────────────────────
    # Output: [n_tokens * top_k, 2*intermediate]
    gate_up = moe_int4_forward(
        hidden_states=hidden_states,
        expert_packed=cache["w1_packed"],
        expert_scales=cache["w1_scales"],
        router_indices=topk_ids,          # [n_tokens, top_k]
        n_experts=n_experts,
        group_size=128,
    )

    # ── Activation: SiLU + elementwise mul (SwiGLU) ──────────────────────────
    n_active = gate_up.shape[0]
    intermediate = torch.empty(
        (n_active, inter2 // 2),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    silu_and_mul(gate_up, intermediate)

    # ── GEMM 2: down projection (w2) ─────────────────────────────────────────
    # intermediate: [n_active, inter] — one row per (token, expert) slot,
    # already in sorted-expert order (same order moe_int4_forward uses).
    # w2_weight: [n_experts, hidden, inter] → packed [n_experts, hidden, inter//2]
    #
    # For GEMM2, each intermediate row belongs to exactly ONE expert.
    # We pass sorted_expert_ids as a [n_active, 1] router_indices tensor so
    # moe_int4_forward dispatches each row to its correct expert.
    r_idx_down = sorted_expert_ids.unsqueeze(1)   # [n_active, 1]
    down_out = moe_int4_forward(
        hidden_states=intermediate,
        expert_packed=cache["w2_packed"],
        expert_scales=cache["w2_scales"],
        router_indices=r_idx_down,       # [n_active, 1] — 1 expert per slot
        n_experts=n_experts,
        group_size=128,
    )
    # down_out: [n_active, hidden], in sorted-expert order

    # ── Combine: weighted sum over top_k ─────────────────────────────────────
    # down_out: [n_active, hidden], in sorted-expert order.
    # We need to unsort back to [n_tokens, top_k, hidden] order then
    # apply topk_weights and sum over top_k.
    n_tokens = hidden_states.shape[0]
    top_k    = topk_ids.shape[1]
    hidden   = cache["hidden"]

    # unsort: map sorted positions back to flat (token, k) positions
    # sort_order[j] = original flat index i, so unsort_order[i] = j
    unsort_ord = torch.argsort(sort_order, stable=True)

    unsorted = down_out[unsort_ord].view(n_tokens, top_k, hidden)
    weights  = topk_weights.unsqueeze(-1).to(unsorted.dtype)
    out      = (unsorted * weights).sum(dim=1)

    return StandardCombineInput(hidden_states=out)


# ── Public API ────────────────────────────────────────────────────────────────

class ExQSGLangBackend:
    """
    ExQ backend for SGLang MoE.

    Holds the artifact metadata and manages the per-layer weight cache.
    Created by patch_sglang() and stored as a module-level singleton so
    the patched forward_cuda can reference it.
    """

    def __init__(self, artifact_path: str):
        import json
        with open(artifact_path, encoding="utf-8") as f:
            artifact = json.load(f)

        qa = artifact.get("quant_assignments", {})
        layer_ids = {int(k.split(":")[0]) for k in qa}
        self.n_layers     = max(layer_ids) + 1 if layer_ids else 0
        self.artifact     = artifact
        self.artifact_path = artifact_path

        logger.info(
            f"ExQ SGLang backend: {self.n_layers} layers in artifact "
            f"({len(qa)} expert assignments)"
        )

    def clear_cache(self):
        """Clear packed weight cache (e.g., after weight updates)."""
        _exq_packed_cache.clear()
        logger.info("ExQ: packed weight cache cleared")

    def cache_stats(self) -> dict:
        return {
            "n_layers_cached": len(_exq_packed_cache),
            "n_layers_in_artifact": self.n_layers,
        }


def patch_sglang(artifact_path: str) -> ExQSGLangBackend:
    """
    Patch SGLang's MoE forward to use ExQ's INT4 kernel.

    Replaces UnquantizedFusedMoEMethod.forward_cuda with a version
    that dispatches through ExQ's packed INT4 Triton kernel for
    layers covered by the artifact.

    Args:
        artifact_path: Path to compiled ExQ artifact JSON.

    Returns:
        ExQSGLangBackend instance (holds artifact metadata, weight cache).

    Raises:
        ImportError: if SGLang is not installed.
        RuntimeError: if SGLang's API doesn't match expected structure.
    """
    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    except ImportError as exc:
        raise ImportError(
            "SGLang not installed. Install with: pip install sglang"
        ) from exc

    backend = ExQSGLangBackend(artifact_path)

    # Guard against double-patching
    if getattr(UnquantizedFusedMoEMethod, "_exq_patched", False):
        logger.warning("ExQ: SGLang already patched, re-patching with new artifact")
        _exq_packed_cache.clear()

    original_forward_cuda = UnquantizedFusedMoEMethod.forward_cuda

    # Capture backend in closure
    n_layers = backend.n_layers

    def patched_forward_cuda(self, layer, dispatch_output):
        # `self` is the UnquantizedFusedMoEMethod instance (bound automatically)
        return _rpgo_forward_cuda(
            self,
            layer,
            dispatch_output,
            original_forward_cuda,
            n_layers,
        )

    UnquantizedFusedMoEMethod.forward_cuda = patched_forward_cuda
    UnquantizedFusedMoEMethod._exq_patched = True

    logger.info(
        f"ExQ: patched SGLang UnquantizedFusedMoEMethod.forward_cuda "
        f"(artifact covers {n_layers} layers)"
    )
    return backend


def unpatch_sglang():
    """Restore SGLang's original forward_cuda (for testing)."""
    try:
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    except ImportError:
        return

    if not getattr(UnquantizedFusedMoEMethod, "_exq_patched", False):
        return

    # The original is captured in the closure; we can't easily restore it
    # from outside. Instead, re-import to get a fresh class.
    import importlib
    import sglang.srt.layers.quantization.unquant as mod
    importlib.reload(mod)
    _exq_packed_cache.clear()
    logger.info("ExQ: SGLang patch removed (module reloaded)")
