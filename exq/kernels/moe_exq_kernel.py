"""
ExQ frequency-aware Triton MoE kernel.

Design: split dispatch based on ExQ quant tier (BF16/INT8 = hot,
INT4 = cold). Two kernel paths:

  Hot (BF16 + INT8): BLOCK_M=128, BLOCK_N=128
    - These experts receive more tokens per forward pass.
    - Larger tiles give better occupancy and amortise kernel overhead.
    - At AI > 52 FLOP/byte (Qwen3-30B hot experts at batch≥8), larger
      tiles improve SRAM utilization.

  Cold (INT4): BLOCK_M=64, BLOCK_N=64
    - Rarely-active experts; small tiles avoid padding waste.

Per-call overhead is minimised by:
  1. Pre-caching mask tensors (built once per layer at first call)
  2. Two kernel launches total (not one per expert)
  3. Adaptive threshold: only launch the hot path when n_hot >= 3,
     because each extra kernel launch costs ~70µs overhead.

Model-specific behavior:
  OLMoE-1B-7B:  1 BF16, 438 INT8 → n_hot=439 → hot path fires every layer
  Qwen3-30B-A3B: 0 BF16, 252 INT8 → n_hot=252 → hot path fires most layers
"""

from __future__ import annotations

import torch

from .exq_artifact import ExpertProfile
from .moe_grouped_gemm import _moe_gemm_kernel


def _build_layer_masks(profile: ExpertProfile, layer_idx: int, n_experts: int):
    """
    Pre-build and cache hot/cold expert masks for a given layer.
    Built once per (profile, layer_idx) pair, reused on all subsequent calls.

    Hot = BF16 (tier 0) or INT8 (tier 1) — receives more tokens, benefits
          from larger tiles.
    Cold = INT4 (tier 2) — rarely activated, small tiles avoid padding waste.
    """
    key = f"_exq_masks_v2_{layer_idx}"
    if hasattr(profile, key):
        return getattr(profile, key)

    layer_p = profile.layer_profile(min(layer_idx, profile.n_layers - 1))
    tiers   = layer_p["tiers"].cpu().tolist()

    # Hot: BF16 (0) or INT8 (1). Cold: INT4 (2)
    hot  = torch.tensor([1 if int(t) <= 1 else 0 for t in tiers],
                         dtype=torch.int32, device="cuda")
    cold = torch.tensor([0 if int(t) <= 1 else 1 for t in tiers],
                         dtype=torch.int32, device="cuda")
    n_hot = int(hot.sum().item())

    result = (hot, cold, n_hot)
    setattr(profile, key, result)
    return result


def rpgo_moe_forward(
    hidden_states: torch.Tensor,    # [n_tokens, K]
    expert_weights: torch.Tensor,   # [n_experts, N, K]
    router_indices: torch.Tensor,   # [n_tokens, top_k]
    profile: ExpertProfile,
    layer_idx: int,
) -> torch.Tensor:
    """
    ExQ frequency-aware MoE forward pass.

    Two kernel calls using the same full expert_ends prefix-sum but
    per-expert enable masks routing each expert to its tile-size path:

      Path 1: BF16 + INT8 experts  → BLOCK_M=128, BLOCK_N=128
      Path 2: INT4 experts          → BLOCK_M=64,  BLOCK_N=64

    Expert masks are pre-cached at first call per layer.
    Falls back to a single unmasked call when n_hot < 3.

    Returns:
        output [n_tokens * top_k, N] in float16
    """
    assert hidden_states.is_cuda and expert_weights.is_cuda

    n_tokens_total, K = hidden_states.shape
    _, N, K2 = expert_weights.shape
    assert K == K2

    top_k     = router_indices.shape[1]
    n_active  = n_tokens_total * top_k
    n_experts = expert_weights.shape[0]

    # --- Sort tokens by expert (identical to baseline) ----------------------
    flat_indices = router_indices.reshape(-1).long()
    sort_order   = torch.argsort(flat_indices, stable=True)
    sorted_ids   = flat_indices[sort_order]
    orig_toks    = (sort_order // top_k).long()
    sorted_hidden = hidden_states[orig_toks]

    # --- Full expert boundary array (identical to baseline) -----------------
    token_counts = torch.bincount(sorted_ids, minlength=n_experts)
    expert_ends  = torch.zeros(n_experts + 1, dtype=torch.int32, device="cuda")
    expert_ends[1:] = token_counts.cumsum(0).int()

    # --- Output buffer -------------------------------------------------------
    output = torch.zeros(n_active, N, dtype=torch.float16, device="cuda")

    # --- Pre-cached expert masks (BF16+INT8 = hot, INT4 = cold) -------------
    hot_mask, cold_mask, n_hot = _build_layer_masks(profile, layer_idx, n_experts)

    max_tokens = int(token_counts.max().item()) if n_active > 0 else 1

    # Min 3 hot experts to justify the extra kernel launch (~70µs overhead).
    # For models with very few hot experts, the split just adds latency.
    HOT_THRESHOLD = 3

    if n_hot >= HOT_THRESHOLD:
        # --- Path 1: Hot experts (BF16+INT8) with large tiles ---------------
        BM, BN, BK = 128, 128, 32
        n_m = max((max_tokens + BM - 1) // BM, 1)
        n_n = (N + BN - 1) // BN
        _moe_gemm_kernel[(n_experts, n_m, n_n)](
            sorted_hidden, expert_weights, output,
            expert_ends,
            N=N, K=K,
            stride_am=sorted_hidden.stride(0),
            stride_be=expert_weights.stride(0),
            stride_bn=expert_weights.stride(1),
            stride_cm=output.stride(0),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            expert_mask_ptr=hot_mask,
            USE_EXPERT_MASK=True,
        )

        # --- Path 2: Cold experts (INT4) with standard tiles ----------------
        BM, BN, BK = 64, 64, 32
        n_m = max((max_tokens + BM - 1) // BM, 1)
        n_n = (N + BN - 1) // BN
        _moe_gemm_kernel[(n_experts, n_m, n_n)](
            sorted_hidden, expert_weights, output,
            expert_ends,
            N=N, K=K,
            stride_am=sorted_hidden.stride(0),
            stride_be=expert_weights.stride(0),
            stride_bn=expert_weights.stride(1),
            stride_cm=output.stride(0),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            expert_mask_ptr=cold_mask,
            USE_EXPERT_MASK=True,
        )

    else:
        # --- Single call (baseline behavior) when too few hot experts -------
        BM, BN, BK = 64, 64, 32
        n_m = max((max_tokens + BM - 1) // BM, 1)
        n_n = (N + BN - 1) // BN
        _moe_gemm_kernel[(n_experts, n_m, n_n)](
            sorted_hidden, expert_weights, output,
            expert_ends,
            N=N, K=K,
            stride_am=sorted_hidden.stride(0),
            stride_be=expert_weights.stride(0),
            stride_bn=expert_weights.stride(1),
            stride_cm=output.stride(0),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
            expert_mask_ptr=0,
            USE_EXPERT_MASK=False,
        )

    return output

