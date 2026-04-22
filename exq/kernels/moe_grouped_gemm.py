"""
Baseline Triton grouped GEMM for MoE expert dispatch.

Implements sorted-token dispatch: tokens are sorted by expert assignment
so each expert's tokens are contiguous. This turns random HBM scatter-gather
into sequential reads, which is critical for bandwidth-limited GEMV regimes.

This is the honest baseline — uniform tile sizes, no routing profile.
The ExQ kernel (moe_exq_kernel.py) adds frequency-aware tile sizing
and BLOCK_N tuning on top.

Reference: vLLM fused_moe.py, Megablox (Gale et al., 2023)

Hardware target: NVIDIA A10G
  - 24GB VRAM, 600 GB/s HBM bandwidth
  - 31.2 TFLOPS BF16, 125 TOPS INT8
  - 96KB shared memory per SM

OLMoE-1B-7B dimensions:
  - hidden_size = 2048, intermediate_size = 1024
  - 64 experts per layer, top-2 routing
  - Expert weight size at BF16: one projection = [1024, 2048] = 4MB
  - Arithmetic intensity at n_tokens=16: ~16 FLOP/byte → memory-bound
  - Ridge point: ~52 FLOP/byte (BF16 compute / HBM bandwidth)

Usage:
    output = moe_grouped_gemm(
        hidden_states,   # [n_tokens, hidden_size]   BF16
        expert_weights,  # [n_experts, N, K]         BF16
        router_indices,  # [n_tokens, top_k]         int64
        n_experts,
    )
    # output shape: [n_tokens * top_k, N]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _moe_gemm_kernel(
    # Matrix pointers
    A_ptr,              # [n_active_tokens, K] — sorted hidden states
    B_ptr,              # [n_experts, N, K] — expert weight matrices
    C_ptr,              # [n_active_tokens, N] — output
    # Expert boundary array [n_experts + 1] — prefix-sum of token counts
    expert_ends_ptr,    # token_ends[e] = index of first token NOT belonging to expert e
    # Dimensions
    N: tl.constexpr,   # output dim (intermediate_size)
    K: tl.constexpr,   # input dim (hidden_size)
    # Strides
    stride_am,          # A row stride (= K for contiguous)
    stride_be,          # B expert stride (= N * K)
    stride_bn,          # B row stride (= K)
    stride_cm,          # C row stride (= N)
    # Tile sizes — constexpr allows Triton to specialise and autotune
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Optional per-expert enable flag (ptr to int32 array, or 0 to skip check)
    expert_mask_ptr,    # [n_experts]: 1=process, 0=skip; pass 0 to process all
    USE_EXPERT_MASK: tl.constexpr,  # compile-time flag: True = check the mask
):
    """
    Grouped GEMM: C[expert_tokens] = A[expert_tokens] @ B[expert].T

    Grid: (n_experts, max_m_tiles, n_tiles_N)
    Each program handles one (expert, M-tile, N-tile) triple.
    """
    expert_id  = tl.program_id(0)
    m_tile_id  = tl.program_id(1)
    n_tile_id  = tl.program_id(2)

    # Optional expert mask: skip this expert if mask[expert_id] == 0
    if USE_EXPERT_MASK:
        enabled = tl.load(expert_mask_ptr + expert_id)
        if enabled == 0:
            return

    # Find token range for this expert via the prefix-sum boundary array
    tok_start = tl.load(expert_ends_ptr + expert_id)
    tok_end   = tl.load(expert_ends_ptr + expert_id + 1)
    n_tokens  = tok_end - tok_start

    # Exit early for experts with no tokens this forward pass
    if n_tokens == 0:
        return

    # Exit early if this M-tile is past the end of this expert's tokens
    m_start = m_tile_id * BLOCK_M
    if m_start >= n_tokens:
        return

    # --- Tile index offsets ------------------------------------------------
    offs_m = m_start + tl.arange(0, BLOCK_M)
    n_start = n_tile_id * BLOCK_N
    offs_n  = n_start + tl.arange(0, BLOCK_N)

    # --- Accumulator (fp32 for numerical stability) -------------------------
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # B pointer for this expert: B_ptr + expert_id * stride_be
    b_base = B_ptr + expert_id * stride_be

    # --- K-loop: accumulate partial dot products ----------------------------
    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        offs_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load A tile: A[tok_start + offs_m, offs_k]
        a_ptrs = A_ptr + (tok_start + offs_m[:, None]) * stride_am + offs_k[None, :]
        mask_a = (offs_m[:, None] < n_tokens) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float16)

        # Load B tile: B[expert_id, offs_n, offs_k]  (weight: [N, K])
        b_ptrs = b_base + offs_n[:, None] * stride_bn + offs_k[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float16)

        # Accumulate: A [BLOCK_M, BLOCK_K] @ B.T [BLOCK_K, BLOCK_N] = [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, tl.trans(b))

    # --- Store output -------------------------------------------------------
    c_ptrs = C_ptr + (tok_start + offs_m[:, None]) * stride_cm + offs_n[None, :]
    mask_c = (offs_m[:, None] < n_tokens) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_c)


# ---------------------------------------------------------------------------
# Shared dispatch helper
# ---------------------------------------------------------------------------

def sort_tokens_by_expert(
    router_indices: torch.Tensor,   # [n_tokens, top_k]
    hidden_states: torch.Tensor,    # [n_tokens, K]
    n_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert assignment and build the expert boundary array.

    This is the core dispatch step shared by every ExQ GEMM variant.
    Returns:
        sort_order      [n_active]     — original flat slot for each sorted row
        sorted_hidden   [n_active, K]  — hidden states in expert-sorted order
        expert_ends     [n_experts+1]  — prefix-sum boundary array
        max_tok         int            — max tokens any single expert receives
    """
    top_k    = router_indices.shape[1]
    n_active = router_indices.shape[0] * top_k

    flat         = router_indices.reshape(-1).long()
    sort_order   = torch.argsort(flat, stable=True)
    sorted_ids   = flat[sort_order]
    sorted_hidden = hidden_states[(sort_order // top_k).long()]

    counts       = torch.bincount(sorted_ids, minlength=n_experts)
    expert_ends  = torch.zeros(n_experts + 1, dtype=torch.int32, device="cuda")
    expert_ends[1:] = counts.cumsum(0).int()
    max_tok      = int(counts.max().item()) if n_active > 0 else 1

    return sort_order, sorted_hidden, expert_ends, max_tok


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def moe_grouped_gemm(
    hidden_states: torch.Tensor,   # [n_tokens, hidden_size]   BF16/FP16
    expert_weights: torch.Tensor,  # [n_experts, N, K]         BF16/FP16
    router_indices: torch.Tensor,  # [n_tokens, top_k]         int64/int32
    n_experts: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """
    Sorted-token grouped GEMM for MoE expert dispatch.

    Tokens are sorted by expert assignment before the GEMM, making each
    expert's token slice contiguous in memory. This converts scatter-gather
    HBM access into sequential access, which is critical at low arithmetic
    intensity (typical for decode-phase inference).

    Returns:
        output [n_tokens * top_k, N] in float16
    """
    assert hidden_states.is_cuda, "hidden_states must be on CUDA"
    assert expert_weights.is_cuda, "expert_weights must be on CUDA"

    n_tokens_total, K = hidden_states.shape
    _, N, K2 = expert_weights.shape
    assert K == K2, f"K mismatch: hidden_states K={K}, expert_weights K={K2}"

    n_active = n_tokens_total * router_indices.shape[1]

    sort_order, sorted_hidden, expert_ends, max_tok = sort_tokens_by_expert(
        router_indices, hidden_states, n_experts)

    # --- Allocate output ---------------------------------------------------
    output = torch.zeros(n_active, N, dtype=torch.float16, device="cuda")

    # --- Launch kernel -----------------------------------------------------
    max_m_tiles = (max_tok + block_m - 1) // block_m
    n_n_tiles   = (N + block_n - 1) // block_n
    grid = (n_experts, max(max_m_tiles, 1), n_n_tiles)

    _moe_gemm_kernel[grid](
        sorted_hidden, expert_weights, output,
        expert_ends,
        N=N, K=K,
        stride_am=sorted_hidden.stride(0),
        stride_be=expert_weights.stride(0),
        stride_bn=expert_weights.stride(1),
        stride_cm=output.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        expert_mask_ptr=0,
        USE_EXPERT_MASK=False,
    )
    return output


# ---------------------------------------------------------------------------
# Utility: unsort output back to token order
# ---------------------------------------------------------------------------

def unsort_output(
    sorted_output: torch.Tensor,   # [n_active, N]
    router_indices: torch.Tensor,  # [n_tokens, top_k]
    router_weights: torch.Tensor,  # [n_tokens, top_k]  combination weights
) -> torch.Tensor:
    """
    Recombine sorted GEMM output into per-token vectors.

    Reverses the sort done in moe_grouped_gemm and applies the router's
    combination weights (weighted sum over top_k experts).

    Returns:
        output [n_tokens, N]
    """
    n_tokens, top_k = router_indices.shape
    _, N = sorted_output.shape

    flat_indices  = router_indices.reshape(-1).long()
    sort_order    = torch.argsort(flat_indices, stable=True)
    unsort_order  = torch.argsort(sort_order, stable=True)

    # Un-sort: sorted_output[unsort_order] → original [n_active, N] order
    unsorted = sorted_output[unsort_order].view(n_tokens, top_k, N)

    # Weighted combination
    weights = router_weights.unsqueeze(-1)          # [n_tokens, top_k, 1]
    return (unsorted * weights).sum(dim=1)           # [n_tokens, N]


# ---------------------------------------------------------------------------
# PyTorch reference implementation (correctness baseline)
# ---------------------------------------------------------------------------

def pytorch_moe_reference(
    hidden_states: torch.Tensor,   # [n_tokens, K]
    expert_weights: torch.Tensor,  # [n_experts, N, K]
    router_indices: torch.Tensor,  # [n_tokens, top_k]
    router_weights: torch.Tensor,  # [n_tokens, top_k]
) -> torch.Tensor:
    """
    Naive PyTorch MoE forward: one matmul per (token, expert) pair.
    Correct by construction; used as the correctness oracle.

    Returns:
        output [n_tokens, N]
    """
    n_tokens, top_k = router_indices.shape
    _, N, K = expert_weights.shape
    output = torch.zeros(n_tokens, N, dtype=hidden_states.dtype,
                         device=hidden_states.device)
    for tok in range(n_tokens):
        for k in range(top_k):
            eid = int(router_indices[tok, k].item())
            w   = float(router_weights[tok, k].item())
            # hidden [K] @ weight.T [K, N] → [N]
            output[tok] += w * (hidden_states[tok] @ expert_weights[eid].T)
    return output
