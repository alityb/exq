"""
Mixed-precision INT4 dequantization kernel for ExQ MoE expert dispatch.

The core insight: in the memory-bound decode regime, loading fewer weight
bytes directly reduces latency. INT4 weights are packed 2-per-byte with
per-group fp16 scales; the kernel unpacks them on-chip during the GEMM.

Weight storage format (RTN, group_size=128):
  packed:  [n_experts, N, K//2]  uint8   — two int4 per byte, lo/hi nibbles
  scales:  [n_experts, N, K//128] fp16   — per-group abs-max scales

Packing convention (consistent with the Python pack function below):
  byte at [n, k//2] holds:
    lo nibble (bits 0-3): element at column k=2i,   stored as (val+8) in [0,15]
    hi nibble (bits 4-7): element at column k=2i+1, stored as (val+8) in [0,15]
  Dequant: fp16_val = (nibble - 8) * scale[n, k//128]

Design choices:
  - group_size=128 (standard GPTQ/AWQ; one scale per 128 K-elements per row)
  - BLOCK_K must be a power of 2 and <= group_size so each K-tile stays
    within a single scale group — avoids multi-scale loads per inner iteration
  - BLOCK_K=64 is the sweet spot: 64 < 128 (one scale), 32 bytes of packed
    INT4 per BLOCK_N row, good L1 utilisation
  - We use BLOCK_K=64 with INT4 vs BLOCK_K=32 with BF16 — same memory
    footprint per inner iteration but 2× more K coverage per FMA
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ─── Packing utilities (Python / CUDA preprocessing) ─────────────────────────

def pack_int4_weights(
    weights: torch.Tensor,   # [N, K] fp16
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack fp16 weight matrix into INT4 with per-group scales.

    Quantizes using symmetric per-group round-to-nearest, range [-8, 7].
    Stores two INT4 values per byte in lo/hi nibble layout.

    Args:
        weights: [N, K] fp16 weight matrix (one expert, one projection)
        group_size: number of K-elements per scale group (default 128)

    Returns:
        packed: [N, K//2] uint8  — two INT4 per byte
        scales: [N, K//group_size] fp16  — per-group scales

    Raises:
        ValueError: if K is not divisible by 2 or group_size
    """
    N, K = weights.shape
    if K % 2 != 0:
        raise ValueError(f"K={K} must be divisible by 2 for INT4 packing")
    # When K < group_size, use K as the single group (one scale per row).
    # This handles small expert dimensions in tests and compact models.
    if K < group_size:
        group_size = K
    if K % group_size != 0:
        raise ValueError(
            f"K={K} must be divisible by group_size={group_size}. "
            f"Choose a group_size that divides K, or use group_size=K for a single group."
        )

    n_groups = K // group_size
    w_fp32 = weights.float()  # promote for quantization arithmetic

    # --- Per-group symmetric quantization -----------------------------------
    # Reshape to [N, n_groups, group_size] for group-wise scaling
    w_grouped = w_fp32.reshape(N, n_groups, group_size)
    scales_fp32 = w_grouped.abs().amax(dim=-1) / 7.0                  # [N, n_groups]
    scales_fp32 = scales_fp32.clamp(min=torch.finfo(torch.float32).eps)

    w_scaled = w_grouped / scales_fp32.unsqueeze(-1)                   # [N, n_groups, group_size]
    quant = w_scaled.round().clamp(-8, 7).to(torch.int8)               # [N, K]
    quant = quant.reshape(N, K)

    # --- Nibble packing: offset to [0,15] then interleave -------------------
    # lo nibble: even K-columns (0, 2, 4, …)
    # hi nibble: odd  K-columns (1, 3, 5, …)
    lo = (quant[:, 0::2] + 8).to(torch.uint8)   # [N, K//2], values in [0,15]
    hi = (quant[:, 1::2] + 8).to(torch.uint8)   # [N, K//2], values in [0,15]
    packed = (lo | (hi << 4)).contiguous()       # [N, K//2], uint8

    scales = scales_fp32.to(torch.float16).contiguous()                # [N, n_groups]

    return packed, scales


def pack_experts_int4(
    expert_weights: torch.Tensor,   # [n_experts, N, K] fp16
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack an entire expert weight bank into INT4.

    Args:
        expert_weights: [n_experts, N, K] fp16

    Returns:
        packed: [n_experts, N, K//2] uint8
        scales: [n_experts, N, K//group_size] fp16
    """
    n_experts, N, K = expert_weights.shape
    packed_list, scale_list = [], []
    for e in range(n_experts):
        p, s = pack_int4_weights(expert_weights[e], group_size=group_size)
        packed_list.append(p)
        scale_list.append(s)
    return (
        torch.stack(packed_list).contiguous(),   # [n_experts, N, K//2]
        torch.stack(scale_list).contiguous(),    # [n_experts, N, K//group_size]
    )


# ─── Triton kernel ─────────────────────────────────────────────────────────────

@triton.jit
def _moe_gemm_int4_kernel(
    # Activation pointer (fp16, sorted by expert)
    A_ptr,           # [n_active, K]
    # Packed INT4 weight (uint8, two INT4 per byte)
    B_packed_ptr,    # [n_experts, N, K//2]
    # Per-group scale factors (fp16)
    B_scales_ptr,    # [n_experts, N, K//GROUP_SIZE]
    # Output
    C_ptr,           # [n_active, N]
    # Expert token boundaries
    expert_ends_ptr, # [n_experts + 1]
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    # Strides
    stride_am,       # A row stride
    stride_be,       # B_packed expert stride = N * (K//2)
    stride_bn,       # B_packed row stride    = K//2
    stride_se,       # scales expert stride   = N * (K//GROUP_SIZE)
    stride_sn,       # scales row stride      = K//GROUP_SIZE
    stride_cm,       # C row stride
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must be <= GROUP_SIZE and even
    GROUP_SIZE: tl.constexpr, # = 128
):
    """
    INT4 grouped GEMM with on-the-fly dequantization.

    Grid: (n_experts, max_M_tiles, N // BLOCK_N)

    For each K-tile:
      1. Load A tile:       [BLOCK_M, BLOCK_K]   fp16
      2. Load B_packed tile:[BLOCK_N, BLOCK_K//2] uint8
      3. Unpack nibbles ->  [BLOCK_N, BLOCK_K]   int4 values in [-8,7]
      4. Load scales:       [BLOCK_N]             fp16  (one per row, same group)
      5. Dequantize B ->    [BLOCK_N, BLOCK_K]   fp16
      6. acc += A @ B.T
    """
    expert_id = tl.program_id(0)
    m_tile_id = tl.program_id(1)
    n_tile_id = tl.program_id(2)

    tok_start = tl.load(expert_ends_ptr + expert_id)
    tok_end   = tl.load(expert_ends_ptr + expert_id + 1)
    n_tokens  = tok_end - tok_start

    if n_tokens == 0:
        return

    m_start = m_tile_id * BLOCK_M
    if m_start >= n_tokens:
        return

    n_start = n_tile_id * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Base pointers for this expert
    b_packed_base = B_packed_ptr + expert_id * stride_be
    b_scales_base = B_scales_ptr + expert_id * stride_se

    # K-loop: each iteration covers BLOCK_K input channels
    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_tile * BLOCK_K
        offs_k  = k_start + tl.arange(0, BLOCK_K)

        # ── Load activation tile A [BLOCK_M, BLOCK_K] ──────────────────────
        a_ptrs = A_ptr + (tok_start + offs_m[:, None]) * stride_am + offs_k[None, :]
        mask_a = (offs_m[:, None] < n_tokens) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float16)

        # ── Load packed INT4 weight tile B [BLOCK_N, BLOCK_K//2] ───────────
        # Each byte holds two consecutive INT4 values (lo=even K, hi=odd K)
        # The packed column index for K-column k is k//2
        offs_k_packed = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        b_ptrs = b_packed_base + offs_n[:, None] * stride_bn + offs_k_packed[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K // 2)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0).to(tl.uint8)  # [BLOCK_N, BLOCK_K//2]

        # ── Unpack nibbles ──────────────────────────────────────────────────
        # lo nibble (bits 0-3): even K-columns within tile → subtract 8 to get [-8,7]
        lo = (b_packed & 0x0F).to(tl.float16) - 8.0   # [BLOCK_N, BLOCK_K//2]
        # hi nibble (bits 4-7): odd K-columns within tile
        hi = ((b_packed >> 4) & 0x0F).to(tl.float16) - 8.0  # [BLOCK_N, BLOCK_K//2]

        # ── Load per-group scale ────────────────────────────────────────────
        # BLOCK_K <= GROUP_SIZE=128, so this entire K-tile is in ONE group per row.
        # Group index for k_start: k_start // GROUP_SIZE
        group_idx = k_start // GROUP_SIZE
        s_ptrs = b_scales_base + offs_n * stride_sn + group_idx
        mask_s = offs_n < N
        scales = tl.load(s_ptrs, mask=mask_s, other=1.0).to(tl.float16)  # [BLOCK_N]

        # ── Dequantize: apply per-group scale ──────────────────────────────
        # BLOCK_K <= GROUP_SIZE so all K-elements in this tile share one scale
        lo_dq = lo * scales[:, None]    # [BLOCK_N, BLOCK_K//2]
        hi_dq = hi * scales[:, None]    # [BLOCK_N, BLOCK_K//2]

        # ── Interleave lo/hi back to [BLOCK_N, BLOCK_K] ────────────────────
        # tl.interleave(a, b) interleaves along the last axis:
        #   interleave([a0,a1,...], [b0,b1,...]) = [a0,b0,a1,b1,...]
        # lo = even K cols (0,2,4,...), hi = odd K cols (1,3,5,...)
        # so interleave(lo, hi) restores the original K ordering.
        b_dq = tl.interleave(lo_dq, hi_dq).to(tl.float16)  # [BLOCK_N, BLOCK_K] fp16

        acc += tl.dot(a, tl.trans(b_dq))

    # Store output
    c_ptrs = C_ptr + (tok_start + offs_m[:, None]) * stride_cm + offs_n[None, :]
    mask_c = (offs_m[:, None] < n_tokens) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_c)


# ─── Python wrappers ──────────────────────────────────────────────────────────

def moe_int4_forward(
    hidden_states: torch.Tensor,     # [n_tokens, K]   fp16
    expert_packed: torch.Tensor,     # [n_experts, N, K//2]  uint8
    expert_scales: torch.Tensor,     # [n_experts, N, K//GROUP_SIZE]  fp16
    router_indices: torch.Tensor,    # [n_tokens, top_k]
    n_experts: int,
    group_size: int = 128,
    block_m: int = 64,
    block_n: int = 128,   # wider N tiles amortise reduced B bandwidth
    block_k: int = 32,
) -> torch.Tensor:
    """
    Single-GEMM INT4 forward (one projection only).

    Used internally by moe_int4_full_forward. Sorts tokens by expert,
    runs one INT4 GEMM with on-chip dequantization, returns sorted output.

    Returns:
        output [n_tokens * top_k, N] fp16 in sorted-expert order
    """
    assert hidden_states.is_cuda
    assert expert_packed.is_cuda
    assert expert_scales.is_cuda

    n_tokens_total, K = hidden_states.shape

    # Auto-adjust group_size and block_k for small K dimensions.
    # When K < default group_size, pack_int4_weights already used K as
    # the group_size. We must match that here.
    n_groups_actual = expert_scales.shape[2]           # [n_experts, N, n_groups]
    effective_group_size = K // n_groups_actual        # derived from actual packed scales
    if group_size != effective_group_size:
        group_size = effective_group_size

    # block_k must be <= group_size and must satisfy tl.dot's min-16 constraint
    # on the half-K dimension (block_k // 2 >= 16 → block_k >= 32).
    block_k = min(block_k, group_size)
    block_k = max(block_k, 32)   # tl.dot requires BLOCK_K//2 >= 16
    block_k = block_k if block_k % 2 == 0 else block_k - 1
    if group_size < 32:
        raise ValueError(
            f"group_size={group_size} is too small for tl.dot (minimum 32). "
            f"K={K} must be >= 64 (group_size >= 32 and BLOCK_K//2 >= 16)."
        )

    assert block_k <= group_size, f"BLOCK_K={block_k} must be <= group_size={group_size}"
    assert block_k % 2 == 0, "BLOCK_K must be even for INT4 nibble unpacking"
    assert block_k // 2 >= 16, f"BLOCK_K//2={block_k//2} must be >= 16 for tl.dot"

    _, N, K_packed = expert_packed.shape
    assert K_packed == K // 2, f"packed K={K_packed} != K//2={K//2}"

    n_active = router_indices.shape[0] * router_indices.shape[1]

    from exq.kernels.moe_grouped_gemm import sort_tokens_by_expert
    sort_order, sorted_hidden, expert_ends, max_tokens = sort_tokens_by_expert(
        router_indices, hidden_states, n_experts)

    output = torch.zeros(n_active, N, dtype=torch.float16, device="cuda")

    n_m_tiles  = max((max_tokens + block_m - 1) // block_m, 1)
    n_n_tiles  = (N + block_n - 1) // block_n
    grid = (n_experts, n_m_tiles, n_n_tiles)

    _moe_gemm_int4_kernel[grid](
        sorted_hidden,
        expert_packed,
        expert_scales,
        output,
        expert_ends,
        N=N, K=K,
        stride_am=sorted_hidden.stride(0),
        stride_be=expert_packed.stride(0),
        stride_bn=expert_packed.stride(1),
        stride_se=expert_scales.stride(0),
        stride_sn=expert_scales.stride(1),
        stride_cm=output.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE=group_size,
    )
    return output


def moe_int4_full_forward(
    hidden_states: torch.Tensor,       # [n_tokens, hidden_size]  fp16
    w13_packed: torch.Tensor,          # [n_experts, 2*inter, hidden//2]  uint8
    w13_scales: torch.Tensor,          # [n_experts, 2*inter, hidden//group_size]  fp16
    w2_packed: torch.Tensor,           # [n_experts, hidden_size, inter//2]  uint8
    w2_scales: torch.Tensor,           # [n_experts, hidden_size, inter//group_size]  fp16
    router_indices: torch.Tensor,      # [n_tokens, top_k]
    router_weights: torch.Tensor,      # [n_tokens, top_k]  fp16
    n_experts: int,
    group_size: int = 128,
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 32,
) -> torch.Tensor:
    """
    Full MoE forward: gate+up → SiLU → down → weighted combine.

    Sorts tokens once and reuses the sort order for both GEMMs, eliminating
    the duplicate argsort that the two-call approach requires. This is the
    key optimization vs calling moe_int4_forward twice.

    Args:
        hidden_states:  [n_tokens, H]
        w13_packed:     [n_experts, 2*inter, H//2]  gate+up packed INT4
        w13_scales:     [n_experts, 2*inter, H//group_size]
        w2_packed:      [n_experts, H, inter//2]    down packed INT4
        w2_scales:      [n_experts, H, inter//group_size]
        router_indices: [n_tokens, top_k]
        router_weights: [n_tokens, top_k]

    Returns:
        output [n_tokens, H] fp16
    """
    from sgl_kernel import silu_and_mul

    assert hidden_states.is_cuda and w13_packed.is_cuda

    n_tokens, H = hidden_states.shape
    _, inter2, _ = w13_packed.shape          # inter2 = 2 * intermediate_size
    inter = inter2 // 2
    n_active = n_tokens * router_indices.shape[1]

    # ── Sort tokens once (shared between GEMM1 and GEMM2) ────────────────────
    from exq.kernels.moe_grouped_gemm import sort_tokens_by_expert
    sort_order, sorted_h, ee, max_tok = sort_tokens_by_expert(
        router_indices, hidden_states, n_experts)
    unsort = torch.argsort(sort_order, stable=True)

    # Guard block sizes
    n_groups_actual   = w13_scales.shape[2]
    eff_group         = H // n_groups_actual
    gs = min(group_size, eff_group)
    bk = min(block_k, gs); bk = max(bk, 32); bk = bk if bk % 2 == 0 else bk - 1
    nm = max((max_tok + block_m - 1) // block_m, 1)

    # ── GEMM1: gate+up projection [n_active, 2*inter] ────────────────────────
    out1 = torch.zeros(n_active, inter2, dtype=torch.float16, device="cuda")
    nn1  = (inter2 + block_n - 1) // block_n
    _moe_gemm_int4_kernel[(n_experts, nm, nn1)](
        sorted_h, w13_packed, w13_scales, out1, ee,
        N=inter2, K=H,
        stride_am=sorted_h.stride(0),
        stride_be=w13_packed.stride(0), stride_bn=w13_packed.stride(1),
        stride_se=w13_scales.stride(0), stride_sn=w13_scales.stride(1),
        stride_cm=out1.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=bk, GROUP_SIZE=gs,
    )

    # ── Activation: SiLU + elementwise mul (SwiGLU) ──────────────────────────
    mid = torch.empty((n_active, inter), dtype=torch.float16, device="cuda")
    silu_and_mul(out1, mid)

    # ── GEMM2: down projection [n_active, H] ─────────────────────────────────
    # mid is already in sorted-expert order — reuse ee and max_tok
    n_groups_w2   = w2_scales.shape[2]
    eff_group_w2  = inter // n_groups_w2
    gs2 = min(group_size, eff_group_w2)
    bk2 = min(block_k, gs2); bk2 = max(bk2, 32); bk2 = bk2 if bk2 % 2 == 0 else bk2 - 1
    nm2 = max((max_tok + block_m - 1) // block_m, 1)

    out2 = torch.zeros(n_active, H, dtype=torch.float16, device="cuda")
    nn2  = (H + block_n - 1) // block_n
    _moe_gemm_int4_kernel[(n_experts, nm2, nn2)](
        mid, w2_packed, w2_scales, out2, ee,
        N=H, K=inter,
        stride_am=mid.stride(0),
        stride_be=w2_packed.stride(0), stride_bn=w2_packed.stride(1),
        stride_se=w2_scales.stride(0), stride_sn=w2_scales.stride(1),
        stride_cm=out2.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=bk2, GROUP_SIZE=gs2,
    )

    # ── Combine: weighted sum over top_k ─────────────────────────────────────
    return (
        out2[unsort].view(n_tokens, top_k, H)
        * router_weights.unsqueeze(-1).to(out2.dtype)
    ).sum(dim=1)

