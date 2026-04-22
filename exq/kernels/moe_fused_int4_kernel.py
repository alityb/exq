"""
Fused gate+up+SiLU+down INT4 Triton kernel.

Visits weight matrices ONCE instead of TWICE, eliminating the HBM
round-trip for the intermediate gate_up tensor.

Grid: (n_experts, ceil(max_tok/BLOCK_M), ceil(H/BLOCK_N))

Inner loop over INTER dimension in BLOCK_KI tiles:
  For each INTER tile:
    1. Compute gate tile:  sorted_hidden @ w13_gate[:, INTER_tile].T
    2. Compute up tile:    sorted_hidden @ w13_up[:, INTER_tile].T
    3. Apply SiLU+mul -> mid tile  [BLOCK_M, BLOCK_KI]
    4. Accumulate:  acc_down += mid_tile @ w2[n_tile, INTER_tile].T

The inner K loop over HIDDEN is inside steps 1 and 2.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _fused_moe_int4_kernel(
    # Sorted activations
    A_ptr,           # [n_active, HIDDEN]  fp16

    # Gate+up weights  [n_experts, 2*INTER, HIDDEN//2] uint8
    W13_ptr,
    W13S_ptr,        # [n_experts, 2*INTER, HIDDEN//GS] fp16 scales

    # Down weights     [n_experts, HIDDEN, INTER//2] uint8
    W2_ptr,
    W2S_ptr,         # [n_experts, HIDDEN, INTER//GS] fp16 scales

    # Output
    C_ptr,           # [n_active, HIDDEN]  fp16

    # Expert boundaries
    ee_ptr,          # [n_experts+1] int32

    # Strides
    sa,              # A row stride
    sw13e,           # W13 expert stride
    sw13n,           # W13 row stride (= HIDDEN//2)
    sw13se,          # W13 scale expert stride
    sw13sn,          # W13 scale row stride
    sw2e,            # W2 expert stride
    sw2n,            # W2 row stride (= INTER//2)
    sw2se,           # W2 scale expert stride
    sw2sn,           # W2 scale row stride
    sc,              # C row stride

    # Dimensions (constexpr for Triton specialisation)
    HIDDEN: tl.constexpr,
    INTER:  tl.constexpr,

    # Tile sizes
    BLOCK_M:  tl.constexpr,   # token tile
    BLOCK_N:  tl.constexpr,   # output hidden tile  (for down projection)
    BLOCK_K:  tl.constexpr,   # hidden dim tile     (≤ GS_HIDDEN = 128)
    BLOCK_KI: tl.constexpr,   # inter dim tile      (≤ GS_INTER  = 128)
    GS_HIDDEN: tl.constexpr,  # group size for W13 (default 128)
    GS_INTER:  tl.constexpr,  # group size for W2  (default 128)
):
    """
    Fused: (sorted_hidden @ w13_gate.T) * silu(sorted_hidden @ w13_up.T) → mid
           mid @ w2.T → down_out
    """
    expert_id = tl.program_id(0)
    m_tile_id = tl.program_id(1)
    n_tile_id = tl.program_id(2)

    tok_start = tl.load(ee_ptr + expert_id)
    tok_end   = tl.load(ee_ptr + expert_id + 1)
    n_tokens  = tok_end - tok_start
    if n_tokens == 0:
        return

    m_start = m_tile_id * BLOCK_M
    if m_start >= n_tokens:
        return

    n_start = n_tile_id * BLOCK_N
    if n_start >= HIDDEN:
        return

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Accumulator for down projection output [BLOCK_M, BLOCK_N]
    acc_down = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # ── Loop over INTER dimension in BLOCK_KI tiles ──────────────────────────
    for ki_tile in range(tl.cdiv(INTER, BLOCK_KI)):
        ki_start = ki_tile * BLOCK_KI
        offs_ki  = ki_start + tl.arange(0, BLOCK_KI)

        # Accumulators for gate and up projections [BLOCK_M, BLOCK_KI]
        acc_gate = tl.zeros([BLOCK_M, BLOCK_KI], dtype=tl.float32)
        acc_up   = tl.zeros([BLOCK_M, BLOCK_KI], dtype=tl.float32)

        # ── Inner K loop: accumulate gate and up projections ─────────────────
        for k_tile in range(tl.cdiv(HIDDEN, BLOCK_K)):
            k_start    = k_tile * BLOCK_K
            offs_k     = k_start + tl.arange(0, BLOCK_K)
            offs_k_pak = k_start // 2 + tl.arange(0, BLOCK_K // 2)

            # Load A (activations) [BLOCK_M, BLOCK_K]
            a_ptrs = A_ptr + (tok_start + offs_m[:, None]) * sa + offs_k[None, :]
            mask_a = (offs_m[:, None] < n_tokens) & (offs_k[None, :] < HIDDEN)
            a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float16)

            # ── Gate weights [BLOCK_KI, BLOCK_K] from W13[0:INTER] ───────────
            gate_base = W13_ptr + expert_id * sw13e + offs_ki[:, None] * sw13n + offs_k_pak[None, :]
            mask_g = (offs_ki[:, None] < INTER) & (offs_k_pak[None, :] < HIDDEN // 2)
            g_pak  = tl.load(gate_base, mask=mask_g, other=0).to(tl.uint8)
            g_lo   = (g_pak & 0x0F).to(tl.float16) - 8.0
            g_hi   = ((g_pak >> 4) & 0x0F).to(tl.float16) - 8.0
            g_grp  = k_start // GS_HIDDEN
            gs_gate_ptrs = W13S_ptr + expert_id * sw13se + offs_ki * sw13sn + g_grp
            mask_gs = offs_ki < INTER
            g_scale = tl.load(gs_gate_ptrs, mask=mask_gs, other=1.0).to(tl.float16)
            g_dq = tl.interleave(g_lo * g_scale[:, None],
                                  g_hi * g_scale[:, None]).to(tl.float16)  # [BLOCK_KI, BLOCK_K]
            acc_gate = acc_gate + tl.dot(a, tl.trans(g_dq)).to(tl.float32)

            # ── Up weights [BLOCK_KI, BLOCK_K] from W13[INTER:2*INTER] ───────
            up_ki = offs_ki + INTER
            up_base = W13_ptr + expert_id * sw13e + up_ki[:, None] * sw13n + offs_k_pak[None, :]
            mask_u = (up_ki[:, None] < 2 * INTER) & (offs_k_pak[None, :] < HIDDEN // 2)
            u_pak  = tl.load(up_base, mask=mask_u, other=0).to(tl.uint8)
            u_lo   = (u_pak & 0x0F).to(tl.float16) - 8.0
            u_hi   = ((u_pak >> 4) & 0x0F).to(tl.float16) - 8.0
            gs_up_ptrs = W13S_ptr + expert_id * sw13se + up_ki * sw13sn + g_grp
            mask_us = up_ki < 2 * INTER
            u_scale = tl.load(gs_up_ptrs, mask=mask_us, other=1.0).to(tl.float16)
            u_dq = tl.interleave(u_lo * u_scale[:, None],
                                  u_hi * u_scale[:, None]).to(tl.float16)
            acc_up = acc_up + tl.dot(a, tl.trans(u_dq)).to(tl.float32)

        # ── SiLU(gate) * up → mid tile [BLOCK_M, BLOCK_KI] ──────────────────
        gate_f = acc_gate.to(tl.float32)
        silu_g = gate_f / (1.0 + tl.exp(-gate_f))
        mid = (silu_g * acc_up).to(tl.float16)   # [BLOCK_M, BLOCK_KI]

        # ── Down projection: mid @ w2[n_tile, ki_tile].T ─────────────────────
        # w2 row = n_tile selects H output dim;  col = ki_tile selects INTER input
        w2_offs_k_pak = ki_start // 2 + tl.arange(0, BLOCK_KI // 2)
        w2_base = W2_ptr + expert_id * sw2e + offs_n[:, None] * sw2n + w2_offs_k_pak[None, :]
        mask_w2 = (offs_n[:, None] < HIDDEN) & (w2_offs_k_pak[None, :] < INTER // 2)
        w2_pak  = tl.load(w2_base, mask=mask_w2, other=0).to(tl.uint8)
        w2_lo   = (w2_pak & 0x0F).to(tl.float16) - 8.0
        w2_hi   = ((w2_pak >> 4) & 0x0F).to(tl.float16) - 8.0
        w2_grp  = ki_start // GS_INTER
        w2s_ptrs = W2S_ptr + expert_id * sw2se + offs_n * sw2sn + w2_grp
        mask_w2s = offs_n < HIDDEN
        w2_scale = tl.load(w2s_ptrs, mask=mask_w2s, other=1.0).to(tl.float16)
        w2_dq = tl.interleave(w2_lo * w2_scale[:, None],
                               w2_hi * w2_scale[:, None]).to(tl.float16)  # [BLOCK_N, BLOCK_KI]

        acc_down = acc_down + tl.dot(mid, tl.trans(w2_dq)).to(tl.float32)

    # Store output
    c_ptrs = C_ptr + (tok_start + offs_m[:, None]) * sc + offs_n[None, :]
    mask_c = (offs_m[:, None] < n_tokens) & (offs_n[None, :] < HIDDEN)
    tl.store(c_ptrs, acc_down.to(tl.float16), mask=mask_c)


def fused_moe_int4(
    sorted_hidden: torch.Tensor,   # [n_active, HIDDEN]
    w13_packed:    torch.Tensor,   # [n_experts, 2*INTER, HIDDEN//2]  uint8
    w13_scales:    torch.Tensor,   # [n_experts, 2*INTER, HIDDEN//GS] fp16
    w2_packed:     torch.Tensor,   # [n_experts, HIDDEN,  INTER//2]   uint8
    w2_scales:     torch.Tensor,   # [n_experts, HIDDEN,  INTER//GS]  fp16
    expert_ends:   torch.Tensor,   # [n_experts+1] int32
    max_tok:       int,
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 32,
    block_ki: int = 32,
    gs_hidden: int = 128,
    gs_inter:  int = 128,
) -> torch.Tensor:
    n_active, HIDDEN = sorted_hidden.shape
    n_experts, _, _  = w13_packed.shape
    INTER = w2_packed.shape[2] * 2   # w2: [E, HIDDEN, INTER//2]

    out = torch.zeros(n_active, HIDDEN, dtype=torch.float16, device=sorted_hidden.device)

    nm = max((max_tok + block_m - 1) // block_m, 1)
    nn = (HIDDEN + block_n - 1) // block_n
    grid = (n_experts, nm, nn)

    _fused_moe_int4_kernel[grid](
        sorted_hidden,
        w13_packed, w13_scales,
        w2_packed,  w2_scales,
        out,
        expert_ends,
        sorted_hidden.stride(0),
        w13_packed.stride(0), w13_packed.stride(1),
        w13_scales.stride(0), w13_scales.stride(1),
        w2_packed.stride(0),  w2_packed.stride(1),
        w2_scales.stride(0),  w2_scales.stride(1),
        out.stride(0),
        HIDDEN=HIDDEN, INTER=INTER,
        BLOCK_M=block_m, BLOCK_N=block_n,
        BLOCK_K=block_k, BLOCK_KI=block_ki,
        GS_HIDDEN=gs_hidden, GS_INTER=gs_inter,
    )
    return out
