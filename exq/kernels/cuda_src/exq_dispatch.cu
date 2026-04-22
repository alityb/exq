/*
 * exq_moe_fused.cu — fused gate+up+silu+down INT4 MoE kernel
 *
 * Combines the three separate operations:
 *   1. GEMM1: sorted_hidden @ w13.T  →  gate_up  [n_active, 2*INTER]
 *   2. SiLU+mul: gate_up → mid  [n_active, INTER]
 *   3. GEMM2: mid @ w2.T  →  down_out  [n_active, H]
 *
 * into a SINGLE kernel that:
 *   - loads w13 tiles and w2 tiles in the same thread block
 *   - computes gate_up in SRAM, applies SiLU+mul immediately
 *   - feeds mid directly to GEMM2 accumulation
 *   - never writes intermediate results to HBM
 *
 * This eliminates:
 *   - 2 kernel launches → 1 kernel launch
 *   - 2x HBM writes (gate_up, mid) + 2x HBM reads
 *
 * Layout:
 *   w13_packed  [n_experts, 2*INTER, HIDDEN//2]  uint8
 *   w13_scales  [n_experts, 2*INTER, HIDDEN//128] fp16
 *   w2_packed   [n_experts, HIDDEN, INTER//2]     uint8
 *   w2_scales   [n_experts, HIDDEN, INTER//128]   fp16
 *
 * Grid: (n_experts, ceil(max_tok/BLOCK_M), ceil(H/BLOCK_N))
 * Each block produces a [BLOCK_M, BLOCK_N] tile of down_out.
 *
 * Strategy per block:
 *   acc_down = 0  [BLOCK_M, BLOCK_N]  fp32
 *   for each INTER tile j:
 *       // compute mid tile [BLOCK_M, BLOCK_K_INTER]
 *       acc_gate = 0  [BLOCK_M, BLOCK_K_INTER]
 *       acc_up   = 0  [BLOCK_M, BLOCK_K_INTER]
 *       for k in 0..HIDDEN step BLOCK_K:
 *           a = load sorted_hidden[m_tile, k]
 *           g = dequant w13[expert, j, k]               // gate half
 *           u = dequant w13[expert, INTER+j, k]         // up half
 *           acc_gate += a @ g.T
 *           acc_up   += a @ u.T
 *       mid_tile = silu(acc_gate) * acc_up
 *       // now use mid_tile as A for down projection
 *       d = dequant w2[expert, n_tile, j]
 *       acc_down += mid_tile @ d.T
 *   store acc_down → down_out
 *
 * NOTE: BLOCK_K_INTER must be a power of 2 ≥ 16 for tl.dot.
 *       BLOCK_K (hidden dim tile) must be ≤ group_size_hidden=128.
 *       We use BLOCK_K_INTER = 32.  BLOCK_K = 32.
 */

#include <cub/cub.cuh>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cmath>

// We implement this in Triton (Python side) because writing
// a fused GEMM1+silu+GEMM2 in raw CUDA correctly for all tile sizes
// is hundreds of lines.  The key insight: Triton's JIT specialises
// on (BLOCK_M, BLOCK_N, BLOCK_K_INTER, BLOCK_K) at compile time,
// so the fused kernel is generated efficiently.
//
// This file provides the dispatch primitives (CUDA) that eliminate
// the Python overhead. The fused GEMM kernel is in exq_fused_kernel.py.
// ─────────────────────────────────────────────────────────────────────────────

// ── build_ends ────────────────────────────────────────────────────────────────
__global__ void build_ends_kernel(
    const int32_t* __restrict__ sorted_expert_ids,
    int32_t*       __restrict__ expert_ends,
    int32_t n_active
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_active) return;
    atomicAdd(&expert_ends[sorted_expert_ids[i] + 1], 1);
}

// ── build_ends_from_slot_ids ─────────────────────────────────────────────────
// Variant that takes flat slot IDs (from sgl_moe_align) and a flat
// router index array, and atomically builds expert_ends.
// slot_ids[i] = original flat slot index (< n_active) or padding (>= n_active)
// flat_router[slot_id] = expert_id for that slot
__global__ void build_ends_from_slots_kernel(
    const int32_t* __restrict__ slot_ids,    // [EM]  from moe_align, may include padding
    const int32_t* __restrict__ flat_router, // [n_active]  r_idx.reshape(-1)
    int32_t*       __restrict__ expert_ends, // [n_experts+1]  zeroed by caller
    int32_t n_active,
    int32_t EM
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= EM) return;
    int32_t slot = slot_ids[i];
    if (slot >= n_active) return;   // padding token
    int32_t expert = flat_router[slot];
    atomicAdd(&expert_ends[expert + 1], 1);
}

// ── exq_dispatch ─────────────────────────────────────────────────────────────
void exq_dispatch(
    torch::Tensor router_indices,
    torch::Tensor sort_order,
    torch::Tensor sorted_expert_ids,
    torch::Tensor expert_ends,
    int64_t n_experts
) {
    auto stream  = at::cuda::getCurrentCUDAStream();
    auto flat    = router_indices.reshape({-1}).to(torch::kInt32).contiguous();
    int64_t n    = flat.numel();
    auto arange  = torch::arange(n,
        torch::TensorOptions().dtype(torch::kInt32).device(flat.device()));
    auto skeys   = torch::empty({n}, flat.options());
    auto svals   = torch::empty({n}, arange.options());

    int bits = (int)std::ceil(std::log2((double)n_experts + 1));
    void* tmp = nullptr; size_t tmp_b = 0;
    cub::DeviceRadixSort::SortPairs(tmp, tmp_b,
        flat.data_ptr<int32_t>(), skeys.data_ptr<int32_t>(),
        arange.data_ptr<int32_t>(), svals.data_ptr<int32_t>(),
        (int)n, 0, bits, stream);
    auto tbuf = torch::empty({(int64_t)tmp_b},
        torch::TensorOptions().dtype(torch::kUInt8).device(flat.device()));
    tmp = tbuf.data_ptr();
    cub::DeviceRadixSort::SortPairs(tmp, tmp_b,
        flat.data_ptr<int32_t>(), skeys.data_ptr<int32_t>(),
        arange.data_ptr<int32_t>(), svals.data_ptr<int32_t>(),
        (int)n, 0, bits, stream);

    sort_order.copy_(svals.to(torch::kInt64));
    sorted_expert_ids.copy_(skeys.to(torch::kInt64));

    expert_ends.zero_();
    int thr = 256, blk = ((int)n + thr - 1) / thr;
    build_ends_kernel<<<blk, thr, 0, stream>>>(
        skeys.data_ptr<int32_t>(),
        expert_ends.data_ptr<int32_t>(), (int32_t)n);

    void* tmp2 = nullptr; size_t tmp2_b = 0;
    cub::DeviceScan::InclusiveSum(tmp2, tmp2_b,
        expert_ends.data_ptr<int32_t>() + 1,
        expert_ends.data_ptr<int32_t>() + 1,
        (int)n_experts, stream);
    auto t2 = torch::empty({(int64_t)tmp2_b},
        torch::TensorOptions().dtype(torch::kUInt8).device(flat.device()));
    tmp2 = t2.data_ptr();
    cub::DeviceScan::InclusiveSum(tmp2, tmp2_b,
        expert_ends.data_ptr<int32_t>() + 1,
        expert_ends.data_ptr<int32_t>() + 1,
        (int)n_experts, stream);
}

// ── compact_valid_slots_kernel ────────────────────────────────────────────────
// Write 1 for valid (non-padding) slots, 0 for padding.
// Then use CUB scan + scatter to compact in order.
// But since we don't want CUB overhead, use a different strategy:
//
// Direct approach: each valid slot at position i writes to output
// at position (its index in the per-expert sorted list) using expert_ends.
// Since sorted_slot_ids is sorted by expert already, we can use a
// per-expert atomic counter.
__global__ void compact_via_expert_counter_kernel(
    const int32_t* __restrict__ slot_ids,        // [EM]
    const int32_t* __restrict__ flat_router,     // [n_active]
    const int32_t* __restrict__ expert_ends,     // [n_experts+1]
    int32_t*       __restrict__ expert_cursors,  // [n_experts] atomic cursors (init = expert_ends[0..n-1])
    int64_t*       __restrict__ sort_order,      // [n_active] output
    int32_t n_active,
    int32_t EM
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= EM) return;
    int32_t slot = slot_ids[i];
    if (slot >= n_active) return;  // padding

    int32_t expert = flat_router[slot];
    // Atomically claim the next position for this expert
    int pos = atomicAdd(&expert_cursors[expert], 1);
    sort_order[pos] = (int64_t)slot;
}

// ── exq_build_ends_from_slots ─────────────────────────────────────────────────
void exq_build_ends_from_slots(
    torch::Tensor sorted_slot_ids,  // [EM] int32  from moe_align
    torch::Tensor flat_router,      // [n_active] int32  = r_idx.reshape(-1)
    torch::Tensor expert_ends,      // [n_experts+1] int32  output (pre-zeroed)
    torch::Tensor sort_order,       // [n_active] int64  output
    int64_t n_active,
    int64_t n_experts
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t EM  = sorted_slot_ids.numel();

    // Step 1: build expert_ends histogram from slot → expert lookup
    expert_ends.zero_();
    int thr = 256, blk = ((int)EM + thr - 1) / thr;
    build_ends_from_slots_kernel<<<blk, thr, 0, stream>>>(
        sorted_slot_ids.data_ptr<int32_t>(),
        flat_router.data_ptr<int32_t>(),
        expert_ends.data_ptr<int32_t>(),
        (int32_t)n_active, (int32_t)EM);

    // Step 2: prefix sum for expert_ends
    void* tmp2 = nullptr; size_t tmp2_b = 0;
    cub::DeviceScan::InclusiveSum(tmp2, tmp2_b,
        expert_ends.data_ptr<int32_t>() + 1,
        expert_ends.data_ptr<int32_t>() + 1,
        (int)n_experts, stream);
    auto t2 = torch::empty({(int64_t)tmp2_b},
        torch::TensorOptions().dtype(torch::kUInt8)
            .device(sorted_slot_ids.device()));
    tmp2 = t2.data_ptr();
    cub::DeviceScan::InclusiveSum(tmp2, tmp2_b,
        expert_ends.data_ptr<int32_t>() + 1,
         expert_ends.data_ptr<int32_t>() + 1,
         (int)n_experts, stream);

    // Step 3: compact sorted_slot_ids → sort_order using per-expert cursors.
    // expert_ends[e] = start of expert e's range in sort_order.
    // Each valid slot atomically claims the next slot in its expert's range.
    // Result: sort_order is filled in expert-boundary-respecting order.
    auto cursors = expert_ends.slice(0, 0, n_experts).clone();
    int thr3 = 256, blk3 = ((int)EM + thr3 - 1) / thr3;
    compact_via_expert_counter_kernel<<<blk3, thr3, 0, stream>>>(
        sorted_slot_ids.data_ptr<int32_t>(),
        flat_router.data_ptr<int32_t>(),
        expert_ends.data_ptr<int32_t>(),
        cursors.data_ptr<int32_t>(),
        sort_order.data_ptr<int64_t>(),
        (int32_t)n_active, (int32_t)EM);
}



// ── exq_gather_hidden ─────────────────────────────────────────────────────────
__global__ void gather_hidden_vec_kernel(
    const __half*  __restrict__ hidden,
    const int64_t* __restrict__ sort_order,
    __half*        __restrict__ out,
    int top_k, int H, int n_active
) {
    int i = blockIdx.x;
    if (i >= n_active) return;
    int tok = (int)(sort_order[i] / top_k);
    const float4* src = reinterpret_cast<const float4*>(hidden + (int64_t)tok * H);
    float4*       dst = reinterpret_cast<float4*>(out + (int64_t)i * H);
    for (int t = threadIdx.x; t < H/8; t += blockDim.x) dst[t] = src[t];
}

void exq_gather_hidden(
    torch::Tensor hidden, torch::Tensor sort_order,
    torch::Tensor out, int64_t top_k
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    int n = (int)sort_order.numel(), H = (int)hidden.size(1);
    int thr = std::min(H/8, 256);
    gather_hidden_vec_kernel<<<n, thr, 0, stream>>>(
        reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>()),
        sort_order.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        (int)top_k, H, n);
}

// ── exq_combine ───────────────────────────────────────────────────────────────
__global__ void scatter_combine_fp32_kernel(
    const __half*   __restrict__ down_out,
    const int64_t*  __restrict__ sort_order,
    const __half*   __restrict__ weights,
    float*          __restrict__ result_f32,
    int n_active, int top_k, int H
) {
    int i = blockIdx.x;
    if (i >= n_active) return;
    int64_t s  = sort_order[i];
    int tok    = (int)(s / top_k);
    float w    = __half2float(weights[s]);
    for (int h = threadIdx.x; h < H; h += blockDim.x)
        atomicAdd(&result_f32[(int64_t)tok * H + h],
                  __half2float(down_out[(int64_t)i * H + h]) * w);
}

__global__ void fp32_to_fp16_kernel(const float* s, __half* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2half(s[i]);
}

void exq_combine(
    torch::Tensor down_out, torch::Tensor sort_order,
    torch::Tensor router_weights, torch::Tensor result
) {
    auto stream  = at::cuda::getCurrentCUDAStream();
    int n_active = (int)sort_order.numel();
    int n_tokens = (int)router_weights.size(0);
    int top_k    = (int)router_weights.size(1);
    int H        = (int)down_out.size(1);
    auto rf32    = torch::zeros({n_tokens, H},
        torch::TensorOptions().dtype(torch::kFloat32).device(down_out.device()));

    int thr = std::min(H, 256);
    scatter_combine_fp32_kernel<<<n_active, thr, 0, stream>>>(
        reinterpret_cast<const __half*>(down_out.data_ptr<at::Half>()),
        sort_order.data_ptr<int64_t>(),
        reinterpret_cast<const __half*>(router_weights.data_ptr<at::Half>()),
        rf32.data_ptr<float>(),
        n_active, top_k, H);

    int tot = n_tokens * H;
    fp32_to_fp16_kernel<<<(tot+255)/256, 256, 0, stream>>>(
        rf32.data_ptr<float>(),
        reinterpret_cast<__half*>(result.data_ptr<at::Half>()),
        tot);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch",               &exq_dispatch);
    m.def("build_ends_from_slots",  &exq_build_ends_from_slots,
          "Build expert_ends from moe_align_block_size sorted_slot_ids output");
    m.def("gather_hidden", &exq_gather_hidden);
    m.def("combine",       &exq_combine);
}
