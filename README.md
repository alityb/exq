# ExQ — Expert Quantization for MoE Inference

ExQ is a compiler that reads offline routing statistics from a MoE model and uses them to make two decisions at compile time:

1. **Precision assignment** — frequently-activated experts keep BF16 or INT8; rarely-activated experts get INT4. This recovers quality lost by uniform INT4 quantization while staying at the same memory budget.
2. **Kernel dispatch** — a Triton kernel that loads packed INT4 weights (two values per byte, per-group fp16 scales) and dequantizes on-chip. This reduces HBM traffic by 3.88× vs fp16 weights, which translates directly to lower decode latency on memory-bound hardware.

These are separate contributions. The compiler assignment improves quality over uniform INT4 at the same memory. The INT4 kernel improves speed over fp16 at lower quality. Together they describe a Pareto-better point than either uniform INT4 or fp16 serving.

## Install

```bash
pip install .
```

Requires Python 3.10+, PyTorch 2.0+, and a Rust toolchain (maturin builds the compiler core).

```bash
pip install ".[profile]"   # transformers, datasets, accelerate
pip install ".[ilp]"       # ortools for the CP-SAT joint optimizer
```

## How it works

1. **Profile** — run the model on a calibration corpus with routing hooks enabled. Records per-expert activation frequency and cross-layer co-activation probability.
2. **Compile** — the routing profile becomes a graph IR. Four compiler passes assign memory layout (Pass A), per-expert precision (Pass B), prefetch schedules (Pass C), and layer specializations (Pass D).
3. **Emit** — write a compiled artifact: per-expert precision assignments, a static prefetch schedule, and layout metadata.
4. **Serve** — ExQ patches SGLang's MoE kernel dispatch to use the INT4 Triton kernel. No prediction model runs at inference time.

## Usage

### One command

```bash
exq serve --model Qwen/Qwen3-30B-A3B
```

Profiles (if needed), compiles, patches SGLang, and starts an OpenAI-compatible server on port 30000.

### Three commands

```bash
exq profile --model allenai/OLMoE-1B-7B-0924 --output profiles/olmoe.json
exq compile --profile profiles/olmoe.json     --output artifacts/olmoe.json
exq serve   --model allenai/OLMoE-1B-7B-0924 --artifact artifacts/olmoe.json
```

### Python API

```python
# Profile
from exq.profiler import CalibrationRunner
runner  = CalibrationRunner("allenai/OLMoE-1B-7B-0924", n_samples=512)
profile = runner.run(output_path="profiles/olmoe.json")

# Compile  (or: exq compile --profile profiles/olmoe.json)
from exq import CompilerPipeline, RoutingProfile, py_build_routing_graph
profile    = RoutingProfile.load("profiles/olmoe.json")
graph      = py_build_routing_graph(profile)
pipeline   = CompilerPipeline()
pipeline.run_auto(graph, n_experts=64, top_k=2)
quant_plan = pipeline.get_quant_plan()   # {(layer, expert): "BF16"|"INT8"|"INT4"}

# Patch SGLang and serve  (or: exq serve --model ...)
from exq.runtime.sglang_backend import patch_sglang
patch_sglang("artifacts/olmoe.json")
# SGLang now routes all covered MoE layers through the ExQ INT4 kernel
```

### Evaluation scripts

```bash
# Perplexity: fp16 vs ExQ vs uniform INT4 (fair quality comparison)
python scripts/eval_ppl.py --model allenai/OLMoE-1B-7B-0924 \
    --precision exq --quant-plan artifacts/olmoe.json --dataset wikitext2

# Compile-time diagnostic: will ExQ help on this model?
python scripts/exq_diagnose.py --profile profiles/olmoe.json

# Kernel speed benchmark: ExQ INT4 vs fp16 baseline (fair speed comparison)
python scripts/bench_int4.py --model olmoe

# End-to-end SGLang benchmark
python scripts/bench_sglang_integration.py --model olmoe

# ILP vs greedy compiler (requires pip install ".[ilp]")
python scripts/compare_greedy_vs_ilp.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --profile profiles/olmoe-1b-7b-0924-256.json \
    --greedy-artifact artifacts/olmoe-1b-7b-0924-256.json
```

## Architecture

```
exq/
  __main__.py             CLI entry point (exq profile / compile / serve)
  model_utils.py          Shared: layer discovery, model loading, artifact I/O
  kernels/                INT4 Triton kernels, artifact reader, MoE dispatch
  profiler/               Phase 1: routing instrumentation (MoE + dense)
  compiler/               Dense quant planner, CP-SAT joint scheduler (optional)
  eval/                   Perplexity, KL divergence, coverage analysis, benchmarks
  runtime/                SGLang backend, transformers integration, prefetch engine
  codegen/                Triton kernel emission from compiled artifacts

src/  (Rust, compiled via maturin as exq._core)
  profile.rs              RoutingProfile, LayerProfile
  ir/                     RoutingGraph, GraphBuilder, GraphAnalysis
  passes/                 Pass A–D: layout, quant, prefetch, specialization
  codegen/                CompiledArtifact, kernel pseudocode
```

The Rust core handles the full compiler pipeline. Python handles model loading, profiling hooks, quantization application, the SGLang patch, and evaluation.

## Results

### Kernel speed — ExQ INT4 vs SGLang INT4 (same weights, same precision)

**The fair comparison:** both kernels receive the exact same RTN-packed INT4
weights (uint8, two values per byte, group_size=128 per-group fp16 scales).
SGLang uses `fused_experts_impl` with `use_int4_w4a16=True, block_shape=[0,128]`,
dispatching to its `fused_moe_kernel_gptq_awq` Triton kernel.
ExQ uses a CUDA dispatch extension + two Triton GEMMs:
- `sgl_kernel.moe_align_block_size` (20 μs) — same kernel SGLang uses internally
- `exq_dispatch_cuda.build_ends_from_slots` (48 μs) — builds expert boundary array
- Two Triton INT4 GEMMs with on-chip dequantization + CUDA combine

Outputs agree to <0.4% relative error (same RTN-quantized weights).

**Cache effects:** INT4 weights are 198 MB (OLMoE) and 297 MB (Qwen3) vs 6 MB
of L2 cache — weights always load from HBM. Warm vs cold cache differs ≤0.8 pp.

**Full production sweep — decode regime (1 token per request):**

| Batch | OLMoE SGLang | OLMoE ExQ | OLMoE Δ | Qwen3 SGLang | Qwen3 ExQ | Qwen3 Δ |
|---|---|---|---|---|---|---|
| 1 | 0.322 ms | 0.407 ms | −26% | 0.420 ms | 0.453 ms | −8% |
| 4 | 0.342 ms | 0.472 ms | −38% | 0.544 ms | 0.591 ms | −9% |
| 8 | 0.434 ms | 0.482 ms | −11% | 0.838 ms | **0.803 ms** | **+4%** |
| 16 | 0.579 ms | 0.608 ms | −5% | 1.127 ms | **1.038 ms** | **+8%** |
| 32 | 0.840 ms | **0.801 ms** | **+5%** | 1.508 ms | **1.335 ms** | **+12%** |
| 64 | 1.121 ms | **1.065 ms** | **+5%** | 1.743 ms | **1.585 ms** | **+9%** |
| 128 | 1.891 ms | **1.181 ms** | **+38%** | 1.790 ms | **1.688 ms** | **+6%** |
| 256 | 1.950 ms | **1.262 ms** | **+35%** | 2.864 ms | **1.922 ms** | **+33%** |
| 512 | 1.980 ms | **1.402 ms** | **+29%** | 2.928 ms | **2.422 ms** | **+17%** |

A10G, seqlen=1 (pure decode), 300-run P50.

**ExQ wins at batch≥32 for OLMoE and batch≥8 for Qwen3.**

ExQ still loses at small batch sizes for OLMoE (batch 1–16) because SGLang's
kernel fuses gate+up+SiLU+down into one Triton kernel launch; ExQ uses two
separate kernel launches. At very small token counts, the per-launch overhead
exceeds the sorted-access benefit.

**Cross-over point** is determined by `avg_tokens_per_expert`:
- OLMoE (top-2/64): cross-over at ~1 token/expert (batch≥32)
- Qwen3 (top-8/128): cross-over at ~0.5 tokens/expert (batch≥8)


**Cache effects:** INT4 weights are 198 MB (OLMoE) and 297 MB (Qwen3) vs 6 MB
of L2 cache on the A10G — weights are 33–50× larger than L2 and always load
from HBM. Warm vs cold cache (L2 flushed before every call) differs by ≤0.8 pp.

**Full production sweep — decode regime (1 token per request):**

| Batch | OLMoE SGLang | OLMoE ExQ | OLMoE Δ | Qwen3 SGLang | Qwen3 ExQ | Qwen3 Δ |
|---|---|---|---|---|---|---|
| 1 | 0.329 ms | 0.435 ms | −32% | 0.415 ms | 0.469 ms | −13% |
| 4 | 0.342 ms | 0.491 ms | −44% | 0.569 ms | 0.616 ms | −8% |
| 8 | 0.414 ms | 0.494 ms | −19% | 0.855 ms | **0.823 ms** | **+4%** |
| 16 | 0.619 ms | 0.690 ms | −11% | 1.232 ms | **1.130 ms** | **+8%** |
| 32 | 0.781 ms | 0.818 ms | −5% | 1.520 ms | **1.386 ms** | **+9%** |
| 64 | 1.053 ms | **1.028 ms** | **+2%** | 1.709 ms | **1.580 ms** | **+8%** |
| 128 | 1.887 ms | **1.200 ms** | **+36%** | 1.798 ms | **1.707 ms** | **+5%** |
| 256 | 1.949 ms | **1.280 ms** | **+34%** | 2.864 ms | **1.942 ms** | **+32%** |
| 512 | 1.978 ms | **1.419 ms** | **+28%** | 2.929 ms | **2.441 ms** | **+17%** |

A10G, seqlen=1 (pure decode), 300-run P50.

**ExQ wins at batch≥64 for OLMoE and batch≥8 for Qwen3.**

The implementation uses three CUDA kernels:
- `exq_dispatch`: CUB radix sort + atomicAdd histogram + CUB prefix sum — 46 μs (was 278 μs in Python)
- `exq_gather_hidden`: vectorised float4 gather — 12 μs
- `exq_combine`: scatter-add with fp32 accumulation — 27 μs

**Where ExQ still loses (small batches for OLMoE):** SGLang's
`fused_moe_kernel_gptq_awq` fuses gate+up+SiLU+down into a single Triton
kernel, visiting the weight matrices once. ExQ uses two separate kernel
launches. At very small token counts (batch 1–32 for OLMoE) the two-launch
overhead exceeds the sorted-access benefit. Eliminating this would require
a CUDA-native fused gate+up+SiLU+down kernel.

**Cross-over point** is determined by `avg_tokens_per_expert`:
- OLMoE (top-2/64): cross-over at ~2 tokens/expert (batch≥64)
- Qwen3 (top-8/128): cross-over at ~0.5 tokens/expert (batch≥8)

### Quality — ExQ mixed-prec vs uniform INT4 (same memory)

**The fair comparison:** uniform INT4 at the same total memory footprint.
ExQ assigns BF16/INT8 to frequently-activated experts and INT4 to the rest;
the memory budget is identical.

Recovery = fraction of the fp16→INT4 quality gap that ExQ eliminates:

| Model | Type | Recovery | quant_diff | Notes |
|---|---|---|---|---|
| OLMoE-1B-7B | MoE | **53.9%** | 0.429 | Strong routing concentration |
| Qwen2.5-3B | Dense | **66.8%** | 0.377 | |
| Qwen2.5-1.5B | Dense | **62.2%** | 0.345 | |
| Qwen1.5-MoE | MoE | 1.4% | 0.010 | Near-uniform routing |
| DeepSeek-V2-Lite | MoE | −0.3% | 0.003 | Near-uniform routing |
| GLM-4.7-Flash | MoE | −14.2% | 0.119 | INT4 already close to fp16 |

`quant_diff` = fraction of experts at INT8 or BF16. Recovery requires routing
concentration: models where a small subset of experts handles most tokens can
meaningfully assign higher precision to those experts. Near-uniform routing
leaves no headroom.

Output distribution (KL divergence vs fp16, Qwen2.5-3B, WikiText2):

| Method | Mean KL | P99 KL |
|---|---|---|
| Uniform INT4 | 0.04138 | 0.35526 |
| ExQ mixed-prec | **0.01780** | **0.17469** |
| AWQ (controlled) | 0.12212 | 1.25371 |

ExQ mean KL is **2.3× lower than uniform INT4** and **6.9× lower than AWQ**
at the same memory budget.

### Compile time

All models compile in under 3 seconds on a single CPU core:

| Model | Nodes | Compile time |
|---|---|---|
| Qwen2.5-1.5B (Dense) | 336 | 1.74 s |
| Qwen2.5-3B (Dense) | 576 | 1.75 s |
| OLMoE-1B-7B (MoE) | 1,024 | 1.73 s |
| Qwen1.5-MoE (MoE) | 1,440 | 2.47 s |
| DeepSeek-V2-Lite (MoE) | 1,664 | 2.87 s |
| GLM-4.7-Flash (MoE) | 2,944 | 1.31 s |
| Qwen3-30B-A3B (MoE) | 6,144 | 1.79 s |

## License

Apache-2.0
