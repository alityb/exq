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
ExQ uses `moe_int4_full_forward`: sorts tokens once, runs gate+up → SiLU → down
in two Triton kernel launches sharing the same sort order, then combines.
Outputs agree to <0.4% relative error (same RTN-quantized weights).

**Cache effects:** INT4 weights are 198 MB (OLMoE) and 297 MB (Qwen3) vs 6 MB
of L2 cache on the A10G — weights are 33–50× larger than L2 and always load
from HBM regardless of cache state. Warm-cache vs cold-cache (L2 flushed before
every call) differs by at most 0.8 pp across all configurations. Cache state
does not affect the comparison.

**Full production sweep — decode regime (1 token per request):**

| Batch | OLMoE SGLang | OLMoE ExQ | OLMoE Δ | Qwen3 SGLang | Qwen3 ExQ | Qwen3 Δ |
|---|---|---|---|---|---|---|
| 1 | 0.334 ms | 0.588 ms | −76% | 0.421 ms | 0.625 ms | −49% |
| 8 | 0.421 ms | 0.645 ms | −53% | 0.863 ms | 0.982 ms | −14% |
| 16 | 0.624 ms | 0.839 ms | −35% | 1.242 ms | 1.289 ms | −4% |
| 32 | 0.787 ms | 0.977 ms | −24% | 1.528 ms | 1.541 ms | ~0% |
| 64 | 1.062 ms | 1.183 ms | −11% | 1.715 ms | 1.735 ms | −1% |
| **128** | 1.893 ms | **1.355 ms** | **+28%** | 1.799 ms | 1.871 ms | −4% |
| **256** | 1.951 ms | **1.433 ms** | **+27%** | 2.868 ms | **2.150 ms** | **+25%** |
| **512** | 1.983 ms | **1.578 ms** | **+20%** | 2.934 ms | **2.696 ms** | **+8%** |

A10G, seqlen=1 (pure decode), 300-run P50.

**ExQ wins at batch≥128 for OLMoE and batch≥256 for Qwen3.** At these batch
sizes, ExQ's sorted-token dispatch pays off: tokens are grouped by expert before
the GEMM, turning scattered HBM reads into sequential accesses.

**SGLang wins at small batches.** ExQ's Python dispatch (argsort + bincount +
cumsum, ~0.27 ms) is not amortised over enough compute at low token counts.
SGLang handles dispatch in C++ with no Python boundary.

**The SGLang cliff at OLMoE batch=128** (1.06→1.89 ms, nearly 2×) is a kernel
config boundary where SGLang's autotuner selects a different block configuration
— ExQ's sorted dispatch avoids this cliff entirely.

**Cross-over point** is determined by `avg_tokens_per_expert`:
- OLMoE (top-2/64): cross-over at ~4 tokens/expert (batch≥128)
- Qwen3 (top-8/128): cross-over at ~16 tokens/expert (batch≥256)

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
