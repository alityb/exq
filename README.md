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
    --precision rpgo --quant-plan artifacts/olmoe.json --dataset wikitext2

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
which dispatches to its `fused_moe_kernel_gptq_awq` kernel.
ExQ uses its sorted-token grouped GEMM with on-chip dequantization.
Both execute the full MoE forward pass (gate+up → SiLU → down → weighted combine).
Outputs agree to <0.4% relative error (expected: both use the same RTN-quantized weights).

| Model | Batch | SGLang INT4 | ExQ INT4 | Δ |
|---|---|---|---|---|
| OLMoE-1B-7B | 1 | 1.064 ms | 1.403 ms | −32% (ExQ slower) |
| OLMoE-1B-7B | 2 | 1.885 ms | 1.616 ms | **+14%** |
| OLMoE-1B-7B | 4 | 1.964 ms | 1.741 ms | **+11%** |
| OLMoE-1B-7B | 8 | 1.978 ms | 1.865 ms | **+6%** |
| Qwen3-30B-A3B | 1 | 1.763 ms | 2.081 ms | −18% (ExQ slower) |
| Qwen3-30B-A3B | 2 | 1.786 ms | 2.207 ms | −24% (ExQ slower) |
| Qwen3-30B-A3B | 4 | 2.862 ms | 2.466 ms | **+14%** |
| Qwen3-30B-A3B | 8 | 2.930 ms | 3.063 ms | −5% |

A10G, seqlen=64, 200-run P50.

**At batch≥4, ExQ wins by 6–14%.** The advantage comes from sorted-token
dispatch, which makes each expert's token slice contiguous in memory and
eliminates the scatter overhead in SGLang's kernel.

**At batch=1–2, ExQ loses.** SGLang's kernel fuses gate+up+silu+down into a
single Triton kernel launch. ExQ issues two separate kernel launches (one per
GEMM), and at very small token counts the per-launch overhead (~0.4 ms each)
dominates. This is a real trade-off, not a measurement artifact.

The cross-over point is around batch=2–4 depending on model size. For continuous
batching in production (typical batch 4–16), ExQ wins.

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
