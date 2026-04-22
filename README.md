# ExQ

Compiler framework that uses MoE routing statistics to make quantization, prefetching, and memory layout decisions at compile time instead of at runtime.

Standard MoE inference systems predict which experts to load at runtime, token by token, using a small model that adds latency. ExQ profiles expert routing behavior offline, then compiles a static artifact where every optimization decision is already made. The inference runtime executes a fixed schedule instead of running a predictor.

The INT4 Triton kernel eliminates the per-token prediction model. Static prefetch schedules (Pass C) currently run via Python forward hooks, which adds ~2-3ms/token dispatch overhead; a native CUDA implementation would eliminate this.

## Install

```bash
pip install .
```

Requires Python 3.10+, PyTorch 2.0+, and a Rust toolchain (maturin builds the core).

```bash
pip install ".[profile]"   # adds transformers, datasets, accelerate
pip install ".[ilp]"       # adds ortools for the CP-SAT joint optimizer
```

## How it works

1. **Profile** the model on a calibration corpus. Records which experts activate, how often, and which co-activate across layers.
2. **Compile** the profile into a routing graph IR. Four passes run over this graph: memory layout (Pass A), quantization assignment (Pass B), prefetch scheduling (Pass C), and layer specialization (Pass D).
3. **Emit** a compiled artifact (JSON) containing per-expert precision assignments, a static prefetch schedule, and layout metadata.
4. **Serve** with the compiled artifact. ExQ patches SGLang's MoE dispatch to use the INT4 Triton kernel. No per-token prediction model runs during inference.

## Usage

### One command

```bash
exq serve --model Qwen/Qwen3-30B-A3B
```

Profiles the model if no profile exists, compiles an artifact, applies the ExQ INT4 patch to SGLang's MoE kernel, and starts an OpenAI-compatible server on port 30000.

### Three commands

```bash
exq profile --model allenai/OLMoE-1B-7B-0924 --output profiles/olmoe.json
exq compile --profile profiles/olmoe.json --output artifacts/olmoe.json
exq serve   --model allenai/OLMoE-1B-7B-0924 --artifact artifacts/olmoe.json
```

### Python API

```python
# Profile
from exq.profiler import CalibrationRunner
runner = CalibrationRunner("allenai/OLMoE-1B-7B-0924", n_samples=512)
profile = runner.run(output_path="profiles/olmoe.json")

# Compile  (or use: exq compile --profile profiles/olmoe.json)
from rpgo import CompilerPipeline, RoutingProfile, py_build_routing_graph
profile  = RoutingProfile.load("profiles/olmoe.json")
graph    = py_build_routing_graph(profile)
pipeline = CompilerPipeline()
pipeline.run_auto(graph, n_experts=64, top_k=2)
quant_plan = pipeline.get_quant_plan()   # {(layer, expert): "BF16"|"INT8"|"INT4"}

# Patch SGLang and serve  (or use: exq serve --model ...)
from exq.runtime.sglang_backend import patch_sglang
patch_sglang("artifacts/olmoe.json")
# launch SGLang normally; ExQ INT4 kernel is now active
```

### Evaluation scripts

```bash
# Perplexity: fp16 vs ExQ vs uniform INT4
python scripts/eval_ppl.py --model allenai/OLMoE-1B-7B-0924 --precision rpgo \
    --quant-plan artifacts/olmoe.json --dataset wikitext2

# Diagnostic: will ExQ help on this model?
python scripts/exq_diagnose.py --profile profiles/olmoe.json

# Latency benchmark: INT4 Triton kernel vs BF16 baseline
python scripts/bench_int4.py --model olmoe

# SGLang integration benchmark
python scripts/bench_sglang_integration.py --model olmoe

# ILP vs greedy compiler comparison (requires pip install ".[ilp]")
python scripts/compare_greedy_vs_ilp.py --model allenai/OLMoE-1B-7B-0924 \
    --profile profiles/olmoe-1b-7b-0924-256.json \
    --greedy-artifact artifacts/olmoe-1b-7b-0924-256.json
```

## Architecture

```
rpgo/
  __main__.py             CLI entry point (exq profile / compile / serve)
  model_utils.py          Shared: layer discovery, model loading, artifact I/O
  kernels/                INT4 Triton kernels, artifact reader, MoE dispatch
  profiler/               Phase 1: routing instrumentation (MoE + dense)
  compiler/               Dense quant planner, CP-SAT joint scheduler (optional)
  eval/                   Perplexity, KL divergence, coverage analysis, benchmarks
  runtime/                SGLang backend, transformers integration, prefetch engine
  codegen/                Triton kernel emission from compiled artifacts

src/  (Rust, compiled via maturin as rpgo._core)
  profile.rs              RoutingProfile, LayerProfile
  ir/                     RoutingGraph, GraphBuilder, GraphAnalysis
  passes/                 Pass A-D: layout, quant, prefetch, specialization
  codegen/                CompiledArtifact, kernel pseudocode
```

The Rust core handles the compiler pipeline. Python handles model loading, profiling hooks, quantization application, the SGLang patch, and evaluation.

## Results

### SGLang integration (+19.8% / +13.6%)

ExQ integrates with SGLang as a monkey-patch on `UnquantizedFusedMoEMethod.forward_cuda`.
The patched method uses ExQ's INT4 Triton kernel (packed uint8 weights,
on-chip dequantization) instead of SGLang's fp16 fused_experts kernel.

| Model | SGLang fp16 P50 | ExQ INT4 P50 | Speedup | Δ |
|---|---|---|---|---|
| OLMoE-1B-7B (64 exp, top-2) | 2.393ms | 1.919ms | **1.25×** | +19.8% |
| Qwen3-30B-A3B (128 exp, top-8) | 3.589ms | 3.101ms | **1.16×** | +13.6% |

Tested at batch=8, seqlen=64 on A10G. Single 200-run measurement.
Direct Triton kernel benchmark shows larger gains (+34% OLMoE, +46% Qwen3)
because it excludes SGLang's sort/dispatch overhead, which is shared cost.

```python
# Apply ExQ to any SGLang model in 2 lines:
from exq.runtime.sglang_backend import patch_sglang
patch_sglang("artifacts/qwen3-30b-a3b.json")
# SGLang now uses ExQ's INT4 kernel for all covered MoE layers
```

```bash
# Or use the one-command serve script:
bash scripts/serve_exq.sh Qwen/Qwen3-30B-A3B
```

### Direct Triton kernel (+34% / +46%)

Same kernel, measured with a warm Triton compilation cache (200 samples):

| Model | BF16 P50 | INT4 P50 | Speedup |
|---|---|---|---|
| OLMoE-1B-7B | 1.173ms | 0.774ms | **+34%** |
| Qwen3-30B-A3B | 2.044ms | 1.112ms | **+46%** |

Weight bandwidth reduction: 3.88× (fp16 → packed INT4 + per-group scales).
Kernel correctness: max_diff=0.0 vs Python dequant reference on both models.

### Quality recovery (53-68%)

On 6 models, ExQ's frequency-stratified quantization recovers 53-68% of the quality lost by uniform INT4 quantization, at the same memory budget:

| Model | Type | Recovery | quant_diff |
|-------|------|----------|------------|
| OLMoE-1B-7B | MoE | 53.9% | 0.429 |
| Qwen2.5-3B | Dense | 66.8% | 0.377 |
| Qwen2.5-1.5B | Dense | 62.2% | 0.345 |

Recovery correlates with `quant_diff` (fraction of experts assigned to INT8 or BF16): higher `quant_diff` means more of the activation mass lands on higher-precision experts, which tends to preserve output quality.

## License

Apache-2.0
