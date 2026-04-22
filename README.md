# ExQ — Expert Quantization for MoE Inference

ExQ is a compiler framework that profiles MoE expert routing behavior offline and uses those statistics to make every quantization, prefetching, and memory layout decision at compile time. The inference runtime executes a fixed schedule — no per-token predictor, no runtime overhead from prediction.

The INT4 Triton kernel eliminates per-token prediction overhead. Static prefetch schedules (Pass C) currently use Python forward hooks (~2–3 ms/token dispatch cost); a native CUDA implementation would remove this.

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

1. **Profile** the model on a calibration corpus. Records which experts activate, how often, and which co-activate across layers.
2. **Compile** the profile into a routing graph IR. Four passes run over this graph: memory layout (Pass A), quantization assignment (Pass B), prefetch scheduling (Pass C), and layer specialization (Pass D).
3. **Emit** a compiled artifact (JSON) with per-expert precision assignments, a static prefetch schedule, and layout metadata.
4. **Serve** with the artifact. ExQ patches SGLang's MoE dispatch to use the INT4 Triton kernel. No predictor runs at inference time.

## Usage

### One command

```bash
exq serve --model Qwen/Qwen3-30B-A3B
```

Profiles the model if no profile exists, compiles an artifact, applies the ExQ INT4 patch to SGLang, and starts an OpenAI-compatible server on port 30000.

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
# Perplexity: fp16 vs ExQ vs uniform INT4
python scripts/eval_ppl.py --model allenai/OLMoE-1B-7B-0924 \
    --precision rpgo --quant-plan artifacts/olmoe.json --dataset wikitext2

# Compile-time diagnostic: will ExQ help on this model?
python scripts/exq_diagnose.py --profile profiles/olmoe.json

# INT4 Triton kernel benchmark vs BF16 baseline
python scripts/bench_int4.py --model olmoe

# SGLang integration benchmark
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

### Decode latency — SGLang integration

ExQ patches `UnquantizedFusedMoEMethod.forward_cuda` in SGLang. The patched method dispatches through ExQ's INT4 Triton kernel (packed uint8 weights, on-chip dequantization) instead of SGLang's fp16 fused_experts.

| Model | SGLang fp16 | ExQ INT4 | Δ |
|---|---|---|---|
| OLMoE-1B-7B (64 exp, top-2) | 2.393 ms | 1.919 ms | **−19.8%** |
| Qwen3-30B-A3B (128 exp, top-8) | 3.589 ms | 3.101 ms | **−13.6%** |

batch=8, seqlen=64, A10G. 200-run P50. Batch sweep (SGLang integration):

| Batch | OLMoE Δ | Qwen3 Δ |
|---|---|---|
| 1 | −5.9% | −19.3% |
| 2 | **−26.7%** | −17.0% |
| 4 | **−25.3%** | **−26.5%** |
| 8 | −20.5% | −13.2% |

### Decode latency — direct INT4 Triton kernel

Measured against the fp16 grouped GEMM baseline, warm Triton cache, 200 runs:

| Model | fp16 | INT4 | Δ |
|---|---|---|---|
| OLMoE-1B-7B | 1.173 ms | 0.774 ms | **−34%** |
| Qwen3-30B-A3B | 2.044 ms | 1.112 ms | **−46%** |

Gains are consistent across batch sizes (OLMoE: −34–36%; Qwen3: −45–51%).
Weight bandwidth reduction: **3.88×** (fp16 256 MB → INT4+scales 66 MB for OLMoE; 384 MB → 99 MB for Qwen3).
Kernel correctness: max_diff = 0.0 vs Python dequant reference.

### Output quality — KL divergence

Qwen2.5-3B on WikiText2, measured against fp16 output distribution:

| Method | Mean KL | P99 KL |
|---|---|---|
| Uniform INT4 | 0.04138 | 0.35526 |
| ExQ (ours) | **0.01780** | **0.17469** |
| AWQ controlled | 0.12212 | 1.25371 |

ExQ mean KL is **6.9× lower than AWQ** and **2.3× lower than uniform INT4**.

### Output quality — PPL recovery

ExQ's frequency-stratified quantization assigns higher precision to frequently-activated experts, recovering a fraction of the quality lost by uniform INT4 at the same memory budget:

| Model | Type | Recovery | quant_diff |
|---|---|---|---|
| OLMoE-1B-7B | MoE | **53.9%** | 0.429 |
| Qwen2.5-3B | Dense | **66.8%** | 0.377 |
| Qwen2.5-1.5B | Dense | **62.2%** | 0.345 |
| Qwen1.5-MoE | MoE | 1.4% | 0.010 |
| DeepSeek-V2-Lite | MoE | −0.3% | 0.003 |
| GLM-4.7-Flash | MoE | −14.2% | 0.119 |

`quant_diff` = fraction of experts assigned to INT8 or BF16. Recovery tracks `quant_diff`: models where routing is concentrated enough for ExQ to assign meaningful higher-precision headroom see large gains; models with near-uniform routing (low `quant_diff`) see little or no recovery. GLM-4.7-Flash is an outlier — routing is moderately concentrated but INT4 was already close to fp16 quality, leaving little room to recover.

### Compile time

All models compile under 3 seconds on a single CPU core:

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
