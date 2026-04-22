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

### Contribution 1 — Kernel speed (ExQ INT4 vs fp16, same model)

**Fair baseline:** our fp16 grouped GEMM kernel, same sorted-token dispatch, same hardware.
The only difference is weight precision: fp16 (2 bytes/param) vs packed INT4 (0.5 bytes/param + scales).

| Model | fp16 P50 | ExQ INT4 P50 | Δ |
|---|---|---|---|
| OLMoE-1B-7B (64 exp, top-2) | 1.173 ms | 0.774 ms | **−34%** |
| Qwen3-30B-A3B (128 exp, top-8) | 2.044 ms | 1.112 ms | **−46%** |

A10G, batch=8, seqlen=64, 200-run P50. Gains are consistent across batch sizes
(OLMoE: −34–36%; Qwen3: −45–51%). Source of speedup: weight HBM traffic cut
by **3.88×** (OLMoE: 256 MB → 66 MB; Qwen3: 384 MB → 99 MB).

This benchmark does not involve SGLang. It compares the two kernels directly.

### Contribution 2 — Quality (ExQ mixed-prec vs uniform INT4, same memory)

**Fair baseline:** uniform INT4 at the same total memory footprint.
ExQ assigns BF16/INT8 to hot experts and INT4 to cold experts; the memory budget is identical.

Recovery measures what fraction of the fp16→INT4 quality gap ExQ eliminates:

| Model | Type | Recovery | quant_diff | Notes |
|---|---|---|---|---|
| OLMoE-1B-7B | MoE | **53.9%** | 0.429 | Strong routing concentration |
| Qwen2.5-3B | Dense | **66.8%** | 0.377 | |
| Qwen2.5-1.5B | Dense | **62.2%** | 0.345 | |
| Qwen1.5-MoE | MoE | 1.4% | 0.010 | Near-uniform routing; no headroom |
| DeepSeek-V2-Lite | MoE | −0.3% | 0.003 | Near-uniform routing |
| GLM-4.7-Flash | MoE | −14.2% | 0.119 | INT4 already close to fp16 baseline |

`quant_diff` = fraction of experts at INT8 or BF16. Recovery only materialises when routing is concentrated enough to assign meaningful higher-precision headroom to hot experts. Models with near-uniform routing (low `quant_diff`) see little or no gain.

Output distribution (KL divergence vs fp16, Qwen2.5-3B, WikiText2):

| Method | Mean KL | P99 KL |
|---|---|---|
| Uniform INT4 | 0.04138 | 0.35526 |
| ExQ mixed-prec | **0.01780** | **0.17469** |
| AWQ (controlled) | 0.12212 | 1.25371 |

ExQ mean KL is **2.3× lower than uniform INT4** and **6.9× lower than AWQ**
at the same memory budget.

### System-level — ExQ via SGLang vs SGLang fp16

This is not a fair speed comparison (INT4 weights vs fp16 weights). It answers
a different question: *what does the full system look like if you deploy ExQ
instead of the default fp16 SGLang setup?* You get lower latency at the cost of
INT4 weight precision; ExQ's mixed-precision assignment partially recovers that
quality cost (see Contribution 2 above).

| Model | SGLang fp16 | ExQ INT4 (via SGLang) | Δ |
|---|---|---|---|
| OLMoE-1B-7B | 2.393 ms | 1.919 ms | −19.8% |
| Qwen3-30B-A3B | 3.589 ms | 3.101 ms | −13.6% |

A10G, batch=8, seqlen=64. The gain is smaller than the direct kernel benchmark
because SGLang's shared dispatch overhead (token sorting, boundary computation)
is the same for both systems and is not affected by weight precision.

Batch sweep (SGLang integration, best operating points in **bold**):

| Batch | OLMoE Δ | Qwen3 Δ |
|---|---|---|
| 1 | −5.9% | −19.3% |
| 2 | **−26.7%** | −17.0% |
| 4 | **−25.3%** | **−26.5%** |
| 8 | −20.5% | −13.2% |

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
