# AGENTS.md — Routing-Profile-Guided Optimization (R-PGO)
## A Compiler Framework for Mixture-of-Experts Inference

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation & Background](#2-motivation--background)
3. [Prior Art & Differentiation](#3-prior-art--differentiation)
4. [Core Insight: PGO Applied to MoE Routing](#4-core-insight-pgo-applied-to-moe-routing)
5. [Architecture](#5-architecture)
6. [Implementation Plan](#6-implementation-plan)
7. [Hardware Requirements](#7-hardware-requirements)
8. [Evaluation Plan](#8-evaluation-plan)
9. [Research Positioning](#9-research-positioning)
10. [References](#10-references)

---

## 1. Project Overview

**R-PGO** is a compiler framework that uses MoE routing behavior — collected offline via lightweight profiling — as the primary signal driving every optimization decision in the compilation pipeline. It is the first system to treat expert routing statistics as *profile data* in the classical compiler sense, analogous to how GCC and Clang use branch-frequency profiles in Profile-Guided Optimization (PGO) to drive inlining, block reordering, and register allocation decisions.

### What It Is Not

R-PGO is **not** a runtime prediction system. Every existing system that addresses MoE expert loading (SP-MoE, ExpertFlow, MoE-SpeQ, Pre-Gated MoE, ProMoE, HOBBIT) operates as a *runtime heuristic* — a predictor that runs alongside the model and makes prefetching decisions token-by-token at inference time. R-PGO makes all of these decisions **at compile time**, emitting a static artifact where layout, quantization, prefetch schedules, and kernel specialization are fully baked in. The inference runtime becomes a simple fetch-and-execute machine.

### The One-Line Pitch

> *"PGO for MoE models: routing statistics feed a compiler that makes every optimization decision jointly at compile time, so the inference artifact runs with zero prediction overhead. All existing work does this at runtime. We do it in the compiler."*

---

## 2. Motivation & Background

### 2.1 The CPU Architecture Analogy

The most productive source of inference optimization ideas has historically been classical CPU architecture. Speculative decoding [1] is a direct translation of branch prediction [2]. Pipeline parallelism [3] is a direct translation of instruction pipelining. PagedAttention [4] is a direct translation of virtual memory and OS paging. Each of these ideas produced systems-level breakthroughs by asking: *what does CPU concept X look like when applied to language model inference?*

R-PGO follows this lineage. The concept being translated is **Profile-Guided Optimization** — the compiler technique in which a program is compiled once with instrumentation, run on representative inputs to collect a runtime profile, and then recompiled using that profile to make better static optimization decisions [5]. GCC and Clang PGO has produced 10–15% speedups on real workloads by better informing inlining decisions, basic block reordering, and loop transformations [6].

The translation: **replace branch-frequency profiles with expert-activation-frequency profiles**. In a transformer, the router is a deterministic function of the residual stream. Its behavior over a representative calibration corpus is measurable, stable, and predictive — exactly the properties that make PGO profiles useful in classical compilation.

### 2.2 The MoE Scaling Problem

Mixture-of-Experts has become the dominant architecture for frontier-scale language models. Qwen3-30B-A3B activates only 3.3B of its 30.5B parameters per forward pass — 8 of 128 experts per layer across 48 layers [7]. The flagship Qwen3-235B-A22B activates 22B of 235B total parameters per token. DeepSeek-V3 activates 37 of 671 billion parameters per token — 8 of 256 routed experts plus 1 shared expert [8]. Llama 4 follows the same architectural pattern. The key property enabling this scale: sparse activation means total parameter count can grow without a proportional increase in per-token compute.

The inference bottleneck this creates is well-documented. For any model where expert parameters exceed GPU VRAM capacity, inactive experts must be stored in CPU RAM or NVMe and loaded over PCIe on demand. Since expert selection is data-dependent — the router decides which experts to activate based on the current token — expert weight loads fall directly on the critical path of inference. Each generated token can require loading new expert weights over PCIe, stalling the GPU [9].

The PCIe bandwidth constraint is severe: a modern PCIe Gen4 x16 slot provides ~32 GB/s. Qwen3-235B-A22B in INT4 (~117GB total) requires continuous expert streaming during inference. Each expert block across 94 layers must be fetched from CPU RAM on demand. Real systems overlap compute and I/O, but the scheduling of these transfers is the dominant performance variable.

### 2.3 Why Existing Solutions Are Insufficient

All current approaches to this problem operate at the **runtime** level. They train or heuristically derive a predictor, run it at inference time, and use its output to prefetch experts one step ahead. The fundamental limitations of this approach are:

1. **Prediction overhead is on the critical path.** Running a predictor (even a small MLP) adds latency to every decode step. SP-MoE reports 70–90% prediction accuracy at the cost of per-step predictor inference [10].

2. **Optimization decisions are made in isolation.** A runtime prefetcher only optimizes prefetching. It cannot simultaneously optimize memory layout, quantization assignments, and kernel specialization because those decisions were already made at compile time without routing knowledge.

3. **No static guarantees.** A runtime predictor can fail silently. There is no mechanism to verify at compile time that expert coverage is adequate for the deployment distribution.

4. **Hardware portability requires re-tuning.** A runtime predictor tuned for A100 memory bandwidth does not automatically retune when deployed on H100 or MI300X.

R-PGO addresses all four limitations by moving the optimization problem into the compiler.

---

## 3. Prior Art & Differentiation

### 3.1 Runtime Expert Prefetching Systems

The following systems all address expert loading latency at the *runtime* level and represent the state of the art that R-PGO is differentiated from:

| System | Mechanism | Key Result | Limitation |
|--------|-----------|-----------|------------|
| Pre-Gated MoE [11] | Modifies router to select next-layer experts; combines with caching | Hides expert loading latency | Requires model architecture modification |
| ProMoE [12] | Trains small MLP to predict expert activation layer-by-layer | High prediction accuracy | Sequential layer-by-layer design limits scheduling flexibility |
| SP-MoE [10] | Exploits draft/target structural correspondence for speculative prefetching | 70–90% accuracy, 44.3% hit rate on drafted tokens | Only addresses speculative decoding workloads |
| ExpertFlow [13] | All-layers routing path predictor (RPP) + token scheduler + expert cache engine | Reduces stall time to <0.1% | Runtime overhead per batch; no joint optimization with other passes |
| MoE-SpeQ [9] | Draft model predicts expert sequences; runtime orchestrator prefetches | 2.34× speedup over baseline offloading | Runtime prediction overhead; single-objective optimization |
| FATE [10] | Combines prediction, caching, and mixed-precision experts | 97.2% coverage above confidence threshold | Runtime system; no compiler integration |
| MoE-Infinity [14] | Activation-aware expert offloading with look-ahead routing | Efficient expert caching | Heuristic-based; no formal optimization framework |
| HOBBIT [15] | Multi-dimensional cache manager + dynamic expert loader | Reduces TPOT | Runtime; no joint optimization |

**The critical observation:** every system in this table is a runtime system. None produces a compiled static artifact. None uses routing statistics to drive memory layout, quantization, or kernel specialization decisions jointly.

### 3.2 Compiler-Level Prior Art

The compiler techniques R-PGO builds on are individually well-established:

**Profile-Guided Optimization (PGO):** PGO is a mature technique in GCC and LLVM where instrumented binary runs produce profiles that inform the compiler's next compilation [6]. The classical applications are inlining decisions, basic block reordering for instruction cache locality, and loop transformation profitability. R-PGO applies the same compile-profile-recompile workflow but the profile contains expert activation frequencies instead of branch frequencies.

**Polyhedral Scheduling:** The polyhedral model represents loop nests as integer polyhedra and applies affine transformations for parallelism, tiling, and cache optimization [16]. MLIR's Affine dialect provides a production implementation [17]. R-PGO's memory layout pass uses polyhedral dependence analysis to reason about expert weight reuse distance.

**Abstract Interpretation for Quantization:** Abstract interpretation has been applied to floating-point programs to bound numerical errors [18]. Recent work has extended this to certified quantization of neural networks [19, 20]. R-PGO's quantization pass uses activation frequency as the primary signal rather than pure sensitivity analysis, but the error-bounding framework from certified quantization informs the precision assignment.

**Equality Saturation:** The `egg` and `egglog` frameworks use e-graphs to represent all equivalent programs simultaneously and extract the cost-optimal one [21]. This is the theoretically cleanest framework for the joint optimization problem R-PGO addresses — every optimization decision (layout, quantization, kernel specialization) can be expressed as a rewrite rule, and the extractor finds the globally optimal combination. R-PGO's v1 uses a greedy per-pass approach; a future version using equality saturation over the routing graph would be strictly more powerful.

### 3.3 The Gap

No existing system applies PGO-style compile-time profiling to the MoE routing dimension in a way that drives the **full compilation pipeline**. The gap is not incremental — it is architectural. Moving routing optimization from the runtime into the compiler changes the entire design space: what can be optimized, how optimizations compose, and what guarantees can be made about the output.

---

## 4. Core Insight: PGO Applied to MoE Routing

### 4.1 The Analogy Formalized

In classical PGO [5, 6]:

```
Phase 1:  Compile with instrumentation → instrumented binary
Phase 2:  Run on representative inputs → profile.profdata
Phase 3:  Recompile using profile → optimized binary
```

The profile contains: function call frequencies, branch taken/not-taken counts, value profiles for indirect calls.

In R-PGO:

```
Phase 1:  Run model with routing instrumentation → routing_profile.json
Phase 2:  Compiler ingests routing profile → builds Routing Graph IR
Phase 3:  Compiler emits optimized model artifact with:
          - Co-activation-aware memory layout
          - Frequency-stratified quantization plan
          - Statically embedded prefetch schedules
          - Low-entropy layer specializations
```

The routing profile contains: expert activation frequencies per layer, pairwise co-activation probabilities across layers, per-layer routing entropy, per-expert average arithmetic intensity.

### 4.2 Why Routing Profiles Are Better Than Branch Profiles

Classical PGO profiles are inherently noisy because branch behavior depends on runtime data that cannot be predicted from source code. Expert routing behavior has a structural property that makes its profile *more* useful:

**Routing stability.** Empirical analysis of MoE models including Mixtral-8x7B and Qwen-series models shows that routing decisions are highly consistent across tokens of the same semantic type — the same small subset of experts handles mathematical reasoning, another handles code generation, another handles multilingual content [22]. For Qwen3's 128-expert layers, this specialization is even more pronounced: with 8-of-128 routing, individual experts develop highly specialized functions and their activation is strongly conditioned on input semantics. This is not true of arbitrary program branches. The implication: a routing profile collected on a representative calibration dataset generalizes well to the deployment distribution. This is exactly the condition under which PGO produces reliable improvements.

**Cross-layer correlations are strong.** The activation of expert *i* at layer *L* is strongly predictive of which experts activate at layer *L+1* [13]. This makes the routing graph edges high-information, high-reliability data. Classical PGO branch profiles have no equivalent cross-basic-block correlation structure.

---

## 5. Architecture

### 5.1 System Overview

```
                    ┌─────────────────────────────────────────┐
                    │           R-PGO Compiler Pipeline         │
                    └─────────────────────────────────────────┘

 HuggingFace Model                    Calibration Corpus
 (safetensors / GGUF)                 (1k–10k representative samples)
        │                                       │
        ▼                                       ▼
 ┌──────────────────────────────────────────────────────┐
 │  Phase 1: Routing Profiler (Instrumentation Pass)    │
 │                                                      │
 │  - Forward pass with activation counters             │
 │  - Records per-layer expert selection frequencies    │
 │  - Computes pairwise cross-layer co-activation       │
 │  - Measures per-layer routing entropy                │
 │  Output: routing_profile.json                        │
 └────────────────────────┬─────────────────────────────┘
                          │
                          ▼
 ┌──────────────────────────────────────────────────────┐
 │  Phase 2: Routing Graph IR Construction              │
 │                                                      │
 │  Nodes: (layer, expert, freq, size, intensity)       │
 │  Edges: conditional co-activation probabilities      │
 │  Node weights: activation frequency, entropy         │
 │  Output: RoutingGraph data structure                 │
 └────────────────────────┬─────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼           ▼
         Pass A       Pass B       Pass C       Pass D
        Layout       Quant      Prefetch    Specialization
        Planner      Planner     Emitter      Pass
              │           │           │           │
              └───────────┴───────────┴───────────┘
                          │
                          ▼
 ┌──────────────────────────────────────────────────────┐
 │  Phase 4: Compiled Model Artifact                    │
 │                                                      │
 │  weight_store:      experts laid out by co-activation│
 │  kernel_store:      per-layer kernels + prefetches   │
 │  fast_path_store:   kernels for low-entropy layers   │
 │  quant_map:         per-expert precision assignments │
 │  coverage_report:   static correctness guarantees   │
 └──────────────────────────────────────────────────────┘
```

### 5.2 Phase 1: The Routing Profiler

The routing profiler is a lightweight instrumentation pass added to the model's forward function. It requires a single pass over a calibration corpus — no gradient computation, no training, no model modification.

```python
class RoutingProfiler:
    """
    Instruments a MoE model's routing decisions during a calibration run.
    Analogous to LLVM's instrumentation pass for PGO profile collection.
    
    Collects:
      - expert_activation_counts[layer][expert]     : raw count
      - expert_activation_freq[layer][expert]        : normalized frequency
      - co_activation_matrix[layer][e_i][layer+1][e_j]: cross-layer P(e_j | e_i)
      - routing_entropy[layer]                       : H(router distribution)
      - per_expert_input_stats[layer][expert]        : mean/var of inputs to each expert
    """
    
    def __init__(self, model, n_experts_per_layer: list[int]):
        self.model = model
        self.counts = {l: Counter() for l in range(len(n_experts_per_layer))}
        self.co_counts = {}  # (layer_i, expert_i) -> Counter of (layer_j, expert_j)
        self._register_hooks()

    def _register_hooks(self):
        # Register forward hooks on each MoE router module.
        # Qwen3-30B-A3B uses Qwen3MoeSparseMoeBlock with a gate linear layer.
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                layer.mlp.gate.register_forward_hook(
                    self._make_hook(layer_idx)
                )
    
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # output: router logits [batch, seq, n_experts]
            topk_experts = output.topk(k=self.top_k, dim=-1).indices
            for expert_idx in topk_experts.view(-1).tolist():
                self.counts[layer_idx][expert_idx] += 1
        return hook
    
    def run_calibration(self, dataloader, n_samples: int = 2048):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx * batch['input_ids'].shape[0] >= n_samples:
                    break
                self.model(**batch)
        return self._build_profile()
    
    def _build_profile(self) -> RoutingProfile:
        # Normalize counts, compute entropy, build co-activation edges
        ...
```

**Output schema:**

```json
{
  "model_id": "Qwen/Qwen3-30B-A3B",
  "calibration_samples": 2048,
  "layers": {
    "12": {
      "expert_activation_freq": [0.18, 0.15, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, ...],
      "routing_entropy": 2.71,
      "co_activation_next_layer": {
        "expert_3": {"expert_7": 0.72, "expert_21": 0.18, "expert_56": 0.06, "expert_99": 0.04},
        "expert_7": {"expert_3": 0.58, "expert_21": 0.29, "expert_99": 0.13}
      },
      "per_expert_avg_l2_norm": [2.41, 2.35, 2.19, 2.08, 1.97, ...]
    }
  }
}
```

### 5.3 Phase 2: The Routing Graph IR

The Routing Graph is the central intermediate representation that all downstream compiler passes consume. It is a directed acyclic graph where:

- **Nodes** represent `(layer_index, expert_index)` pairs, annotated with activation frequency, weight tensor size, and average arithmetic intensity.
- **Edges** represent cross-layer conditional co-activation probabilities: `P(expert_j at layer L+1 | expert_i at layer L)`.
- **Node weights** encode the optimization priority: hot experts (high frequency) are treated like hot code paths in PGO — they receive higher-quality optimization.

```python
@dataclass
class RoutingGraphNode:
    layer: int
    expert: int
    activation_freq: float      # P(this expert activates) over calibration corpus
    weight_size_bytes: int      # size of this expert's parameters
    avg_arithmetic_intensity: float  # FLOP/byte ratio when this expert runs
    routing_entropy: float      # entropy of router at this layer

@dataclass  
class RoutingGraphEdge:
    src: RoutingGraphNode       # (layer L, expert i)
    dst: RoutingGraphNode       # (layer L+1, expert j)
    conditional_prob: float     # P(dst activates | src activated)

class RoutingGraph:
    nodes: dict[tuple[int,int], RoutingGraphNode]
    edges: list[RoutingGraphEdge]
    
    def hot_experts(self, threshold=0.10) -> list[RoutingGraphNode]:
        # For Qwen3-30B-A3B: with 128 experts, top-8 routing, expect ~10-15
        # experts per layer above this threshold (the consistently-used core)
        return [n for n in self.nodes.values() if n.activation_freq >= threshold]
    
    def cold_experts(self, threshold=0.01) -> list[RoutingGraphNode]:
        # For Qwen3-30B-A3B: with 128 experts and top-8 routing, expect ~50-60
        # experts below this threshold — aggressive INT4 candidates
        return [n for n in self.nodes.values() if n.activation_freq < threshold]
    
    def high_prob_prefetch_edges(self, min_prob=0.60) -> list[RoutingGraphEdge]:
        return [e for e in self.edges if e.conditional_prob >= min_prob]
    
    def low_entropy_layers(self, max_entropy=0.8) -> list[int]:
        layers = {n.layer for n in self.nodes.values()}
        return [l for l in layers 
                if self._layer_entropy(l) <= max_entropy]
```

### 5.4 Compiler Pass A: Expert Memory Layout Planner

**Objective:** Place expert weight tensors in memory such that experts with high co-activation probability are physically co-located, maximizing prefetch and cache efficiency.

**Theory:** This is an instance of the **graph bandwidth minimization problem** [23] — assign each node (expert) a memory address such that the weighted sum of address distances between co-activated pairs is minimized. High co-activation probability → small address distance → same HBM memory page → single DMA transfer fetches both.

```
Input:  RoutingGraph, HBM page size (default: 2MB)
Output: expert_placement_plan: dict[(layer, expert) -> memory_offset]

Algorithm:
  1. Build weighted adjacency matrix W where W[i][j] = co_activation_prob(i, j)
  2. Apply spectral graph partitioning to cluster experts by co-activation
  3. Within each cluster, sort by activation_freq descending (hottest first)
  4. Assign contiguous memory blocks: cluster_0_expert_0, cluster_0_expert_1, ...
  5. Verify: for each high-prob edge (P > 0.6), check that src and dst 
     experts are within same or adjacent HBM pages
```

**Analogy to PGO:** Classical PGO uses branch frequency profiles to reorder basic blocks so that the hot path falls through without jumps, improving instruction cache hit rate [6]. Pass A does the same for expert weights: the hot co-activation path falls through in memory without large address jumps.

### 5.5 Compiler Pass B: Frequency-Stratified Quantization Planner

**Objective:** Assign bit-widths to expert weight tensors based on activation frequency, using frequency as a proxy for the optimization objective (minimize degradation on the hot path; aggressively compress the cold path).

**Theory:** This extends the certified quantization framework [19, 20] with frequency weighting. Standard mixed-precision quantization treats all experts equally when assigning precision. Frequency-stratified quantization recognizes that a cold expert (0.5% activation rate) contributes negligibly to the output distribution — it can be compressed more aggressively without measurable quality loss.

```
Frequency Tiers:

  HOT   (freq ≥ 0.10):  bf16
    → These experts are on the critical path. Their precision loss
      directly affects the majority of token generations.
      Treat like "hot code" in PGO — optimize for quality.

  WARM  (0.03 ≤ freq < 0.10):  int8
    → Moderate contribution. Standard quantization acceptable.
      Verify with abstract interpretation that ε_output ≤ budget.

  COLD  (freq < 0.03):  int4 or fp8
    → Rarely activated. Aggressive compression has negligible
      effect on average output quality.
      Treat like "cold code" in PGO — optimize for size.

  FROZEN (freq < 0.005): optional skip / load-on-demand only
    → Almost never activated. May be omitted from GPU memory
      budget entirely; loaded only on explicit miss.
```

**Error budget allocation:** The compiler tracks cumulative error across all layers, ensuring that the sum of per-expert quantization errors (weighted by activation frequency) stays within a user-specified KL divergence budget from the fp16 baseline.

```python
def assign_precisions(
    routing_graph: RoutingGraph,
    error_budget: float = 0.01,    # max KL divergence from fp16 baseline
    memory_budget_gb: float = 20.0
) -> dict[tuple[int,int], Precision]:
    """
    Solves a constrained optimization:
      minimize: sum over experts of memory_cost(expert, precision)
      subject to: sum over experts of freq(expert) * error(expert, precision) 
                  <= error_budget
                  total_memory(plan) <= memory_budget_gb
    """
    ...
```

### 5.6 Compiler Pass C: Static Prefetch Schedule Emitter

**Objective:** Embed explicit async prefetch instructions into each layer's compiled kernel, using routing graph edge probabilities to determine which expert weights to prefetch and at what priority.

**Theory:** This is the core novel contribution of R-PGO. Unlike all runtime prefetching systems [9, 10, 11, 12, 13, 15], the prefetch schedule is computed statically from the routing graph and embedded in the compiled kernel. At inference time, there is no predictor running — the GPU executes the statically emitted prefetch instructions unconditionally.

```
For each layer L:
  For each expert E that is hot (freq ≥ threshold):
    For each edge (L:E → L+1:E') in the routing graph:
      if edge.conditional_prob ≥ HIGH_THRESHOLD (0.70):
        emit: async_prefetch(weight_ptr[L+1][E'], priority=HIGH)
      elif edge.conditional_prob ≥ MED_THRESHOLD (0.35):
        emit: async_prefetch(weight_ptr[L+1][E'], priority=MEDIUM)
      else:
        do not emit prefetch (avoid polluting transfer queue)
```

**Generated kernel pseudocode:**

```c
// Compiler-generated kernel for layer 12, expert 3 (freq=0.41)
__device__ void layer12_expert3_kernel(float* input, float* output, 
                                        PrefetchQueue* pq) {
    // 1. Compute expert output (the actual work)
    matmul(input, expert_weights[12][3], output);
    
    // 2. Statically-emitted prefetches based on routing graph edges
    //    P(L13:E1 | L12:E3) = 0.78  -> HIGH priority prefetch
    //    P(L13:E5 | L12:E3) = 0.19  -> MED priority prefetch  
    //    P(L13:E7 | L12:E3) = 0.03  -> no prefetch emitted
    async_dma_prefetch(&expert_weights[13][1], sizeof(ExpertBlock), 
                       pq, PRIORITY_HIGH);
    async_dma_prefetch(&expert_weights[13][5], sizeof(ExpertBlock), 
                       pq, PRIORITY_MEDIUM);
    // expert 7 not prefetched — compiler decided P too low
}
```

**Why this is strictly better than runtime prediction:**

1. Zero per-token overhead — the prefetch instructions are unconditional loads embedded in the kernel.
2. The compiler can verify statically that prefetch coverage is adequate (coverage = sum of freq-weighted high-prob edges / total activation events).
3. The prefetch schedule interacts correctly with Pass A (memory layout) — co-located experts can be fetched in a single DMA transfer.

### 5.7 Compiler Pass D: Low-Entropy Layer Specialization

**Objective:** For layers where routing entropy is very low (the router almost always selects the same expert combination), emit a *specialized fast-path kernel* that skips the routing computation and directly executes the predicted expert combination.

**Theory:** This is **partial evaluation** [24] driven by routing statistics. Partial evaluation specializes a program for statically-known inputs by precomputing the parts that only depend on those inputs. When routing entropy is low, the router's output is effectively statically known — the specialized kernel treats the dominant expert combination as a constant.

```
Layer entropy classification:

  HIGH ENTROPY (H > 1.5 bits):
    → Router output is unpredictable. No specialization possible.
       Emit only the general kernel.

  MEDIUM ENTROPY (0.5 < H ≤ 1.5 bits):
    → A few dominant combinations. Emit 2–3 specialized kernels
       plus general fallback. Dispatch based on router output.

  LOW ENTROPY (H ≤ 0.5 bits):
    → One expert combination dominates (>70% of tokens).
       Emit single specialized kernel with routing check + fallback.
       Expected savings: skip router softmax + top-k selection.
```

---

## 6. Implementation Plan

### 6.1 Repository Structure

```
rpgo/
├── AGENTS.md                   ← this document
├── README.md
├── pyproject.toml
│
├── rpgo/
│   ├── __init__.py
│   ├── profiler/
│   │   ├── __init__.py
│   │   ├── routing_profiler.py     # Phase 1: instrumentation hooks
│   │   ├── calibration_runner.py   # runs profiling over dataset
│   │   └── profile_schema.py       # RoutingProfile dataclass + JSON I/O
│   │
│   ├── ir/
│   │   ├── __init__.py
│   │   ├── routing_graph.py        # RoutingGraph, Node, Edge dataclasses
│   │   ├── graph_builder.py        # constructs graph from routing profile
│   │   └── graph_analysis.py       # entropy, coverage, co-activation queries
│   │
│   ├── passes/
│   │   ├── __init__.py
│   │   ├── base_pass.py            # abstract CompilerPass interface
│   │   ├── layout_planner.py       # Pass A: memory layout optimization
│   │   ├── quant_planner.py        # Pass B: frequency-stratified quantization
│   │   ├── prefetch_emitter.py     # Pass C: static prefetch schedule
│   │   └── specialization.py       # Pass D: low-entropy kernel specialization
│   │
│   ├── codegen/
│   │   ├── __init__.py
│   │   ├── artifact_builder.py     # assembles compiled model artifact
│   │   ├── kernel_generator.py     # emits CUDA/Triton kernels with prefetches
│   │   └── runtime_shim.py         # minimal runtime for executing artifact
│   │
│   └── eval/
│       ├── __init__.py
│       ├── benchmark.py            # tokens/sec, TTFT, TPOT measurement
│       ├── quality.py              # perplexity, KL div vs fp16 baseline
│       └── coverage.py             # static prefetch coverage analysis
│
├── scripts/
│   ├── profile_model.py            # CLI: collect routing profile
│   ├── compile_model.py            # CLI: run full compilation pipeline  
│   ├── benchmark.py                # CLI: evaluate compiled vs. baseline
│   └── inspect_routing_graph.py    # CLI: visualize routing graph statistics
│
├── configs/
│   ├── deepseek-v2-lite.yaml       # per-model compilation config
│   └── mixtral-8x7b.yaml
│
└── tests/
    ├── test_profiler.py
    ├── test_routing_graph.py
    ├── test_passes.py
    └── test_artifact.py
```

### 6.2 Development Milestones

#### Milestone 1 — Routing Profiler (Weeks 1–2)

**Goal:** Collect accurate routing profiles from HuggingFace MoE models.

**Tasks:**
- Implement `RoutingProfiler` with PyTorch forward hooks on MoE gate modules
- Support DeepSeek-V2-Lite and OLMoE-1B-7B architectures
- Build calibration runner with configurable sample count and dataset
- Implement `RoutingProfile` JSON serialization
- Write unit tests verifying that activation frequencies sum to 1.0 per layer

**Done when:** `python scripts/profile_model.py --model deepseek-ai/DeepSeek-V2-Lite --samples 2048` produces a valid `routing_profile.json` with per-layer expert frequencies and cross-layer co-activation matrices.

**Hardware needed:** Any GPU with ≥12GB VRAM. A10G is more than sufficient.

#### Milestone 2 — Routing Graph IR (Weeks 3–4)

**Goal:** Build the core IR that downstream passes consume.

**Tasks:**
- Implement `RoutingGraph` with node and edge dataclasses
- Implement `GraphBuilder` that ingests `RoutingProfile` and produces `RoutingGraph`
- Implement graph analysis utilities: `hot_experts()`, `cold_experts()`, `high_prob_edges()`, `low_entropy_layers()`
- Implement graph visualization (matplotlib/networkx) for inspection and debugging
- Write unit tests verifying graph properties (edge probabilities sum to 1.0, etc.)

**Done when:** `python scripts/inspect_routing_graph.py --profile routing_profile.json` renders a visual of the expert co-activation graph and prints per-layer entropy statistics.

#### Milestone 3 — Pass B: Quantization Planner (Weeks 5–6)

**Rationale:** Pass B is implemented first because it is the highest-impact, most self-contained pass. It produces a measurable quality/size tradeoff without requiring kernel codegen.

**Tasks:**
- Implement frequency-tier classification (HOT/WARM/COLD/FROZEN)
- Implement error budget allocator using KL divergence tracking
- Integrate with `bitsandbytes` for actual quantization of assigned precision
- Implement perplexity evaluation comparing R-PGO quantized vs. uniform quantized baseline
- Target: achieve better perplexity than uniform INT4 at same or lower memory footprint

**Done when:** `python scripts/compile_model.py --passes quant_only` produces a mixed-precision model that, measured on WikiText-103, has lower perplexity than uniform INT4 quantization at the same total parameter size. Run on Qwen3-30B-A3B.

**Key hypothesis to validate:** Cold experts (freq < 0.03) across Qwen3's 128-expert layers can be quantized to INT4 without measurable perplexity impact because their weighted contribution to the output distribution is negligible. With 128 experts and top-8 routing, the long tail of rarely-activated experts (~50% of the 128) can absorb aggressive compression. Hot experts (freq > 0.10) in bf16 preserve the quality of the majority of token generations.

#### Milestone 4 — Pass C: Prefetch Schedule Emitter (Weeks 7–8)

**Rationale:** This is the core novel contribution of R-PGO. The goal is to produce static prefetch schedules embedded in executable kernels.

**Tasks:**
- Implement prefetch threshold analysis over routing graph edges
- Implement Triton kernel template with async prefetch intrinsics
- Implement `PrefetchCoverageAnalyzer`: compute what % of actual expert activations are covered by static prefetches
- Implement comparison runtime: R-PGO static prefetches vs. ExpertFlow runtime prediction
- Measure: tokens/sec, TTFT, TPOT on Mixtral-8x7B in CPU-offload mode

**Architecture of generated kernel (Triton pseudocode):**

```python
@triton.jit
def moe_layer_kernel_with_prefetch(
    input_ptr, output_ptr,
    expert_weight_ptrs,      # pointer array to all expert weight blocks
    prefetch_indices,        # statically embedded from routing graph
    prefetch_priorities,     # HIGH=0, MED=1, LOW=2
    n_prefetches: tl.constexpr,  # statically known at compile time
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Run the actual expert computation
    pid = tl.program_id(axis=0)
    # ... matmul logic ...
    
    # 2. Issue static prefetches for next-layer experts
    # These indices were determined by the compiler from the routing graph
    # — no runtime prediction needed
    for i in tl.static_range(n_prefetches):
        if prefetch_priorities[i] == 0:  # HIGH priority
            tl.extra.cuda.libdevice.prefetch(
                expert_weight_ptrs[prefetch_indices[i]], 
                cache_level=1
            )
```

**Done when:** On Qwen3-235B-A22B INT4 with experts on CPU RAM (A10G + 64GB RAM setup), R-PGO static prefetches achieve within 10% of ExpertFlow's TPOT with zero runtime prediction overhead. For Mixtral-8x7B (prior art comparison baseline), match or exceed ExpertFlow's published numbers.

#### Milestone 5 — Pass A: Memory Layout Planner (Weeks 9–10)

**Tasks:**
- Implement spectral clustering on co-activation adjacency matrix
- Implement expert memory placement solver (greedy bandwidth minimization)
- Integrate layout plan with artifact builder to emit correctly-ordered weight files
- Measure: reduction in DMA transfer count for high-confidence token generations

**Done when:** Expert memory layout reordering reduces average number of DMA transfers per layer (measured empirically on calibration set) by ≥15% compared to original parameter file ordering.

#### Milestone 6 — Evaluation & Write-Up (Weeks 11–12)

**Tasks:**
- Full pipeline evaluation: Profile → Routing Graph → All Passes → Compiled Artifact
- Baseline comparisons: ExpertFlow, vanilla llama.cpp offloading, MoE-SpeQ (if replicable)
- Ablation study: each pass individually, then combined
- Coverage analysis: what % of expert activations are covered by static prefetch schedules
- Write technical blog post / paper draft

---

## 7. Hardware Requirements

### 7.1 Development Machine

```
GPU:   NVIDIA A10G (24GB GDDR6, 600 GB/s BW, PCIe Gen4)
CPU:   ≥16 cores (compiler passes run on CPU)
RAM:   ≥64 GB  (expert weight storage for offloading benchmark)
Disk:  NVMe SSD (fast weight loading for worst-case baseline measurement)
OS:    Ubuntu 22.04 or 24.04
CUDA:  12.1+
```

The A10G is the right development machine for this project for a non-obvious reason: its 24GB VRAM creates *meaningful memory pressure* for large MoE models, forcing the offloading path to actually exercise. Qwen3-30B-A3B at INT4 (~15GB) fits with headroom, enabling fast iteration on compiler passes. Qwen3-235B-A22B at INT4 (~117GB) does not fit at all, making it the ideal benchmark target — every decode step must stream experts over PCIe, which is exactly the bottleneck R-PGO is designed to eliminate. An A100 (80GB) would partially fit the 235B model, hiding the severity of the problem.

### 7.2 Model Fit Analysis

```
Model                    Total Params  Active/Token  Experts  Layers  INT4 Size   A10G 24GB?
─────────────────────────────────────────────────────────────────────────────────────────────
Qwen3-30B-A3B            30.5 B        3.3 B         128/8    48      ~15 GB      ✅ Primary dev target
Qwen1.5-MoE-A2.7B        14.3 B        2.7 B         60/4     24      ~4 GB       ✅ Smoke-test model
OLMoE-1B-7B              6.9 B         1.3 B         64/8     16      ~3 GB       ✅ Unit test model
Qwen3-235B-A22B          235 B         22 B          128/8    94      ~117 GB     ❌ CPU-offload benchmark
Mixtral-8x7B             46.7 B        12.9 B        8/2      32      ~24 GB      ⚠️  Tight, prior art baseline
DeepSeek-V3              671 B         37 B          256/9    61      ~350 GB     ❌ Multi-node only
```

**Primary development target: Qwen3-30B-A3B.**
- 30.5B total parameters, 3.3B active per token
- 48 transformer layers, **128 experts per MoE layer, top-8 routing** — far richer routing graph than any prior dev target
- At INT4: ~15GB, leaving 9GB headroom on the A10G for instrumentation, activations, and KV cache
- Apache 2.0 license, publicly available: `Qwen/Qwen3-30B-A3B`
- The 128-expert structure means the routing graph has 128 × 48 = 6,144 nodes and a dense co-activation edge set — exactly the complexity level where the compiler's joint optimization pays off most

**Primary benchmark target: Qwen3-235B-A22B in CPU-offload mode.**
- 235B total parameters, 22B active per token, 128 experts, 94 layers
- At INT4 (~117GB), far exceeds A10G VRAM — every decode step requires PCIe expert transfers
- This is the deployment scenario R-PGO is designed for: large model, constrained GPU, expert streaming over PCIe
- Publicly available: `Qwen/Qwen3-235B-A22B`

**Secondary benchmark: Mixtral-8x7B in CPU-offload mode.**
- Kept for direct comparison with all published prior art (ExpertFlow, SP-MoE, MoE-SpeQ all report on Mixtral)
- Results on Mixtral let reviewers directly compare R-PGO against existing systems on a known baseline

### 7.3 PCIe as the Bottleneck to Optimize Against

```
PCIe Gen4 x16 bandwidth:              ~32 GB/s

Qwen3-30B-A3B (dev target, INT4):
  Expert block size:                   ~30 MB per expert
  Active experts per token:            8 per layer × 48 layers = 384 loads
  Naive serial load time:              384 × 30MB / 32GB/s ≈ 360ms per token

Qwen3-235B-A22B (benchmark, INT4):
  Expert block size:                   ~120 MB per expert  
  Active experts per token:            8 per layer × 94 layers = 752 loads
  Naive serial load time:              752 × 120MB / 32GB/s ≈ 2.8s per token

With R-PGO static prefetching (target):
  - Hot co-activated experts prefetched during prior layer's compute window
  - Co-located experts (Pass A) fetched in single DMA burst transfer
  - Low-entropy layers (Pass D) skip router compute on fast path
  - Target: <30ms critical-path stall on 30B-A3B
            <200ms critical-path stall on 235B-A22B
```

**Why Qwen3's 128-expert structure makes the compiler more impactful than Mixtral's 8:**
Mixtral's routing graph has 8 × 32 = 256 nodes. Qwen3-30B-A3B's has 128 × 48 = 6,144 nodes. The co-activation edge space is 24× larger. Pass A (memory layout) has far more degrees of freedom to exploit, Pass B (quantization) stratifies across a much wider frequency distribution, and Pass D (specialization) can identify more low-entropy layers. The compiler's joint optimization matters more when the routing problem is richer.



## 8. Evaluation Plan

### 8.1 Metrics

| Metric | Description | Comparison Target |
|--------|-------------|-------------------|
| TPOT | Time per output token (decode phase) | ExpertFlow [13], MoE-SpeQ [9] |
| TTFT | Time to first token (prefill phase) | Vanilla llama.cpp offloading |
| Perplexity | WikiText-103 PPL vs. fp16 baseline | Uniform INT4 (bitsandbytes) |
| KL Divergence | Output distribution shift from fp16 | User-configurable budget |
| Coverage | % of expert activations covered by static prefetches | N/A (novel metric) |
| Memory | Total GPU VRAM usage of compiled artifact | Original model size |
| Compile time | Time to run full R-PGO pipeline on Qwen3-30B-A3B | N/A |

### 8.2 Ablation Study Design

```
Primary eval model: Qwen3-235B-A22B (CPU-offload on A10G + 64GB RAM)
Secondary eval:     Qwen3-30B-A3B   (fits in VRAM — tests quant quality only)
Prior art baseline: Mixtral-8x7B    (CPU-offload, for comparison with published results)

Condition A:  Baseline (llama.cpp offloading, uniform INT4)
Condition B:  + Pass B only (frequency-stratified quant, on Qwen3-30B-A3B)
Condition C:  + Pass C only (static prefetch schedules, on Qwen3-235B-A22B)
Condition D:  + Pass A only (co-activation layout, on Qwen3-235B-A22B)
Condition E:  + Pass B + Pass C (quant + prefetch)
Condition F:  Full R-PGO (all passes)
Comparison:   ExpertFlow (best published runtime system)
Comparison:   MoE-SpeQ   (best published speculative system)
```

### 8.3 Prefetch Coverage Analysis

This is a metric unique to R-PGO — no runtime system can measure it because they don't emit static schedules. The compiler can compute it statically:

```
Coverage = Σ over all (layer, expert) pairs of:
           freq(layer, expert) × 
           Σ over high-prob outgoing edges of:
             conditional_prob(edge) × I(expert was prefetched)

Interpretation:
  Coverage = 1.0: every expert activation was anticipated by a static prefetch
  Coverage = 0.9: 90% of activations (weighted by frequency) were pre-loaded
  Coverage < 0.7: calibration corpus is unrepresentative of deployment — warning
```

---

## 9. Research Positioning

### 9.1 Venue Targets

| Venue | Angle | Fit |
|-------|-------|-----|
| MLSys | Systems contribution: compiler framework for MoE inference | ★★★★★ |
| ASPLOS | Architecture + compiler co-design for heterogeneous memory | ★★★★☆ |
| ISCA | Hardware-aware inference compilation | ★★★★☆ |
| OSDI | Systems: zero-overhead serving via compile-time optimization | ★★★☆☆ |
| PLDI | PGO framework applied to a new domain | ★★★☆☆ |

MLSys is the primary target. The paper framing is: *a compiler framework that applies profile-guided optimization to MoE routing, unifying memory layout, quantization, and prefetch scheduling under a single routing-graph IR, and showing that compile-time optimization matches or exceeds the performance of runtime prediction systems with zero inference overhead.*

### 9.2 Headline Claims (Hypothesis)

1. **Quality:** R-PGO frequency-stratified quantization achieves lower perplexity than uniform INT4 at equivalent memory footprint.
2. **Latency:** R-PGO static prefetch schedules match ExpertFlow's TPOT with zero per-token prediction overhead.
3. **Composability:** Running all four passes jointly (the full compiler pipeline) outperforms any single-pass approach, demonstrating the value of joint optimization via a shared IR.
4. **Coverage:** For deployment-representative calibration data, static prefetch coverage exceeds 85% — close to the theoretical maximum achievable by any predictor.

### 9.3 Limitations to Acknowledge

1. **Calibration representativeness.** R-PGO's static schedules are only as good as the calibration corpus is representative of the deployment distribution. This is the same limitation as classical PGO. The coverage metric provides a diagnostic.

2. **Dynamic routing at test time.** For tokens that follow an unusual routing path not represented in calibration, static prefetches may miss. The compiled artifact must always include a fallback path. Claim 4 above quantifies how often this matters.

3. **Recompilation cost.** If the deployment distribution shifts significantly (domain shift), the model should be recompiled. This is an expected cost, not a bug — it is the same cost as rerunning PGO when workload changes.

---

## 10. References

[1] Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast inference from transformers via speculative decoding.* ICML 2023.

[2] McFarling, S. (1993). *Combining Branch Predictors.* DEC WRL Technical Note TN-36.

[3] Narayanan, D., et al. (2021). *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC '21.

[4] Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.

[5] Chang, P. P., Mahlke, S. A., Chen, W. Y., Warter, N. J., & Hwu, W. W. (1991). *IMPACT: An architectural framework for multiple-instruction-issue processors.* ISCA 1991.

[6] Lattner, C., & Adve, V. (2004). *LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation.* CGO 2004.

[7] Qwen Team, Alibaba Cloud. (2025). *Qwen3 Technical Report.* arXiv:2505.09388.

[8] DeepSeek-AI. (2024). *DeepSeek-V3 Technical Report.* arXiv:2412.19437.

[9] Chen, L., et al. (2025). *MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts.* arXiv:2511.14102.

[10] Chen, L., et al. (2025). *SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference.* arXiv:2510.10302.

[11] Hwang, J., et al. (2024). *Pre-Gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.* ISCA 2024.

[12] Song, S., et al. (2025). *ProMoE: Fast MoE-based LLM Serving using Proactive Caching.* arXiv preprint.

[13] Shao, H., et al. (2025). *ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling.* arXiv:2410.17954.

[14] Xue, L., et al. (2024). *MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving.* arXiv:2401.14361.

[15] Tang, P., et al. (2024). *HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference.* arXiv:2411.01433.

[16] Bondhugula, U., Hartono, A., Ramanujam, J., & Sadayappan, P. (2008). *A Practical Automatic Polyhedral Parallelizer and Locality Optimizer.* PLDI 2008.

[17] Lattner, C., et al. (2020). *MLIR: A Compiler Infrastructure for the End of Moore's Law.* arXiv:2002.11054.

[18] Cousot, P., & Cousot, R. (1977). *Abstract Interpretation: A Unified Lattice Model for Static Analysis of Programs.* POPL 1977.

[19] Zhang, Y., Chen, G., Song, F., Sun, J., & Dong, J.S. (2025). *Certified Quantization Strategy Synthesis for Neural Networks.* FM 2024, LNCS 14933.

[20] Bachiri, W., Seladji, Y., & Garoche, P. (2025). *Formal specification and SMT verification of quantized neural network for autonomous vehicles.* Science of Computer Programming.

[21] Willsey, M., et al. (2021). *egg: Fast and Extensible Equality Saturation.* POPL 2021.

[22] Muennighoff, N., et al. (2024). *A Closer Look into Mixture-of-Experts in Large Language Models.* arXiv:2406.18219.

[23] Levin, L., & Shiloach, Y. (1981). *A Linear Time Algorithm for Finding Minimum Cut in a Graph.* SIAM J. Computing.

[24] Jones, N. D., Gomard, C. K., & Sestoft, P. (1993). *Partial Evaluation and Automatic Program Generation.* Prentice Hall.

---

*Document version: 0.1 — working draft*
*Hardware target: NVIDIA A10G (24GB) + 64GB CPU RAM*
*Primary dev model: Qwen3-30B-A3B (30.5B total / 3.3B active / 128 experts)*
*Primary benchmark model: Qwen3-235B-A22B (CPU-offload mode)*
*Secondary benchmark: Mixtral-8x7B (CPU-offload, for comparison with prior art)*
