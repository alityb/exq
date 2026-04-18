//! Pass C: Static Prefetch Schedule Emitter.
//!
//! Embeds explicit async prefetch instructions into each layer's kernel,
//! using routing graph edge probabilities to determine what to prefetch.
//!
//! Stage 2 pass — depends on:
//!   - Pass A (layout): co-located experts → batch into single DMA
//!   - Pass B (quant): precision → prefetch SIZE
//!   - Pass D (specialization): skip prefetch for specialized layers
//!
//! Internal parallelism: layers are independent, use rayon par_iter.

use crate::ir::routing_graph::RoutingGraph;
use crate::passes::layout_planner::LayoutPlan;
use crate::passes::quant_planner::{Precision, QuantPlan};
use crate::passes::specialization::{is_specialized, SpecializationPlan};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Prefetch priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchPriority {
    High,
    Medium,
}

/// Configuration for the prefetch pass.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    pub high_prob_threshold: f64,
    pub med_prob_threshold: f64,
    /// Min activation frequency for a source expert to emit prefetches.
    pub min_src_freq: f64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            high_prob_threshold: 0.70,
            med_prob_threshold: 0.35,
            min_src_freq: 0.01,
        }
    }
}

/// Defaults tuned for high-entropy models (Qwen3-style, 128 experts).
impl PrefetchConfig {
    pub fn high_entropy() -> Self {
        Self {
            high_prob_threshold: 0.30,
            med_prob_threshold: 0.15,
            min_src_freq: 0.005,
        }
    }

    /// Auto-configure based on model architecture.
    pub fn auto(n_experts: usize, top_k: usize) -> Self {
        let uniform = top_k as f64 / n_experts as f64;
        if uniform < 0.1 {
            // Many experts, sparse routing (Qwen3-style) → lower thresholds
            Self::high_entropy()
        } else {
            Self::default()
        }
    }
}

/// Legacy constants for tests (use PrefetchConfig in production).
#[allow(dead_code)]
const HIGH_PROB_THRESHOLD: f64 = 0.70;
#[allow(dead_code)]
const MED_PROB_THRESHOLD: f64 = 0.35;

/// A single prefetch instruction to be embedded in a kernel.
#[derive(Debug, Clone)]
pub struct PrefetchEntry {
    /// Layer whose kernel this prefetch is embedded in.
    pub src_layer: usize,
    /// Expert whose kernel this prefetch is embedded in.
    pub src_expert: usize,
    /// Target layer to prefetch for.
    pub dst_layer: usize,
    /// Target expert to prefetch.
    pub dst_expert: usize,
    /// Priority of the prefetch.
    pub priority: PrefetchPriority,
    /// Conditional probability (from routing graph edge).
    pub conditional_prob: f64,
    /// Size to prefetch in bytes (depends on quant precision).
    pub prefetch_size_bytes: u64,
    /// Whether this can be batched with an adjacent prefetch (from layout plan).
    pub can_batch: bool,
}

/// Output of Pass C: the full prefetch schedule.
#[derive(Debug, Clone, Default)]
pub struct PrefetchSchedule {
    pub entries: Vec<PrefetchEntry>,
    /// Per-layer entry count for analysis.
    pub per_layer_counts: HashMap<usize, usize>,
}

/// Run the prefetch schedule pass (default config).
pub fn run_prefetch_pass(
    graph: &RoutingGraph,
    layout: &LayoutPlan,
    quant: &QuantPlan,
    specialization: &SpecializationPlan,
) -> PrefetchSchedule {
    run_prefetch_pass_with_config(
        graph,
        layout,
        quant,
        specialization,
        &PrefetchConfig::default(),
    )
}

/// Run with explicit config.
pub fn run_prefetch_pass_with_config(
    graph: &RoutingGraph,
    layout: &LayoutPlan,
    quant: &QuantPlan,
    specialization: &SpecializationPlan,
    config: &PrefetchConfig,
) -> PrefetchSchedule {
    let layers = graph.layer_indices();

    let layer_schedules: Vec<Vec<PrefetchEntry>> = layers
        .par_iter()
        .map(|&layer| compute_layer_schedule(graph, layout, quant, specialization, layer, config))
        .collect();

    let mut entries = Vec::new();
    let mut per_layer_counts = HashMap::new();
    for (i, schedule) in layer_schedules.into_iter().enumerate() {
        let layer = layers[i];
        per_layer_counts.insert(layer, schedule.len());
        entries.extend(schedule);
    }

    PrefetchSchedule {
        entries,
        per_layer_counts,
    }
}

/// Compute the prefetch schedule for a single layer.
fn compute_layer_schedule(
    graph: &RoutingGraph,
    layout: &LayoutPlan,
    quant: &QuantPlan,
    specialization: &SpecializationPlan,
    layer: usize,
    config: &PrefetchConfig,
) -> Vec<PrefetchEntry> {
    let mut entries = Vec::new();
    let nodes = graph.nodes_in_layer(layer);

    for node in &nodes {
        if node.activation_freq < config.min_src_freq {
            continue;
        }

        let out_edges = graph.outgoing_edges(node.key());

        for edge in &out_edges {
            let dst_key = edge.dst_key();

            if is_specialized(specialization, edge.dst_layer) {
                continue;
            }

            let priority = if edge.conditional_prob >= config.high_prob_threshold {
                PrefetchPriority::High
            } else if edge.conditional_prob >= config.med_prob_threshold {
                PrefetchPriority::Medium
            } else {
                continue;
            };

            // Determine prefetch size from quant plan
            let precision = quant
                .assignments
                .get(&dst_key)
                .copied()
                .unwrap_or(Precision::BF16);
            let base_size = graph
                .nodes
                .get(&dst_key)
                .map(|n| n.weight_size_bytes)
                .unwrap_or(0);
            let prefetch_size = (base_size as f64 * precision.bytes_per_param() / 2.0) as u64;

            // Check if this can be batched with co-located experts (from layout plan)
            let can_batch = check_batchable(layout, node.key(), dst_key);

            entries.push(PrefetchEntry {
                src_layer: node.layer,
                src_expert: node.expert,
                dst_layer: edge.dst_layer,
                dst_expert: edge.dst_expert,
                priority,
                conditional_prob: edge.conditional_prob,
                prefetch_size_bytes: prefetch_size,
                can_batch,
            });
        }
    }

    entries
}

/// Check if two experts are close enough in memory to batch into one DMA transfer.
fn check_batchable(layout: &LayoutPlan, src_key: (usize, usize), dst_key: (usize, usize)) -> bool {
    let src_offset = layout.placements.get(&src_key).copied();
    let dst_offset = layout.placements.get(&dst_key).copied();
    match (src_offset, dst_offset) {
        (Some(s), Some(d)) => {
            let distance = if s > d { s - d } else { d - s };
            // Batchable if within same page
            distance < layout.page_size
        }
        _ => false,
    }
}

/// Compute prefetch coverage: fraction of weighted activations covered by schedule.
pub fn compute_coverage(schedule: &PrefetchSchedule, graph: &RoutingGraph) -> f64 {
    let mut covered_mass = 0.0f64;
    let mut total_mass = 0.0f64;

    for node in graph.nodes.values() {
        let freq = node.activation_freq;
        if freq <= 0.0 {
            continue;
        }
        let out_edges = graph.outgoing_edges(node.key());
        for edge in &out_edges {
            let weighted = freq * edge.conditional_prob;
            total_mass += weighted;

            // Check if this edge is covered by a prefetch entry
            let is_covered = schedule.entries.iter().any(|e| {
                e.src_layer == edge.src_layer
                    && e.src_expert == edge.src_expert
                    && e.dst_layer == edge.dst_layer
                    && e.dst_expert == edge.dst_expert
            });
            if is_covered {
                covered_mass += weighted;
            }
        }
    }

    if total_mass > 0.0 {
        covered_mass / total_mass
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

#[pyclass(name = "PrefetchSchedule")]
#[derive(Clone)]
pub struct PyPrefetchSchedule {
    pub inner: PrefetchSchedule,
}

#[pymethods]
impl PyPrefetchSchedule {
    #[getter]
    fn n_entries(&self) -> usize {
        self.inner.entries.len()
    }

    /// Count of HIGH vs MEDIUM priority entries.
    fn priority_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for e in &self.inner.entries {
            let key = format!("{:?}", e.priority);
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    /// Count of batchable entries.
    fn batchable_count(&self) -> usize {
        self.inner.entries.iter().filter(|e| e.can_batch).count()
    }

    /// Total prefetch bytes across all entries.
    fn total_prefetch_bytes(&self) -> u64 {
        self.inner
            .entries
            .iter()
            .map(|e| e.prefetch_size_bytes)
            .sum()
    }

    fn __repr__(&self) -> String {
        let high = self
            .inner
            .entries
            .iter()
            .filter(|e| e.priority == PrefetchPriority::High)
            .count();
        let med = self.inner.entries.len() - high;
        format!(
            "PrefetchSchedule(entries={}, high={high}, med={med})",
            self.inner.entries.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::routing_graph::{RoutingGraph, RoutingGraphEdge, RoutingGraphNode};
    use crate::passes::layout_planner::run_layout_pass;
    use crate::passes::quant_planner::run_quant_pass;
    use crate::passes::specialization::run_specialization_pass;

    fn make_prefetch_graph() -> RoutingGraph {
        let mut g = RoutingGraph::new("prefetch-test".into());
        for layer in 0..3 {
            for expert in 0..4 {
                let freq = [0.35, 0.30, 0.20, 0.15][expert];
                g.add_node(RoutingGraphNode {
                    layer,
                    expert,
                    activation_freq: freq,
                    weight_size_bytes: 2_000_000,
                    avg_arithmetic_intensity: 100.0,
                    routing_entropy: 1.3,
                });
            }
            if layer < 2 {
                // High-prob edge
                g.add_edge(RoutingGraphEdge {
                    src_layer: layer,
                    src_expert: 0,
                    dst_layer: layer + 1,
                    dst_expert: 0,
                    conditional_prob: 0.80,
                });
                // Medium-prob edge
                g.add_edge(RoutingGraphEdge {
                    src_layer: layer,
                    src_expert: 0,
                    dst_layer: layer + 1,
                    dst_expert: 1,
                    conditional_prob: 0.45,
                });
                // Low-prob edge (should NOT be prefetched)
                g.add_edge(RoutingGraphEdge {
                    src_layer: layer,
                    src_expert: 1,
                    dst_layer: layer + 1,
                    dst_expert: 3,
                    conditional_prob: 0.10,
                });
                // Medium from expert 1
                g.add_edge(RoutingGraphEdge {
                    src_layer: layer,
                    src_expert: 1,
                    dst_layer: layer + 1,
                    dst_expert: 1,
                    conditional_prob: 0.50,
                });
            }
        }
        g
    }

    #[test]
    fn test_prefetch_emits_high_prob() {
        let g = make_prefetch_graph();
        let layout = run_layout_pass(&g);
        let quant = run_quant_pass(&g);
        let spec = run_specialization_pass(&g);
        let schedule = run_prefetch_pass(&g, &layout, &quant, &spec);

        // Should have entries for the 0.80 and 0.45/0.50 edges
        assert!(!schedule.entries.is_empty());

        // The 0.80 edge should produce HIGH priority
        let high: Vec<_> = schedule
            .entries
            .iter()
            .filter(|e| e.priority == PrefetchPriority::High)
            .collect();
        assert!(!high.is_empty(), "should have HIGH priority prefetches");
    }

    #[test]
    fn test_prefetch_skips_low_prob() {
        let g = make_prefetch_graph();
        let layout = run_layout_pass(&g);
        let quant = run_quant_pass(&g);
        let spec = run_specialization_pass(&g);
        let schedule = run_prefetch_pass(&g, &layout, &quant, &spec);

        // The 0.10 edge should NOT be prefetched
        let low_prob = schedule
            .entries
            .iter()
            .any(|e| e.conditional_prob < MED_PROB_THRESHOLD);
        assert!(!low_prob, "should not prefetch edges below threshold");
    }

    #[test]
    fn test_prefetch_size_reflects_quant() {
        let g = make_prefetch_graph();
        let layout = run_layout_pass(&g);
        let quant = run_quant_pass(&g);
        let spec = run_specialization_pass(&g);
        let schedule = run_prefetch_pass(&g, &layout, &quant, &spec);

        // All entries should have non-zero prefetch size
        for entry in &schedule.entries {
            assert!(entry.prefetch_size_bytes > 0);
        }
    }

    #[test]
    fn test_coverage_computation() {
        let g = make_prefetch_graph();
        let layout = run_layout_pass(&g);
        let quant = run_quant_pass(&g);
        let spec = run_specialization_pass(&g);
        let schedule = run_prefetch_pass(&g, &layout, &quant, &spec);
        let coverage = compute_coverage(&schedule, &g);
        assert!(coverage > 0.0, "coverage should be positive");
        assert!(coverage <= 1.0, "coverage should be <= 1.0");
    }

    #[test]
    fn test_per_layer_counts() {
        let g = make_prefetch_graph();
        let layout = run_layout_pass(&g);
        let quant = run_quant_pass(&g);
        let spec = run_specialization_pass(&g);
        let schedule = run_prefetch_pass(&g, &layout, &quant, &spec);

        // Layer 2 (last layer) should have 0 prefetches (no outgoing edges)
        assert_eq!(schedule.per_layer_counts.get(&2).copied().unwrap_or(0), 0);
        // Layer 0 should have some
        assert!(schedule.per_layer_counts.get(&0).copied().unwrap_or(0) > 0);
    }
}
