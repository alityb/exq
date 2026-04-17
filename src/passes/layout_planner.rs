//! Pass A: Expert Memory Layout Planner.
//!
//! Places expert weight tensors in memory such that experts with high
//! co-activation probability are physically co-located, maximizing
//! prefetch and cache efficiency.
//!
//! Stage 1 pass — reads only from RoutingGraph, no dependencies on B/C/D.

use crate::ir::routing_graph::RoutingGraph;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Default HBM page size (2 MB).
const DEFAULT_PAGE_SIZE: u64 = 2 * 1024 * 1024;

/// Output of Pass A: mapping from (layer, expert) → memory offset.
#[derive(Debug, Clone, Default)]
pub struct LayoutPlan {
    pub placements: HashMap<(usize, usize), u64>,
    pub page_size: u64,
}

/// Run layout planning on the routing graph.
///
/// Algorithm (greedy co-activation clustering):
/// 1. For each layer, build weighted adjacency by co-activation probability.
/// 2. Greedily cluster: start with highest-frequency expert, attach the
///    expert with the strongest co-activation edge, repeat.
/// 3. Assign contiguous memory offsets within each layer, clusters packed together.
/// 4. Cross-layer: pack layers sequentially.
pub fn run_layout_pass(graph: &RoutingGraph) -> LayoutPlan {
    run_layout_pass_with_config(graph, DEFAULT_PAGE_SIZE)
}

pub fn run_layout_pass_with_config(graph: &RoutingGraph, page_size: u64) -> LayoutPlan {
    let mut placements = HashMap::new();
    let mut current_offset: u64 = 0;

    let layers = graph.layer_indices();

    for &layer in &layers {
        let mut nodes: Vec<_> = graph.nodes_in_layer(layer).into_iter().cloned().collect();
        if nodes.is_empty() {
            continue;
        }

        // Build a co-activation score for each expert in this layer.
        // Score = sum of conditional_prob on outgoing edges (how "connected" this expert is).
        let mut edge_scores: HashMap<usize, f64> = HashMap::new();
        for node in &nodes {
            let out = graph.outgoing_edges(node.key());
            let score: f64 = out.iter().map(|e| e.conditional_prob).sum();
            edge_scores.insert(node.expert, score);
        }

        // Sort experts: primary key = activation_freq desc, secondary = edge_score desc.
        // This puts hot, well-connected experts first → contiguous in memory.
        nodes.sort_by(|a, b| {
            let freq_cmp = b
                .activation_freq
                .partial_cmp(&a.activation_freq)
                .unwrap_or(std::cmp::Ordering::Equal);
            if freq_cmp != std::cmp::Ordering::Equal {
                return freq_cmp;
            }
            let sa = edge_scores.get(&a.expert).unwrap_or(&0.0);
            let sb = edge_scores.get(&b.expert).unwrap_or(&0.0);
            sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedy clustering pass: for each expert, check if its best co-activation
        // partner is adjacent. If not, try to swap.
        let ordered_experts = greedy_coactivation_order(&nodes, graph, layer);

        // Assign offsets
        for &expert_id in &ordered_experts {
            let node = nodes.iter().find(|n| n.expert == expert_id).unwrap();
            placements.insert((layer, expert_id), current_offset);
            current_offset += node.weight_size_bytes;
            // Align to page boundary if we've crossed a page
            // (optional: only align between clusters for locality)
        }

        // Align next layer to page boundary
        let remainder = current_offset % page_size;
        if remainder != 0 {
            current_offset += page_size - remainder;
        }
    }

    LayoutPlan {
        placements,
        page_size,
    }
}

/// Greedy co-activation ordering within a single layer.
///
/// Start with the highest-frequency expert. At each step, pick the unplaced
/// expert with the strongest co-activation link to the most recently placed expert.
/// Falls back to highest-frequency unplaced expert if no edges exist.
fn greedy_coactivation_order(
    nodes: &[crate::ir::routing_graph::RoutingGraphNode],
    graph: &RoutingGraph,
    layer: usize,
) -> Vec<usize> {
    if nodes.is_empty() {
        return vec![];
    }

    let n = nodes.len();
    let mut placed = Vec::with_capacity(n);
    let mut remaining: Vec<usize> = nodes.iter().map(|n| n.expert).collect();

    // Start with highest-freq expert (nodes already sorted by freq desc)
    placed.push(remaining.remove(0));

    while !remaining.is_empty() {
        let last_placed = *placed.last().unwrap();
        let out_edges = graph.outgoing_edges((layer, last_placed));

        // Find best co-activation partner among remaining
        let mut best_idx = None;
        let mut best_prob = -1.0f64;

        for (idx, &expert_id) in remaining.iter().enumerate() {
            // Check outgoing edges from last_placed that target this expert's
            // layer+1 counterpart — but we want intra-layer affinity.
            // Use a proxy: experts that share high-prob targets in next layer
            // are "co-active" in the sense that they'll be needed together.
            let this_edges = graph.outgoing_edges((layer, expert_id));

            // Compute Jaccard-like similarity of next-layer targets
            let mut shared_prob = 0.0f64;
            for e1 in &out_edges {
                for e2 in &this_edges {
                    if e1.dst_expert == e2.dst_expert {
                        shared_prob += e1.conditional_prob * e2.conditional_prob;
                    }
                }
            }

            if shared_prob > best_prob {
                best_prob = shared_prob;
                best_idx = Some(idx);
            }
        }

        // If no co-activation signal, just pick the highest-freq remaining
        let pick_idx = if best_prob > 0.0 {
            best_idx.unwrap()
        } else {
            0 // remaining is still sorted by freq
        };

        placed.push(remaining.remove(pick_idx));
    }

    placed
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

#[pyclass(name = "LayoutPlan")]
#[derive(Clone)]
pub struct PyLayoutPlan {
    pub inner: LayoutPlan,
}

#[pymethods]
impl PyLayoutPlan {
    #[getter]
    fn page_size(&self) -> u64 {
        self.inner.page_size
    }

    #[getter]
    fn n_placements(&self) -> usize {
        self.inner.placements.len()
    }

    fn get_offset(&self, layer: usize, expert: usize) -> PyResult<u64> {
        self.inner
            .placements
            .get(&(layer, expert))
            .copied()
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "no placement for ({layer}, {expert})"
                ))
            })
    }

    fn __repr__(&self) -> String {
        format!(
            "LayoutPlan(placements={}, page_size={})",
            self.inner.placements.len(),
            self.inner.page_size,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::routing_graph::{RoutingGraphEdge, RoutingGraphNode};

    fn make_layout_graph() -> RoutingGraph {
        let mut g = RoutingGraph::new("layout-test".into());
        for layer in 0..2 {
            for expert in 0..4 {
                let freq = [0.35, 0.30, 0.20, 0.15][expert];
                g.add_node(RoutingGraphNode {
                    layer,
                    expert,
                    activation_freq: freq,
                    weight_size_bytes: 500_000, // 0.5 MB
                    avg_arithmetic_intensity: 100.0,
                    routing_entropy: 1.3,
                });
            }
        }
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 0,
            dst_layer: 1,
            dst_expert: 0,
            conditional_prob: 0.8,
        });
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 1,
            dst_layer: 1,
            dst_expert: 0,
            conditional_prob: 0.6,
        });
        g
    }

    #[test]
    fn test_layout_all_experts_placed() {
        let g = make_layout_graph();
        let plan = run_layout_pass(&g);
        // 4 experts × 2 layers = 8 placements
        assert_eq!(plan.placements.len(), 8);
    }

    #[test]
    fn test_layout_offsets_increasing() {
        let g = make_layout_graph();
        let plan = run_layout_pass(&g);
        // Within layer 0, offsets should be strictly increasing
        let mut l0_offsets: Vec<u64> = (0..4)
            .filter_map(|e| plan.placements.get(&(0, e)).copied())
            .collect();
        l0_offsets.sort();
        for w in l0_offsets.windows(2) {
            assert!(
                w[1] > w[0],
                "offsets not strictly increasing: {:?}",
                l0_offsets
            );
        }
    }

    #[test]
    fn test_layout_layers_separated() {
        let g = make_layout_graph();
        let plan = run_layout_pass(&g);
        // Layer 1 offsets should all be > layer 0 offsets (page-aligned gap)
        let max_l0 = (0..4)
            .filter_map(|e| plan.placements.get(&(0, e)).copied())
            .max()
            .unwrap();
        let min_l1 = (0..4)
            .filter_map(|e| plan.placements.get(&(1, e)).copied())
            .min()
            .unwrap();
        assert!(min_l1 > max_l0);
    }

    #[test]
    fn test_hot_experts_placed_first() {
        let g = make_layout_graph();
        let plan = run_layout_pass(&g);
        // Expert 0 (freq=0.35) should have the lowest offset in layer 0
        let offsets_l0: Vec<(usize, u64)> = (0..4).map(|e| (e, plan.placements[&(0, e)])).collect();
        let min_expert = offsets_l0.iter().min_by_key(|x| x.1).unwrap().0;
        assert_eq!(min_expert, 0, "hottest expert should be placed first");
    }
}
