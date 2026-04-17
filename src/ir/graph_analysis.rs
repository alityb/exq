//! Graph analysis utilities: entropy queries, coverage computation,
//! frequency distribution stats, co-activation clustering hints.

use crate::ir::routing_graph::RoutingGraph;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Per-layer summary statistics derived from the routing graph.
#[derive(Debug, Clone)]
pub struct LayerAnalysis {
    pub layer: usize,
    pub n_experts: usize,
    pub routing_entropy: f64,
    /// Entropy normalized to [0,1] range: H / ln(n_experts).
    pub normalized_entropy: f64,
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
    pub frozen_count: usize,
    /// Sum of activation freqs (should be ~1.0 if graph is well-formed).
    pub freq_sum: f64,
}

/// Frequency tier thresholds (matching AGENTS.md §5.5).
pub const HOT_THRESHOLD: f64 = 0.10;
pub const WARM_THRESHOLD: f64 = 0.03;
pub const COLD_THRESHOLD: f64 = 0.005;

/// Classify an activation frequency into a tier name.
pub fn frequency_tier(freq: f64) -> &'static str {
    if freq >= HOT_THRESHOLD {
        "HOT"
    } else if freq >= WARM_THRESHOLD {
        "WARM"
    } else if freq >= COLD_THRESHOLD {
        "COLD"
    } else {
        "FROZEN"
    }
}

/// Analyze a single layer of the routing graph.
pub fn analyze_layer(graph: &RoutingGraph, layer: usize) -> LayerAnalysis {
    let nodes = graph.nodes_in_layer(layer);
    let n_experts = nodes.len();
    let entropy = nodes.first().map(|n| n.routing_entropy).unwrap_or(0.0);
    let max_entropy = if n_experts > 1 {
        (n_experts as f64).ln()
    } else {
        1.0
    };
    let normalized_entropy = if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    };

    let mut hot = 0usize;
    let mut warm = 0usize;
    let mut cold = 0usize;
    let mut frozen = 0usize;
    let mut freq_sum = 0.0f64;

    for n in &nodes {
        freq_sum += n.activation_freq;
        match frequency_tier(n.activation_freq) {
            "HOT" => hot += 1,
            "WARM" => warm += 1,
            "COLD" => cold += 1,
            _ => frozen += 1,
        }
    }

    LayerAnalysis {
        layer,
        n_experts,
        routing_entropy: entropy,
        normalized_entropy,
        hot_count: hot,
        warm_count: warm,
        cold_count: cold,
        frozen_count: frozen,
        freq_sum,
    }
}

/// Analyze all layers in the routing graph.
pub fn analyze_all_layers(graph: &RoutingGraph) -> Vec<LayerAnalysis> {
    graph
        .layer_indices()
        .into_iter()
        .map(|l| analyze_layer(graph, l))
        .collect()
}

/// Layers with routing entropy <= max_entropy (candidates for Pass D specialization).
pub fn low_entropy_layers(graph: &RoutingGraph, max_entropy: f64) -> Vec<usize> {
    graph
        .layer_indices()
        .into_iter()
        .filter(|&l| {
            let nodes = graph.nodes_in_layer(l);
            nodes
                .first()
                .map(|n| n.routing_entropy <= max_entropy)
                .unwrap_or(false)
        })
        .collect()
}

/// Compute static prefetch coverage: the fraction of weighted expert activations
/// that are anticipated by high-probability routing graph edges.
///
/// coverage = sum over (layer,expert) of:
///     freq(l,e) * sum over outgoing high-prob edges of conditional_prob
///
/// Normalized by total weighted edge mass.
pub fn prefetch_coverage(graph: &RoutingGraph, min_edge_prob: f64) -> f64 {
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
            if edge.conditional_prob >= min_edge_prob {
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

/// Build a per-layer co-activation adjacency matrix for a single layer.
/// Returns (expert_ids, matrix) where matrix[i][j] = P(expert_j at L+1 | expert_i at L).
pub fn co_activation_matrix(graph: &RoutingGraph, layer: usize) -> (Vec<usize>, Vec<Vec<f64>>) {
    let nodes = graph.nodes_in_layer(layer);
    let mut expert_ids: Vec<usize> = nodes.iter().map(|n| n.expert).collect();
    expert_ids.sort();

    let id_to_idx: HashMap<usize, usize> = expert_ids
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, i))
        .collect();

    let n = expert_ids.len();
    // Find dst layer experts
    let next_layers = graph.layer_indices();
    let next_layer = next_layers.iter().find(|&&l| l > layer).copied();

    let dst_expert_ids: Vec<usize> = if let Some(nl) = next_layer {
        let mut ids: Vec<usize> = graph.nodes_in_layer(nl).iter().map(|n| n.expert).collect();
        ids.sort();
        ids
    } else {
        return (expert_ids, vec![vec![0.0; n]; n]);
    };

    let dst_id_to_idx: HashMap<usize, usize> = dst_expert_ids
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, i))
        .collect();

    let m = dst_expert_ids.len();
    let mut matrix = vec![vec![0.0f64; m]; n];

    for (src_e, &src_idx) in &id_to_idx {
        for edge in graph.outgoing_edges((layer, *src_e)) {
            if let Some(&dst_idx) = dst_id_to_idx.get(&edge.dst_expert) {
                matrix[src_idx][dst_idx] = edge.conditional_prob;
            }
        }
    }

    (expert_ids, matrix)
}

/// Summary stats across the entire routing graph.
#[derive(Debug, Clone)]
pub struct GraphSummary {
    pub model_id: String,
    pub n_layers: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub total_hot: usize,
    pub total_warm: usize,
    pub total_cold: usize,
    pub total_frozen: usize,
    pub avg_entropy: f64,
    pub min_entropy: f64,
    pub max_entropy: f64,
    pub low_entropy_layer_count: usize,
    pub high_prob_edge_count: usize,
    pub prefetch_coverage_at_60: f64,
    pub prefetch_coverage_at_35: f64,
}

/// Compute a full summary of the routing graph.
pub fn graph_summary(graph: &RoutingGraph) -> GraphSummary {
    let analyses = analyze_all_layers(graph);
    let n_layers = analyses.len();

    let total_hot: usize = analyses.iter().map(|a| a.hot_count).sum();
    let total_warm: usize = analyses.iter().map(|a| a.warm_count).sum();
    let total_cold: usize = analyses.iter().map(|a| a.cold_count).sum();
    let total_frozen: usize = analyses.iter().map(|a| a.frozen_count).sum();

    let entropies: Vec<f64> = analyses.iter().map(|a| a.routing_entropy).collect();
    let avg_entropy = if !entropies.is_empty() {
        entropies.iter().sum::<f64>() / entropies.len() as f64
    } else {
        0.0
    };
    let min_entropy = entropies.iter().copied().fold(f64::INFINITY, f64::min);
    let max_entropy = entropies.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    GraphSummary {
        model_id: graph.model_id.clone(),
        n_layers,
        total_nodes: graph.n_nodes(),
        total_edges: graph.n_edges(),
        total_hot,
        total_warm,
        total_cold,
        total_frozen,
        avg_entropy,
        min_entropy: if min_entropy.is_infinite() {
            0.0
        } else {
            min_entropy
        },
        max_entropy: if max_entropy.is_infinite() {
            0.0
        } else {
            max_entropy
        },
        low_entropy_layer_count: low_entropy_layers(graph, 0.5).len(),
        high_prob_edge_count: graph.high_prob_edges(0.60).len(),
        prefetch_coverage_at_60: prefetch_coverage(graph, 0.60),
        prefetch_coverage_at_35: prefetch_coverage(graph, 0.35),
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

/// Compute a full summary of the routing graph (callable from Python).
/// Returns a dict with all summary fields.
#[pyfunction]
pub fn py_graph_summary(
    graph: &crate::ir::routing_graph::PyRoutingGraph,
) -> HashMap<String, PyObject> {
    use pyo3::types::PyFloat;
    let s = graph_summary(&graph.inner);
    Python::with_gil(|py| {
        let mut m = HashMap::new();
        m.insert(
            "model_id".into(),
            s.model_id.into_pyobject(py).unwrap().into_any().unbind(),
        );
        m.insert(
            "n_layers".into(),
            s.n_layers.into_pyobject(py).unwrap().into_any().unbind(),
        );
        m.insert(
            "total_nodes".into(),
            s.total_nodes.into_pyobject(py).unwrap().into_any().unbind(),
        );
        m.insert(
            "total_edges".into(),
            s.total_edges.into_pyobject(py).unwrap().into_any().unbind(),
        );
        m.insert(
            "total_hot".into(),
            s.total_hot.into_pyobject(py).unwrap().into_any().unbind(),
        );
        m.insert(
            "total_warm".into(),
            s.total_warm.into_pyobject(py).unwrap().into_any().unbind(),
        );
        m.insert(
            "total_cold".into(),
            s.total_cold.into_pyobject(py).unwrap().into_any().unbind(),
        );
        m.insert(
            "total_frozen".into(),
            s.total_frozen
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        m.insert(
            "avg_entropy".into(),
            PyFloat::new(py, s.avg_entropy).into_any().unbind(),
        );
        m.insert(
            "min_entropy".into(),
            PyFloat::new(py, s.min_entropy).into_any().unbind(),
        );
        m.insert(
            "max_entropy".into(),
            PyFloat::new(py, s.max_entropy).into_any().unbind(),
        );
        m.insert(
            "low_entropy_layer_count".into(),
            s.low_entropy_layer_count
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        m.insert(
            "high_prob_edge_count".into(),
            s.high_prob_edge_count
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        m.insert(
            "prefetch_coverage_at_60".into(),
            PyFloat::new(py, s.prefetch_coverage_at_60)
                .into_any()
                .unbind(),
        );
        m.insert(
            "prefetch_coverage_at_35".into(),
            PyFloat::new(py, s.prefetch_coverage_at_35)
                .into_any()
                .unbind(),
        );
        m
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::routing_graph::{RoutingGraphEdge, RoutingGraphNode};

    fn make_analysis_graph() -> RoutingGraph {
        let mut g = RoutingGraph::new("analysis-test".into());
        // Layer 0: 8 experts with skewed distribution
        let freqs_l0 = [0.25, 0.20, 0.15, 0.10, 0.08, 0.08, 0.07, 0.07];
        for (i, &f) in freqs_l0.iter().enumerate() {
            g.add_node(RoutingGraphNode {
                layer: 0,
                expert: i,
                activation_freq: f,
                weight_size_bytes: 1_000_000,
                avg_arithmetic_intensity: 100.0,
                routing_entropy: 1.95,
            });
        }
        // Layer 1: 8 experts, more uniform
        for i in 0..8 {
            g.add_node(RoutingGraphNode {
                layer: 1,
                expert: i,
                activation_freq: 0.125,
                weight_size_bytes: 1_000_000,
                avg_arithmetic_intensity: 100.0,
                routing_entropy: 2.08, // near max = ln(8) ≈ 2.079
            });
        }
        // Edges: L0:E0 → L1 (strong to E0, weaker to E1)
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 0,
            dst_layer: 1,
            dst_expert: 0,
            conditional_prob: 0.75,
        });
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 0,
            dst_layer: 1,
            dst_expert: 1,
            conditional_prob: 0.25,
        });
        // L0:E1 → L1
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 1,
            dst_layer: 1,
            dst_expert: 2,
            conditional_prob: 0.40,
        });
        g.add_edge(RoutingGraphEdge {
            src_layer: 0,
            src_expert: 1,
            dst_layer: 1,
            dst_expert: 3,
            conditional_prob: 0.60,
        });
        g
    }

    #[test]
    fn test_frequency_tier_classification() {
        assert_eq!(frequency_tier(0.15), "HOT");
        assert_eq!(frequency_tier(0.10), "HOT");
        assert_eq!(frequency_tier(0.05), "WARM");
        assert_eq!(frequency_tier(0.03), "WARM");
        assert_eq!(frequency_tier(0.02), "COLD");
        assert_eq!(frequency_tier(0.004), "FROZEN");
    }

    #[test]
    fn test_analyze_layer_counts() {
        let g = make_analysis_graph();
        let a0 = analyze_layer(&g, 0);
        assert_eq!(a0.n_experts, 8);
        // freqs: 0.25, 0.20, 0.15, 0.10 are HOT (>=0.10), rest WARM
        assert_eq!(a0.hot_count, 4);
        assert_eq!(a0.warm_count, 4);
        assert_eq!(a0.cold_count, 0);
    }

    #[test]
    fn test_analyze_freq_sum() {
        let g = make_analysis_graph();
        let a0 = analyze_layer(&g, 0);
        assert!((a0.freq_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_low_entropy_layers() {
        let g = make_analysis_graph();
        // Both layers have entropy > 0.5, so none qualify
        let low = low_entropy_layers(&g, 0.5);
        assert!(low.is_empty());
        // With threshold 2.0, layer 0 (1.95) qualifies
        let low2 = low_entropy_layers(&g, 2.0);
        assert_eq!(low2, vec![0]);
    }

    #[test]
    fn test_prefetch_coverage() {
        let g = make_analysis_graph();
        // At min_prob=0.60: edges with prob>=0.60 are:
        //   L0:E0→L1:E0 (0.75) and L0:E1→L1:E3 (0.60)
        let cov60 = prefetch_coverage(&g, 0.60);
        assert!(cov60 > 0.0 && cov60 <= 1.0, "coverage@0.60 = {cov60}");

        // At min_prob=0.35: all 4 edges qualify except L0:E0→L1:E1 (0.25)
        let cov35 = prefetch_coverage(&g, 0.35);
        assert!(cov35 > cov60, "lower threshold should give higher coverage");
    }

    #[test]
    fn test_graph_summary() {
        let g = make_analysis_graph();
        let s = graph_summary(&g);
        assert_eq!(s.n_layers, 2);
        assert_eq!(s.total_nodes, 16);
        assert_eq!(s.total_edges, 4);
        assert!(s.avg_entropy > 0.0);
    }

    #[test]
    fn test_co_activation_matrix() {
        let g = make_analysis_graph();
        let (experts, mat) = co_activation_matrix(&g, 0);
        assert_eq!(experts.len(), 8);
        // Expert 0 (idx 0) → Expert 0 at L1 should be 0.75
        assert!((mat[0][0] - 0.75).abs() < 1e-10);
        // Expert 0 → Expert 1 at L1 should be 0.25
        assert!((mat[0][1] - 0.25).abs() < 1e-10);
    }
}
