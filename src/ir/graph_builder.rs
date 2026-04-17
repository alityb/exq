//! Builds a RoutingGraph from a RoutingProfile.

use crate::ir::routing_graph::{RoutingGraph, RoutingGraphEdge, RoutingGraphNode};
use crate::profile::{LayerProfile, RoutingProfile};

/// Default expert weight size estimate (bytes) when unknown.
/// Qwen3-30B-A3B: ~30MB per expert block.
const DEFAULT_EXPERT_SIZE_BYTES: u64 = 30_000_000;

/// Default arithmetic intensity (FLOP/byte).
const DEFAULT_ARITHMETIC_INTENSITY: f64 = 100.0;

/// Build a RoutingGraph from a RoutingProfile.
///
/// For each MoE layer in the profile, creates one node per expert.
/// For each pair of adjacent MoE layers with co-activation data,
/// creates edges weighted by conditional probability.
pub fn build_routing_graph(profile: &RoutingProfile) -> RoutingGraph {
    build_routing_graph_with_config(
        profile,
        DEFAULT_EXPERT_SIZE_BYTES,
        DEFAULT_ARITHMETIC_INTENSITY,
    )
}

/// Build with explicit expert size + intensity overrides.
pub fn build_routing_graph_with_config(
    profile: &RoutingProfile,
    expert_size_bytes: u64,
    arithmetic_intensity: f64,
) -> RoutingGraph {
    let mut graph = RoutingGraph::new(profile.model_id.clone());
    let layer_indices = profile.moe_layer_indices();

    // --- Create nodes ---
    for &layer_idx in &layer_indices {
        let lp: &LayerProfile = &profile.layers[&layer_idx];
        for stats in &lp.expert_stats {
            graph.add_node(RoutingGraphNode {
                layer: layer_idx,
                expert: stats.expert_id,
                activation_freq: stats.activation_freq,
                weight_size_bytes: expert_size_bytes,
                avg_arithmetic_intensity: arithmetic_intensity,
                routing_entropy: lp.routing_entropy,
            });
        }
    }

    // --- Create edges (cross-layer co-activation) ---
    for (i, &src_layer_idx) in layer_indices.iter().enumerate() {
        if i + 1 >= layer_indices.len() {
            break;
        }
        let dst_layer_idx = layer_indices[i + 1];
        let src_lp = &profile.layers[&src_layer_idx];

        for (&src_expert, dsts) in &src_lp.co_activation_next_layer {
            for (&dst_expert, &prob) in dsts {
                if prob > 0.0 {
                    graph.add_edge(RoutingGraphEdge {
                        src_layer: src_layer_idx,
                        src_expert,
                        dst_layer: dst_layer_idx,
                        dst_expert,
                        conditional_prob: prob,
                    });
                }
            }
        }
    }

    graph
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::{ExpertStats, LayerProfile, RoutingProfile};
    use std::collections::HashMap;

    fn make_test_profile() -> RoutingProfile {
        let mut profile = RoutingProfile::new("test-model".into(), 100);

        // Layer 0: 4 experts
        let mut lp0 = LayerProfile::new(0, 4, 2);
        lp0.expert_stats = vec![
            ExpertStats {
                expert_id: 0,
                activation_count: 40,
                activation_freq: 0.4,
                avg_input_l2_norm: 0.0,
            },
            ExpertStats {
                expert_id: 1,
                activation_count: 30,
                activation_freq: 0.3,
                avg_input_l2_norm: 0.0,
            },
            ExpertStats {
                expert_id: 2,
                activation_count: 20,
                activation_freq: 0.2,
                avg_input_l2_norm: 0.0,
            },
            ExpertStats {
                expert_id: 3,
                activation_count: 10,
                activation_freq: 0.1,
                avg_input_l2_norm: 0.0,
            },
        ];
        lp0.routing_entropy = 1.28;
        let mut co = HashMap::new();
        let mut e0_dsts = HashMap::new();
        e0_dsts.insert(0, 0.6);
        e0_dsts.insert(1, 0.3);
        e0_dsts.insert(2, 0.1);
        co.insert(0, e0_dsts);
        let mut e1_dsts = HashMap::new();
        e1_dsts.insert(1, 0.7);
        e1_dsts.insert(3, 0.3);
        co.insert(1, e1_dsts);
        lp0.co_activation_next_layer = co;

        // Layer 1: 4 experts
        let mut lp1 = LayerProfile::new(1, 4, 2);
        lp1.expert_stats = vec![
            ExpertStats {
                expert_id: 0,
                activation_count: 25,
                activation_freq: 0.25,
                avg_input_l2_norm: 0.0,
            },
            ExpertStats {
                expert_id: 1,
                activation_count: 35,
                activation_freq: 0.35,
                avg_input_l2_norm: 0.0,
            },
            ExpertStats {
                expert_id: 2,
                activation_count: 15,
                activation_freq: 0.15,
                avg_input_l2_norm: 0.0,
            },
            ExpertStats {
                expert_id: 3,
                activation_count: 25,
                activation_freq: 0.25,
                avg_input_l2_norm: 0.0,
            },
        ];
        lp1.routing_entropy = 1.35;

        profile.layers.insert(0, lp0);
        profile.layers.insert(1, lp1);
        profile
    }

    #[test]
    fn test_build_graph_node_count() {
        let profile = make_test_profile();
        let graph = build_routing_graph(&profile);
        assert_eq!(graph.n_nodes(), 8); // 4 experts × 2 layers
    }

    #[test]
    fn test_build_graph_edge_count() {
        let profile = make_test_profile();
        let graph = build_routing_graph(&profile);
        // Expert 0: 3 edges, Expert 1: 2 edges = 5 total
        assert_eq!(graph.n_edges(), 5);
    }

    #[test]
    fn test_node_frequencies_preserved() {
        let profile = make_test_profile();
        let graph = build_routing_graph(&profile);
        let node = &graph.nodes[&(0, 0)];
        assert!((node.activation_freq - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_edge_probabilities_preserved() {
        let profile = make_test_profile();
        let graph = build_routing_graph(&profile);
        let out = graph.outgoing_edges((0, 0));
        let to_e0: Vec<_> = out.iter().filter(|e| e.dst_expert == 0).collect();
        assert_eq!(to_e0.len(), 1);
        assert!((to_e0[0].conditional_prob - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_adjacency_correct() {
        let profile = make_test_profile();
        let graph = build_routing_graph(&profile);
        assert_eq!(graph.outgoing_edges((0, 0)).len(), 3);
        assert_eq!(graph.outgoing_edges((0, 1)).len(), 2);
        assert_eq!(graph.outgoing_edges((0, 2)).len(), 0); // no co-act data
        assert_eq!(graph.incoming_edges((1, 1)).len(), 2); // from E0 and E1 in L0
    }

    #[test]
    fn test_entropy_preserved() {
        let profile = make_test_profile();
        let graph = build_routing_graph(&profile);
        let node = &graph.nodes[&(0, 0)];
        assert!((node.routing_entropy - 1.28).abs() < 1e-10);
    }
}
