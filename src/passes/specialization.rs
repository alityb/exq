//! Pass D: Low-Entropy Layer Specialization.
//!
//! For layers where routing entropy is very low, emit a specialized
//! fast-path decision that skips the router and directly executes
//! the dominant expert combination.
//!
//! Stage 1 pass — reads only from RoutingGraph, no dependencies on A/B/C.

use crate::ir::routing_graph::RoutingGraph;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Entropy thresholds for layer classification (in nats).
/// These correspond roughly to the bit thresholds in AGENTS.md §5.7
/// converted: 0.5 bits ≈ 0.347 nats, 1.5 bits ≈ 1.039 nats.
const LOW_ENTROPY_NATS: f64 = 0.347;
const MED_ENTROPY_NATS: f64 = 1.039;

/// How a layer should be specialized.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerSpecialization {
    /// High entropy: no specialization, use general kernel only.
    General,
    /// Medium entropy: emit 2-3 specialized kernels + general fallback.
    MultiPath {
        /// The dominant expert combinations and their probabilities.
        dominant_combinations: Vec<(Vec<usize>, f64)>,
    },
    /// Low entropy: single specialized kernel with routing check + fallback.
    FastPath {
        /// The single dominant expert combination.
        dominant_experts: Vec<usize>,
        /// Probability mass covered by this combination.
        coverage: f64,
    },
}

/// Output of Pass D.
#[derive(Debug, Clone, Default)]
pub struct SpecializationPlan {
    pub layer_decisions: HashMap<usize, LayerSpecialization>,
}

impl Default for LayerSpecialization {
    fn default() -> Self {
        Self::General
    }
}

/// Run the specialization pass.
pub fn run_specialization_pass(graph: &RoutingGraph) -> SpecializationPlan {
    let mut decisions = HashMap::new();

    for &layer in &graph.layer_indices() {
        let nodes = graph.nodes_in_layer(layer);
        if nodes.is_empty() {
            continue;
        }

        let entropy = nodes[0].routing_entropy;
        let decision = classify_layer(entropy, &nodes);
        decisions.insert(layer, decision);
    }

    SpecializationPlan {
        layer_decisions: decisions,
    }
}

/// Classify a layer's specialization based on entropy and expert distribution.
fn classify_layer(
    entropy: f64,
    nodes: &[&crate::ir::routing_graph::RoutingGraphNode],
) -> LayerSpecialization {
    if entropy <= LOW_ENTROPY_NATS {
        // Low entropy: find the dominant expert(s)
        let mut sorted: Vec<_> = nodes
            .iter()
            .map(|n| (n.expert, n.activation_freq))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top experts that collectively cover >70% of probability mass
        let mut dominant = Vec::new();
        let mut coverage = 0.0;
        for (expert, freq) in &sorted {
            dominant.push(*expert);
            coverage += freq;
            if coverage >= 0.70 {
                break;
            }
        }

        LayerSpecialization::FastPath {
            dominant_experts: dominant,
            coverage,
        }
    } else if entropy <= MED_ENTROPY_NATS {
        // Medium entropy: find top 2-3 combinations
        let mut sorted: Vec<_> = nodes
            .iter()
            .map(|n| (n.expert, n.activation_freq))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut combos = Vec::new();
        let mut covered = 0.0;
        for (expert, freq) in &sorted {
            if combos.len() >= 3 || covered >= 0.80 {
                break;
            }
            combos.push((vec![*expert], *freq));
            covered += freq;
        }

        LayerSpecialization::MultiPath {
            dominant_combinations: combos,
        }
    } else {
        LayerSpecialization::General
    }
}

/// Check if a layer is specialized (not General).
pub fn is_specialized(plan: &SpecializationPlan, layer: usize) -> bool {
    plan.layer_decisions
        .get(&layer)
        .map(|d| !matches!(d, LayerSpecialization::General))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

#[pyclass(name = "SpecializationPlan")]
#[derive(Clone)]
pub struct PySpecializationPlan {
    pub inner: SpecializationPlan,
}

#[pymethods]
impl PySpecializationPlan {
    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.layer_decisions.len()
    }

    /// Count of layers by specialization type.
    fn type_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for d in self.inner.layer_decisions.values() {
            let key = match d {
                LayerSpecialization::General => "General",
                LayerSpecialization::MultiPath { .. } => "MultiPath",
                LayerSpecialization::FastPath { .. } => "FastPath",
            };
            *counts.entry(key.to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Check if a layer is specialized.
    fn is_specialized(&self, layer: usize) -> bool {
        is_specialized(&self.inner, layer)
    }

    fn __repr__(&self) -> String {
        let counts = self.type_counts();
        format!(
            "SpecializationPlan(General={}, MultiPath={}, FastPath={})",
            counts.get("General").unwrap_or(&0),
            counts.get("MultiPath").unwrap_or(&0),
            counts.get("FastPath").unwrap_or(&0),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::routing_graph::{RoutingGraph, RoutingGraphNode};

    fn graph_with_entropy(entropy: f64) -> RoutingGraph {
        let mut g = RoutingGraph::new("spec-test".into());
        for expert in 0..8 {
            g.add_node(RoutingGraphNode {
                layer: 0,
                expert,
                activation_freq: 0.125,
                weight_size_bytes: 1_000_000,
                avg_arithmetic_intensity: 100.0,
                routing_entropy: entropy,
            });
        }
        g
    }

    #[test]
    fn test_high_entropy_general() {
        let g = graph_with_entropy(2.0); // well above MED threshold
        let plan = run_specialization_pass(&g);
        assert!(matches!(
            plan.layer_decisions[&0],
            LayerSpecialization::General
        ));
    }

    #[test]
    fn test_medium_entropy_multipath() {
        let g = graph_with_entropy(0.8); // between LOW and MED thresholds
        let plan = run_specialization_pass(&g);
        assert!(matches!(
            plan.layer_decisions[&0],
            LayerSpecialization::MultiPath { .. }
        ));
    }

    #[test]
    fn test_low_entropy_fastpath() {
        // Create a very skewed distribution
        let mut g = RoutingGraph::new("spec-test".into());
        g.add_node(RoutingGraphNode {
            layer: 0,
            expert: 0,
            activation_freq: 0.80,
            weight_size_bytes: 1_000_000,
            avg_arithmetic_intensity: 100.0,
            routing_entropy: 0.2, // very low
        });
        g.add_node(RoutingGraphNode {
            layer: 0,
            expert: 1,
            activation_freq: 0.20,
            weight_size_bytes: 1_000_000,
            avg_arithmetic_intensity: 100.0,
            routing_entropy: 0.2,
        });
        let plan = run_specialization_pass(&g);
        match &plan.layer_decisions[&0] {
            LayerSpecialization::FastPath {
                dominant_experts,
                coverage,
            } => {
                assert!(dominant_experts.contains(&0));
                assert!(*coverage >= 0.70);
            }
            other => panic!("expected FastPath, got {other:?}"),
        }
    }

    #[test]
    fn test_is_specialized_helper() {
        let g = graph_with_entropy(0.2);
        let plan = run_specialization_pass(&g);
        assert!(is_specialized(&plan, 0));
    }

    #[test]
    fn test_is_not_specialized_general() {
        let g = graph_with_entropy(2.0);
        let plan = run_specialization_pass(&g);
        assert!(!is_specialized(&plan, 0));
    }
}
