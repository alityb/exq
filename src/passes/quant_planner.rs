//! Pass B: Frequency-Stratified Quantization Planner.
//!
//! Assigns bit-widths per expert based on activation frequency.
//! HOT → bf16, WARM → int8, COLD → int4, FROZEN → int4 (load-on-demand).
//!
//! Stage 1 pass — reads only from RoutingGraph, no dependencies on A/C/D.

use crate::ir::graph_analysis::FrequencyThresholds;
use crate::ir::routing_graph::RoutingGraph;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Quantization precision level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    BF16,
    FP8,
    INT8,
    INT4,
}

impl Precision {
    pub fn bytes_per_param(&self) -> f64 {
        match self {
            Precision::BF16 => 2.0,
            Precision::FP8 => 1.0,
            Precision::INT8 => 1.0,
            Precision::INT4 => 0.5,
        }
    }

    pub fn error_weight(&self) -> f64 {
        match self {
            Precision::BF16 => 0.0,
            Precision::FP8 => 0.05,
            Precision::INT8 => 0.10,
            Precision::INT4 => 0.25,
        }
    }
}

/// Output of Pass B.
#[derive(Debug, Clone, Default)]
pub struct QuantPlan {
    pub assignments: HashMap<(usize, usize), Precision>,
    pub estimated_memory_bytes: u64,
    pub estimated_weighted_error: f64,
}

/// Configuration for the quantization pass.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub thresholds: FrequencyThresholds,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            thresholds: FrequencyThresholds::default(),
        }
    }
}

impl QuantConfig {
    /// Auto-configure based on model architecture.
    pub fn auto(n_experts: usize, top_k: usize) -> Self {
        Self {
            thresholds: FrequencyThresholds::auto(n_experts, top_k),
        }
    }
}

/// Run with default thresholds.
pub fn run_quant_pass(graph: &RoutingGraph) -> QuantPlan {
    run_quant_pass_with_config(graph, &QuantConfig::default())
}

/// Run with explicit config.
pub fn run_quant_pass_with_config(graph: &RoutingGraph, config: &QuantConfig) -> QuantPlan {
    let mut assignments = HashMap::new();
    let mut total_memory: u64 = 0;
    let mut weighted_error: f64 = 0.0;

    for node in graph.nodes.values() {
        let precision = classify_precision(node.activation_freq, &config.thresholds);
        let expert_params = node.weight_size_bytes as f64 / 2.0;
        let expert_memory = (expert_params * precision.bytes_per_param()) as u64;

        total_memory += expert_memory;
        weighted_error += node.activation_freq * precision.error_weight();

        assignments.insert((node.layer, node.expert), precision);
    }

    QuantPlan {
        assignments,
        estimated_memory_bytes: total_memory,
        estimated_weighted_error: weighted_error,
    }
}

/// Classify an expert's precision tier based on activation frequency.
fn classify_precision(freq: f64, t: &FrequencyThresholds) -> Precision {
    if freq >= t.hot {
        Precision::BF16
    } else if freq >= t.warm {
        Precision::INT8
    } else {
        Precision::INT4
    }
}

/// Compute the memory savings ratio vs. an all-bf16 baseline.
pub fn memory_savings_ratio(plan: &QuantPlan, graph: &RoutingGraph) -> f64 {
    let bf16_total: u64 = graph.nodes.values().map(|n| n.weight_size_bytes).sum();
    if bf16_total == 0 {
        return 0.0;
    }
    1.0 - (plan.estimated_memory_bytes as f64 / bf16_total as f64)
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

#[pyclass(name = "QuantPlan")]
#[derive(Clone)]
pub struct PyQuantPlan {
    pub inner: QuantPlan,
}

#[pymethods]
impl PyQuantPlan {
    #[getter]
    fn n_assignments(&self) -> usize {
        self.inner.assignments.len()
    }

    #[getter]
    fn estimated_memory_bytes(&self) -> u64 {
        self.inner.estimated_memory_bytes
    }

    #[getter]
    fn estimated_weighted_error(&self) -> f64 {
        self.inner.estimated_weighted_error
    }

    fn get_precision(&self, layer: usize, expert: usize) -> PyResult<String> {
        self.inner
            .assignments
            .get(&(layer, expert))
            .map(|p| format!("{p:?}"))
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "no assignment for ({layer}, {expert})"
                ))
            })
    }

    fn tier_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for p in self.inner.assignments.values() {
            *counts.entry(format!("{p:?}")).or_insert(0) += 1;
        }
        counts
    }

    fn __repr__(&self) -> String {
        let counts = self.tier_counts_inner();
        format!(
            "QuantPlan(BF16={}, INT8={}, INT4={}, memory={:.1}MB, error={:.4})",
            counts.get(&Precision::BF16).unwrap_or(&0),
            counts.get(&Precision::INT8).unwrap_or(&0),
            counts.get(&Precision::INT4).unwrap_or(&0),
            self.inner.estimated_memory_bytes as f64 / 1_000_000.0,
            self.inner.estimated_weighted_error,
        )
    }
}

impl PyQuantPlan {
    fn tier_counts_inner(&self) -> HashMap<Precision, usize> {
        let mut counts = HashMap::new();
        for p in self.inner.assignments.values() {
            *counts.entry(*p).or_insert(0) += 1;
        }
        counts
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::routing_graph::{RoutingGraph, RoutingGraphNode};

    fn make_quant_graph() -> RoutingGraph {
        let mut g = RoutingGraph::new("quant-test".into());
        let test_data = [
            (0, 0.25),  // HOT  (default thresholds)
            (1, 0.20),  // HOT
            (2, 0.15),  // HOT
            (3, 0.10),  // HOT
            (4, 0.08),  // WARM
            (5, 0.05),  // WARM
            (6, 0.02),  // COLD
            (7, 0.003), // FROZEN
        ];
        for &(expert, freq) in &test_data {
            g.add_node(RoutingGraphNode {
                layer: 0,
                expert,
                activation_freq: freq,
                weight_size_bytes: 2_000_000,
                avg_arithmetic_intensity: 100.0,
                routing_entropy: 1.8,
            });
        }
        g
    }

    #[test]
    fn test_classification_default() {
        let t = FrequencyThresholds::default();
        assert_eq!(classify_precision(0.25, &t), Precision::BF16);
        assert_eq!(classify_precision(0.10, &t), Precision::BF16);
        assert_eq!(classify_precision(0.08, &t), Precision::INT8);
        assert_eq!(classify_precision(0.03, &t), Precision::INT8);
        assert_eq!(classify_precision(0.02, &t), Precision::INT4);
        assert_eq!(classify_precision(0.003, &t), Precision::INT4);
    }

    #[test]
    fn test_classification_auto_qwen3() {
        // 128 experts, top-8: uniform=0.0625, hot=0.125, warm=0.03125
        let t = FrequencyThresholds::auto(128, 8);
        assert_eq!(classify_precision(0.13, &t), Precision::BF16);
        assert_eq!(classify_precision(0.08, &t), Precision::INT8);
        assert_eq!(classify_precision(0.04, &t), Precision::INT8);
        assert_eq!(classify_precision(0.01, &t), Precision::INT4);
    }

    #[test]
    fn test_quant_all_experts_assigned() {
        let g = make_quant_graph();
        let plan = run_quant_pass(&g);
        assert_eq!(plan.assignments.len(), 8);
    }

    #[test]
    fn test_quant_hot_get_bf16() {
        let g = make_quant_graph();
        let plan = run_quant_pass(&g);
        assert_eq!(plan.assignments[&(0, 0)], Precision::BF16);
        assert_eq!(plan.assignments[&(0, 3)], Precision::BF16);
    }

    #[test]
    fn test_quant_warm_get_int8() {
        let g = make_quant_graph();
        let plan = run_quant_pass(&g);
        assert_eq!(plan.assignments[&(0, 4)], Precision::INT8);
    }

    #[test]
    fn test_quant_cold_get_int4() {
        let g = make_quant_graph();
        let plan = run_quant_pass(&g);
        assert_eq!(plan.assignments[&(0, 6)], Precision::INT4);
    }

    #[test]
    fn test_quant_saves_memory() {
        let g = make_quant_graph();
        let plan = run_quant_pass(&g);
        let savings = memory_savings_ratio(&plan, &g);
        assert!(savings > 0.0);
    }

    #[test]
    fn test_weighted_error_bounded() {
        let g = make_quant_graph();
        let plan = run_quant_pass(&g);
        assert!(plan.estimated_weighted_error > 0.0);
        assert!(plan.estimated_weighted_error < 0.25);
    }
}
