//! Base trait + pipeline orchestrator with DAG-parallel execution.

use crate::ir::graph_analysis::FrequencyThresholds;
use crate::ir::routing_graph::RoutingGraph;
use pyo3::prelude::*;
use std::collections::HashMap;

use super::layout_planner::{run_layout_pass, LayoutPlan};
use super::prefetch_emitter::{
    analyze_bandwidth_feasibility, compute_coverage, run_prefetch_pass_with_config, PrefetchConfig,
    PrefetchSchedule,
};
use super::quant_planner::{run_quant_pass_with_config, QuantConfig, QuantPlan};
use super::specialization::{run_specialization_pass, SpecializationPlan};

/// Full configuration for the compiler pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineConfig {
    pub quant: QuantConfig,
    pub prefetch: PrefetchConfig,
}

impl PipelineConfig {
    /// Auto-configure based on model architecture.
    pub fn auto(n_experts: usize, top_k: usize) -> Self {
        Self {
            quant: QuantConfig::auto(n_experts, top_k),
            prefetch: PrefetchConfig::auto(n_experts, top_k),
        }
    }

    /// Configure with explicit frequency thresholds.
    pub fn with_thresholds(freq_hot: f64, freq_warm: f64, freq_cold: f64) -> Self {
        Self {
            quant: QuantConfig {
                thresholds: FrequencyThresholds {
                    hot: freq_hot,
                    warm: freq_warm,
                    cold: freq_cold,
                },
            },
            prefetch: PrefetchConfig::default(),
        }
    }
}

/// Full output of the compiler pipeline.
#[derive(Debug, Clone)]
pub struct CompilerOutput {
    pub layout: LayoutPlan,
    pub quant: QuantPlan,
    pub specialization: SpecializationPlan,
    pub prefetch: PrefetchSchedule,
}

/// Run the full 3-stage compiler pipeline with default config.
pub fn run_pipeline(graph: &RoutingGraph) -> CompilerOutput {
    run_pipeline_with_config(graph, &PipelineConfig::default())
}

/// Run with explicit config.
pub fn run_pipeline_with_config(graph: &RoutingGraph, config: &PipelineConfig) -> CompilerOutput {
    let quant_config = config.quant.clone();
    let prefetch_config = config.prefetch.clone();

    // Stage 1: A, B, D in parallel
    let (layout, (quant, specialization)) = rayon::join(
        || run_layout_pass(graph),
        || {
            rayon::join(
                || run_quant_pass_with_config(graph, &quant_config),
                || run_specialization_pass(graph),
            )
        },
    );

    // Stage 2: C depends on all three
    let prefetch =
        run_prefetch_pass_with_config(graph, &layout, &quant, &specialization, &prefetch_config);

    CompilerOutput {
        layout,
        quant,
        specialization,
        prefetch,
    }
}

/// Run only specific passes.
pub fn run_pipeline_selective(
    graph: &RoutingGraph,
    run_layout: bool,
    run_quant: bool,
    run_specialize: bool,
    run_prefetch: bool,
) -> CompilerOutput {
    run_pipeline_selective_with_config(
        graph,
        run_layout,
        run_quant,
        run_specialize,
        run_prefetch,
        &PipelineConfig::default(),
    )
}

pub fn run_pipeline_selective_with_config(
    graph: &RoutingGraph,
    run_layout: bool,
    run_quant: bool,
    run_specialize: bool,
    run_prefetch: bool,
    config: &PipelineConfig,
) -> CompilerOutput {
    let layout = if run_layout {
        run_layout_pass(graph)
    } else {
        LayoutPlan::default()
    };
    let quant = if run_quant {
        run_quant_pass_with_config(graph, &config.quant)
    } else {
        QuantPlan::default()
    };
    let specialization = if run_specialize {
        run_specialization_pass(graph)
    } else {
        SpecializationPlan::default()
    };
    let prefetch = if run_prefetch {
        run_prefetch_pass_with_config(graph, &layout, &quant, &specialization, &config.prefetch)
    } else {
        PrefetchSchedule::default()
    };

    CompilerOutput {
        layout,
        quant,
        specialization,
        prefetch,
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------

#[pyclass(name = "CompilerPipeline")]
pub struct PyCompilerPipeline {
    output: Option<CompilerOutput>,
}

type PyPrefetchEntry = (usize, usize, usize, usize, String, u64);

#[pymethods]
impl PyCompilerPipeline {
    #[new]
    fn new() -> Self {
        Self { output: None }
    }

    /// Run the full pipeline with default config.
    fn run(&mut self, graph: &crate::ir::PyRoutingGraph) {
        self.output = Some(run_pipeline(&graph.inner));
    }

    /// Run with explicit frequency thresholds.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (graph, freq_hot=0.10, freq_warm=0.03, freq_cold=0.005, prefetch_high=0.70, prefetch_med=0.35, prefetch_min_freq=0.01))]
    fn run_with_config(
        &mut self,
        graph: &crate::ir::PyRoutingGraph,
        freq_hot: f64,
        freq_warm: f64,
        freq_cold: f64,
        prefetch_high: f64,
        prefetch_med: f64,
        prefetch_min_freq: f64,
    ) {
        let config = PipelineConfig {
            quant: QuantConfig {
                thresholds: FrequencyThresholds {
                    hot: freq_hot,
                    warm: freq_warm,
                    cold: freq_cold,
                },
            },
            prefetch: PrefetchConfig {
                high_prob_threshold: prefetch_high,
                med_prob_threshold: prefetch_med,
                min_src_freq: prefetch_min_freq,
            },
        };
        self.output = Some(run_pipeline_with_config(&graph.inner, &config));
    }

    /// Run with auto-detected thresholds based on n_experts + top_k.
    #[pyo3(signature = (graph, n_experts, top_k))]
    fn run_auto(&mut self, graph: &crate::ir::PyRoutingGraph, n_experts: usize, top_k: usize) {
        let config = PipelineConfig::auto(n_experts, top_k);
        self.output = Some(run_pipeline_with_config(&graph.inner, &config));
    }

    /// Run selective passes.
    #[pyo3(signature = (graph, layout=true, quant=true, specialize=true, prefetch=true))]
    fn run_selective(
        &mut self,
        graph: &crate::ir::PyRoutingGraph,
        layout: bool,
        quant: bool,
        specialize: bool,
        prefetch: bool,
    ) {
        self.output = Some(run_pipeline_selective(
            &graph.inner,
            layout,
            quant,
            specialize,
            prefetch,
        ));
    }

    fn get_quant_plan(&self) -> PyResult<HashMap<(usize, usize), String>> {
        let out = self
            .output
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pipeline not run yet"))?;
        Ok(out
            .quant
            .assignments
            .iter()
            .map(|(&k, v)| (k, format!("{:?}", v)))
            .collect())
    }

    fn get_layout_plan(&self) -> PyResult<HashMap<(usize, usize), u64>> {
        let out = self
            .output
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pipeline not run yet"))?;
        Ok(out.layout.placements.clone())
    }

    fn get_specialization_plan(&self) -> PyResult<HashMap<usize, String>> {
        let out = self
            .output
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pipeline not run yet"))?;
        Ok(out
            .specialization
            .layer_decisions
            .iter()
            .map(|(&l, v)| (l, format!("{:?}", v)))
            .collect())
    }

    fn get_prefetch_entry_count(&self) -> PyResult<usize> {
        let out = self
            .output
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pipeline not run yet"))?;
        Ok(out.prefetch.entries.len())
    }

    fn get_prefetch_schedule(&self) -> PyResult<Vec<PyPrefetchEntry>> {
        let out = self
            .output
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pipeline not run yet"))?;
        Ok(out
            .prefetch
            .entries
            .iter()
            .map(|e| {
                (
                    e.src_layer,
                    e.src_expert,
                    e.dst_layer,
                    e.dst_expert,
                    format!("{:?}", e.priority),
                    e.prefetch_size_bytes,
                )
            })
            .collect())
    }

    /// Compute prefetch coverage against the routing graph.
    fn get_prefetch_coverage(&self, graph: &crate::ir::PyRoutingGraph) -> PyResult<f64> {
        let out = self
            .output
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pipeline not run yet"))?;
        Ok(compute_coverage(&out.prefetch, &graph.inner))
    }

    /// Analyze bandwidth feasibility of the prefetch schedule.
    /// Returns {layer: bytes} for over-budget layers, plus summary stats.
    /// budget_per_layer_bytes = pcie_bw_gbps * layer_compute_ms / 1000 * 1e9
    #[pyo3(signature = (pcie_bw_gbps=32.0, layer_compute_ms=25.0))]
    fn get_bandwidth_analysis(
        &self,
        pcie_bw_gbps: f64,
        layer_compute_ms: f64,
    ) -> PyResult<HashMap<String, PyObject>> {
        use pyo3::types::PyFloat;

        let out = self
            .output
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pipeline not run yet"))?;

        let budget = (pcie_bw_gbps * layer_compute_ms / 1000.0 * 1e9) as u64;
        let analysis = analyze_bandwidth_feasibility(&out.prefetch, budget);

        Python::with_gil(|py| {
            let mut m = HashMap::new();
            m.insert(
                "total_bytes".into(),
                analysis
                    .total_bytes
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "max_layer_bytes".into(),
                analysis
                    .max_layer_bytes
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "budget_per_layer_bytes".into(),
                budget.into_pyobject(py).unwrap().into_any().unbind(),
            );
            m.insert(
                "over_budget_layer_count".into(),
                analysis
                    .over_budget_layers
                    .len()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "total_layers".into(),
                analysis
                    .per_layer_bytes
                    .len()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            let feasible = analysis.over_budget_layers.is_empty();
            m.insert(
                "feasible".into(),
                feasible
                    .into_pyobject(py)
                    .unwrap()
                    .to_owned()
                    .into_any()
                    .unbind(),
            );
            m.insert(
                "utilization".into(),
                PyFloat::new(
                    py,
                    if budget > 0 {
                        analysis.max_layer_bytes as f64 / budget as f64
                    } else {
                        0.0
                    },
                )
                .into_any()
                .unbind(),
            );
            Ok(m)
        })
    }

    fn __repr__(&self) -> String {
        match &self.output {
            Some(o) => {
                format!(
                "CompilerPipeline(quant={} assignments, layout={} placements, prefetch={} entries)",
                o.quant.assignments.len(), o.layout.placements.len(), o.prefetch.entries.len(),
            )
            }
            None => "CompilerPipeline(not run)".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::routing_graph::{RoutingGraph, RoutingGraphEdge, RoutingGraphNode};

    fn make_pipeline_graph() -> RoutingGraph {
        let mut g = RoutingGraph::new("pipeline-test".into());
        for layer in 0..3 {
            for expert in 0..4 {
                let freq = match expert {
                    0 => 0.35,
                    1 => 0.30,
                    2 => 0.20,
                    _ => 0.15,
                };
                g.add_node(RoutingGraphNode {
                    layer,
                    expert,
                    activation_freq: freq,
                    weight_size_bytes: 1_000_000,
                    avg_arithmetic_intensity: 100.0,
                    routing_entropy: 1.3,
                });
            }
            if layer < 2 {
                g.add_edge(RoutingGraphEdge {
                    src_layer: layer,
                    src_expert: 0,
                    dst_layer: layer + 1,
                    dst_expert: 0,
                    conditional_prob: 0.80,
                });
                g.add_edge(RoutingGraphEdge {
                    src_layer: layer,
                    src_expert: 0,
                    dst_layer: layer + 1,
                    dst_expert: 1,
                    conditional_prob: 0.20,
                });
                g.add_edge(RoutingGraphEdge {
                    src_layer: layer,
                    src_expert: 1,
                    dst_layer: layer + 1,
                    dst_expert: 1,
                    conditional_prob: 0.50,
                });
                g.add_edge(RoutingGraphEdge {
                    src_layer: layer,
                    src_expert: 1,
                    dst_layer: layer + 1,
                    dst_expert: 2,
                    conditional_prob: 0.50,
                });
            }
        }
        g
    }

    #[test]
    fn test_full_pipeline_runs() {
        let g = make_pipeline_graph();
        let output = run_pipeline(&g);
        assert!(!output.quant.assignments.is_empty());
        assert!(!output.layout.placements.is_empty());
        assert!(!output.specialization.layer_decisions.is_empty());
        assert!(!output.prefetch.entries.is_empty());
    }

    #[test]
    fn test_pipeline_with_config() {
        let g = make_pipeline_graph();
        let config = PipelineConfig::with_thresholds(0.30, 0.15, 0.05);
        let output = run_pipeline_with_config(&g, &config);
        // With hot=0.30, only experts 0 (0.35) and 1 (0.30) are BF16
        let bf16_count = output
            .quant
            .assignments
            .values()
            .filter(|&&p| p == crate::passes::quant_planner::Precision::BF16)
            .count();
        assert_eq!(bf16_count, 6); // 2 experts × 3 layers
    }

    #[test]
    fn test_auto_config() {
        let g = make_pipeline_graph();
        // 4 experts, top-2: uniform=0.50, hot=1.0 (capped at 0.99), warm=0.25
        let config = PipelineConfig::auto(4, 2);
        let output = run_pipeline_with_config(&g, &config);
        assert!(!output.quant.assignments.is_empty());
    }

    #[test]
    fn test_selective_pipeline() {
        let g = make_pipeline_graph();
        let output = run_pipeline_selective(&g, true, false, false, false);
        assert!(!output.layout.placements.is_empty());
        assert!(output.quant.assignments.is_empty());
        assert!(output.specialization.layer_decisions.is_empty());
        assert!(output.prefetch.entries.is_empty());
    }
}
