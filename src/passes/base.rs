//! Base trait + pipeline orchestrator with DAG-parallel execution.

use crate::ir::routing_graph::RoutingGraph;
use pyo3::prelude::*;
use std::collections::HashMap;

use super::layout_planner::{run_layout_pass, LayoutPlan};
use super::prefetch_emitter::{run_prefetch_pass, PrefetchSchedule};
use super::quant_planner::{run_quant_pass, QuantPlan};
use super::specialization::{run_specialization_pass, SpecializationPlan};

/// Full output of the compiler pipeline.
#[derive(Debug, Clone)]
pub struct CompilerOutput {
    pub layout: LayoutPlan,
    pub quant: QuantPlan,
    pub specialization: SpecializationPlan,
    pub prefetch: PrefetchSchedule,
}

/// Run the full 3-stage compiler pipeline.
///
/// Stage 1: A, B, D in parallel (rayon::join).
/// Stage 2: C using outputs of A, B, D (parallel across layers internally).
/// Stage 3: Assemble into CompilerOutput.
pub fn run_pipeline(graph: &RoutingGraph) -> CompilerOutput {
    // Stage 1: A, B, D are independent — run in parallel
    let (layout, (quant, specialization)) = rayon::join(
        || run_layout_pass(graph),
        || rayon::join(|| run_quant_pass(graph), || run_specialization_pass(graph)),
    );

    // Stage 2: C depends on all three
    let prefetch = run_prefetch_pass(graph, &layout, &quant, &specialization);

    CompilerOutput {
        layout,
        quant,
        specialization,
        prefetch,
    }
}

/// Run only specific passes (for ablation studies).
pub fn run_pipeline_selective(
    graph: &RoutingGraph,
    run_layout: bool,
    run_quant: bool,
    run_specialize: bool,
    run_prefetch: bool,
) -> CompilerOutput {
    let layout = if run_layout {
        run_layout_pass(graph)
    } else {
        LayoutPlan::default()
    };
    let quant = if run_quant {
        run_quant_pass(graph)
    } else {
        QuantPlan::default()
    };
    let specialization = if run_specialize {
        run_specialization_pass(graph)
    } else {
        SpecializationPlan::default()
    };
    let prefetch = if run_prefetch {
        run_prefetch_pass(graph, &layout, &quant, &specialization)
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
// PyO3 wrapper: CompilerPipeline
// ---------------------------------------------------------------------------

/// Python-exposed compiler pipeline.
#[pyclass(name = "CompilerPipeline")]
pub struct PyCompilerPipeline {
    output: Option<CompilerOutput>,
}

#[pymethods]
impl PyCompilerPipeline {
    #[new]
    fn new() -> Self {
        Self { output: None }
    }

    /// Run the full pipeline on a RoutingGraph.
    fn run(&mut self, graph: &crate::ir::PyRoutingGraph) {
        self.output = Some(run_pipeline(&graph.inner));
    }

    /// Run selective passes for ablation.
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

    /// Get the quantization plan as {(layer, expert): precision_name}.
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

    /// Get the layout plan as {(layer, expert): memory_offset}.
    fn get_layout_plan(&self) -> PyResult<HashMap<(usize, usize), u64>> {
        let out = self
            .output
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pipeline not run yet"))?;
        Ok(out.layout.placements.clone())
    }

    /// Get specialization decisions as {layer: kind}.
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

    /// Get prefetch schedule entry count.
    fn get_prefetch_entry_count(&self) -> PyResult<usize> {
        let out = self
            .output
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("pipeline not run yet"))?;
        Ok(out.prefetch.entries.len())
    }

    /// Get prefetch schedule as list of (src_layer, src_expert, dst_layer, dst_expert, priority, size_bytes).
    fn get_prefetch_schedule(&self) -> PyResult<Vec<(usize, usize, usize, usize, String, u64)>> {
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

    fn __repr__(&self) -> String {
        match &self.output {
            Some(o) => format!(
                "CompilerPipeline(quant={} assignments, layout={} placements, prefetch={} entries)",
                o.quant.assignments.len(),
                o.layout.placements.len(),
                o.prefetch.entries.len(),
            ),
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
        // All passes produce non-empty output
        assert!(!output.quant.assignments.is_empty());
        assert!(!output.layout.placements.is_empty());
        assert!(!output.specialization.layer_decisions.is_empty());
        // Prefetch should have entries since we have high-prob edges
        assert!(!output.prefetch.entries.is_empty());
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
