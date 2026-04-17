//! R-PGO: Routing-Profile-Guided Optimization for MoE Inference.
//!
//! Rust core exposing profile parsing, routing graph IR, compiler passes,
//! and codegen via PyO3 to a thin Python boundary layer.

pub mod codegen;
pub mod ir;
pub mod passes;
pub mod profile;

use pyo3::prelude::*;

/// Python module entry-point: `rpgo._core`
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // -- profile types --
    m.add_class::<profile::PyExpertStats>()?;
    m.add_class::<profile::PyLayerProfile>()?;
    m.add_class::<profile::PyRoutingProfile>()?;

    // -- IR types --
    m.add_class::<ir::PyRoutingGraphNode>()?;
    m.add_class::<ir::PyRoutingGraphEdge>()?;
    m.add_class::<ir::PyRoutingGraph>()?;

    // -- passes --
    m.add_class::<passes::PyQuantPlan>()?;
    m.add_class::<passes::PyPrefetchSchedule>()?;
    m.add_class::<passes::PyLayoutPlan>()?;
    m.add_class::<passes::PySpecializationPlan>()?;
    m.add_class::<passes::PyCompilerPipeline>()?;

    Ok(())
}
