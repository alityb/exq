//! Compiler passes driven by the Routing Graph IR.
//!
//! Dependency DAG:
//!
//! ```text
//!            RoutingGraph (immutable)
//!                 │
//!    ┌────────────┼────────────┐
//!    ▼            ▼            ▼
//!  Pass A       Pass B       Pass D       <- Stage 1: fully parallel
//!  (Layout)     (Quant)      (Specialize)
//!    │            │            │
//!    └────────────┴────────────┘
//!                 │
//!                 ▼
//!              Pass C                      <- Stage 2: parallel across layers
//!             (Prefetch)
//!                 │
//!                 ▼
//!          Artifact Builder                <- Stage 3
//! ```

pub mod base;
pub mod layout_planner;
pub mod prefetch_emitter;
pub mod quant_planner;
pub mod specialization;

// Selective re-exports to avoid name conflicts
pub use base::{run_pipeline, run_pipeline_with_config};
pub use base::{CompilerOutput, PipelineConfig, PyCompilerPipeline};
pub use layout_planner::PyLayoutPlan;
pub use prefetch_emitter::PyPrefetchSchedule;
pub use quant_planner::PyQuantPlan;
pub use specialization::PySpecializationPlan;
