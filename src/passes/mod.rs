//! Compiler passes driven by the Routing Graph IR.
//!
//! Dependency DAG:
//!
//! ```text
//!            RoutingGraph (immutable)
//!                 │
//!    ┌────────────┼────────────┐
//!    ▼            ▼            ▼
//!  Pass A       Pass B       Pass D       ← Stage 1: fully parallel
//!  (Layout)     (Quant)      (Specialize)
//!    │            │            │
//!    └────────────┴────────────┘
//!                 │
//!                 ▼
//!              Pass C                      ← Stage 2: parallel across layers
//!             (Prefetch)
//!                 │
//!                 ▼
//!          Artifact Builder                ← Stage 3
//! ```

pub mod base;
pub mod layout_planner;
pub mod prefetch_emitter;
pub mod quant_planner;
pub mod specialization;

pub use base::*;
pub use layout_planner::*;
pub use prefetch_emitter::*;
pub use quant_planner::*;
pub use specialization::*;
