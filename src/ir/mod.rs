//! Routing Graph intermediate representation.
//!
//! The IR that all compiler passes consume. Built from a RoutingProfile.

pub mod graph_analysis;
pub mod graph_builder;
pub mod routing_graph;

pub use graph_analysis::*;
pub use graph_builder::*;
pub use routing_graph::*;
