"""R-PGO: Routing-Profile-Guided Optimization for MoE Inference.

Rust core (rpgo._core) + Python boundary layer (rpgo.profiler, rpgo.eval).
"""

__version__ = "0.1.0"

# Re-export Rust core types for convenience
from rpgo._core import (
    CompilerPipeline,
    ExpertStats,
    LayerProfile,
    LayoutPlan,
    PrefetchSchedule,
    QuantPlan,
    RoutingGraph,
    RoutingGraphEdge,
    RoutingGraphNode,
    RoutingProfile,
    SpecializationPlan,
)

__all__ = [
    "CompilerPipeline",
    "ExpertStats",
    "LayerProfile",
    "LayoutPlan",
    "PrefetchSchedule",
    "QuantPlan",
    "RoutingGraph",
    "RoutingGraphEdge",
    "RoutingGraphNode",
    "RoutingProfile",
    "SpecializationPlan",
]
