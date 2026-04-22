"""ExQ: Routing-Profile-Guided Optimization for MoE Inference.

Rust core (exq._core) + Python boundary layer (rpgo.profiler, rpgo.eval).
"""

__version__ = "0.1.0"

# Re-export Rust core types for convenience
from exq._core import (
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
    py_build_routing_graph,
    py_graph_summary,
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
    "py_build_routing_graph",
    "py_graph_summary",
]
