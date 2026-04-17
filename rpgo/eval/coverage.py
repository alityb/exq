"""Static prefetch coverage analysis.

Computes what fraction of expert activations are anticipated by the
compiler's static prefetch schedule -- a metric unique to R-PGO.
"""

from __future__ import annotations

from rpgo._core import RoutingGraph, RoutingGraphEdge, RoutingGraphNode


class CoverageAnalyzer:
    """Analyzes prefetch coverage of a compiled schedule against a routing graph.

    Coverage = sum over (layer, expert) of:
        freq(l, e) * sum over outgoing high-prob edges of
            conditional_prob * I(edge was prefetched)

    Normalized by total weighted edge mass.
    """

    def __init__(self, graph: RoutingGraph, prefetch_schedule: list[tuple]):
        """
        Args:
            graph: The routing graph IR.
            prefetch_schedule: List of (src_l, src_e, dst_l, dst_e, priority, size)
                               tuples from CompilerPipeline.get_prefetch_schedule().
        """
        self.graph = graph
        self._prefetched: set[tuple[int, int, int, int]] = set()
        for entry in prefetch_schedule:
            src_l, src_e, dst_l, dst_e = entry[:4]
            self._prefetched.add((src_l, src_e, dst_l, dst_e))

    def compute_coverage(self) -> float:
        """Compute the weighted prefetch coverage ratio.

        Returns:
            Float in [0, 1]. 1.0 means every activation was anticipated.
        """
        high_prob_edges = self.graph.high_prob_edges(0.35)
        if not high_prob_edges:
            return 0.0

        covered_mass = 0.0
        total_mass = 0.0

        for src_l, src_e, dst_l, dst_e, prob in high_prob_edges:
            # Get source node frequency
            hot_experts = self.graph.hot_experts(0.0)  # all experts
            src_freq = 0.0
            for l, e, f in hot_experts:
                if l == src_l and e == src_e:
                    src_freq = f
                    break

            weighted = src_freq * prob
            total_mass += weighted

            if (src_l, src_e, dst_l, dst_e) in self._prefetched:
                covered_mass += weighted

        return covered_mass / total_mass if total_mass > 0 else 0.0

    def coverage_report(self) -> dict:
        """Generate a detailed coverage report."""
        coverage = self.compute_coverage()
        n_prefetched = len(self._prefetched)

        return {
            "coverage_ratio": coverage,
            "n_prefetch_entries": n_prefetched,
            "interpretation": (
                "EXCELLENT" if coverage >= 0.85
                else "GOOD" if coverage >= 0.70
                else "FAIR" if coverage >= 0.50
                else "WARNING: calibration may be unrepresentative"
            ),
        }
