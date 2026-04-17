"""End-to-end pipeline test: profile -> graph -> compile -> artifact.

Uses a mock profile (no real model needed) to test the full flow.
Tests both the manual Python graph builder and the Rust-side graph builder.
"""

import json
import tempfile
from pathlib import Path

import pytest

from rpgo._core import (
    RoutingProfile,
    LayerProfile,
    RoutingGraph,
    RoutingGraphNode,
    RoutingGraphEdge,
    CompilerPipeline,
    py_build_routing_graph,
    py_graph_summary,
)
from rpgo.eval.coverage import CoverageAnalyzer


def make_realistic_profile():
    """Create a realistic-ish routing profile for 3 layers, 8 experts, top-2."""
    profile = RoutingProfile("test-e2e-model", 500)
    profile.calibration_tokens = 5000

    # Simulate skewed expert distributions per layer
    distributions = [
        # Layer 0: highly skewed
        [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
        # Layer 1: moderately skewed
        [0.20, 0.18, 0.15, 0.13, 0.12, 0.10, 0.07, 0.05],
        # Layer 2: more uniform
        [0.16, 0.14, 0.13, 0.13, 0.12, 0.12, 0.11, 0.09],
    ]

    for layer_idx, freqs in enumerate(distributions):
        lp = LayerProfile(layer_idx, 8, 2)
        total = 1000
        for expert_id, freq in enumerate(freqs):
            count = int(freq * total)
            lp.set_expert_count(expert_id, count)

        # Add co-activation data
        if layer_idx < len(distributions) - 1:
            # Hot expert 0 strongly co-activates with next layer's expert 0
            lp.add_co_activation(0, 0)
            lp.add_co_activation(0, 0)
            lp.add_co_activation(0, 0)
            lp.add_co_activation(0, 1)
            # Hot expert 1 co-activates with expert 1 and 2
            lp.add_co_activation(1, 1)
            lp.add_co_activation(1, 1)
            lp.add_co_activation(1, 2)

        lp.finalize()
        profile.add_layer(lp)

    return profile


def build_graph_manual(profile):
    """Build routing graph from profile (manual Python-side, for comparison)."""
    graph = RoutingGraph(profile.model_id)
    layer_indices = profile.moe_layer_indices()

    for layer_idx in layer_indices:
        lp = profile.get_layer(layer_idx)
        freqs = lp.get_activation_freqs()
        for expert_id, freq in enumerate(freqs):
            node = RoutingGraphNode(
                layer=layer_idx,
                expert=expert_id,
                activation_freq=freq,
                weight_size_bytes=2_000_000,
                routing_entropy=lp.routing_entropy,
            )
            graph.add_node(node)

    # Manually add edges (simplified patterns)
    for i in range(len(layer_indices) - 1):
        src_l = layer_indices[i]
        dst_l = layer_indices[i + 1]
        graph.add_edge(RoutingGraphEdge(src_l, 0, dst_l, 0, 0.75))
        graph.add_edge(RoutingGraphEdge(src_l, 0, dst_l, 1, 0.25))
        graph.add_edge(RoutingGraphEdge(src_l, 1, dst_l, 1, 0.67))
        graph.add_edge(RoutingGraphEdge(src_l, 1, dst_l, 2, 0.33))

    return graph


class TestRustGraphBuilder:
    """Tests for py_build_routing_graph (Rust-side graph construction)."""

    def test_builds_correct_node_count(self):
        profile = make_realistic_profile()
        graph = py_build_routing_graph(profile)
        assert graph.n_nodes == 24  # 8 experts x 3 layers

    def test_builds_edges_from_co_activation(self):
        profile = make_realistic_profile()
        graph = py_build_routing_graph(profile)
        # Co-activation data was added in make_realistic_profile
        # E0->E0 (3 counts), E0->E1 (1 count), E1->E1 (2 counts), E1->E2 (1 count)
        # across 2 layer transitions = edges exist
        assert graph.n_edges > 0

    def test_hot_experts_populated(self):
        profile = make_realistic_profile()
        graph = py_build_routing_graph(profile)
        hot = graph.hot_experts(0.10)
        assert len(hot) > 0

    def test_custom_expert_size(self):
        profile = make_realistic_profile()
        graph = py_build_routing_graph(profile, expert_size_bytes=500_000)
        # Should still work, just with different weight sizes
        assert graph.n_nodes == 24

    def test_graph_summary(self):
        profile = make_realistic_profile()
        graph = py_build_routing_graph(profile)
        summary = py_graph_summary(graph)
        assert summary["total_nodes"] == 24
        assert summary["n_layers"] == 3
        assert summary["total_hot"] > 0
        assert summary["avg_entropy"] > 0


class TestEndToEnd:
    def test_profile_creation_and_validation(self):
        profile = make_realistic_profile()
        warnings = profile.validate()
        # Small rounding errors from int counts are acceptable
        for w in warnings:
            assert "frequencies sum to" in w  # expected small rounding

    def test_graph_construction_manual(self):
        profile = make_realistic_profile()
        graph = build_graph_manual(profile)
        assert graph.n_nodes == 24
        assert graph.n_edges == 8  # 4 edges x 2 layer transitions

    def test_full_pipeline_manual_graph(self):
        profile = make_realistic_profile()
        graph = build_graph_manual(profile)

        pipe = CompilerPipeline()
        pipe.run(graph)

        quant = pipe.get_quant_plan()
        assert len(quant) == 24
        layout = pipe.get_layout_plan()
        assert len(layout) == 24
        spec = pipe.get_specialization_plan()
        assert len(spec) == 3
        assert pipe.get_prefetch_entry_count() > 0

    def test_full_pipeline_rust_graph(self):
        """End-to-end via Rust graph builder: profile -> Rust graph -> compile."""
        profile = make_realistic_profile()
        graph = py_build_routing_graph(profile)

        pipe = CompilerPipeline()
        pipe.run(graph)

        quant = pipe.get_quant_plan()
        assert len(quant) == 24
        layout = pipe.get_layout_plan()
        assert len(layout) == 24
        spec = pipe.get_specialization_plan()
        assert len(spec) == 3

    def test_quant_respects_frequency(self):
        """Hot experts should get BF16, cold experts should get INT4/INT8."""
        profile = make_realistic_profile()
        graph = build_graph_manual(profile)

        pipe = CompilerPipeline()
        pipe.run(graph)
        quant = pipe.get_quant_plan()

        # Expert 0 in layer 0 (freq=0.25) should be BF16
        assert quant[(0, 0)] == "BF16", f"hot expert got {quant[(0, 0)]}"
        # Expert 7 in layer 0 (freq=0.04) should be WARM or COLD
        assert quant[(0, 7)] in ("INT8", "INT4"), f"cold expert got {quant[(0, 7)]}"

    def test_prefetch_targets_high_prob_edges(self):
        """Prefetch entries should exist for high-probability edges."""
        profile = make_realistic_profile()
        graph = build_graph_manual(profile)

        pipe = CompilerPipeline()
        pipe.run(graph)
        schedule = pipe.get_prefetch_schedule()

        # The 0.75 edge (L0:E0 -> L1:E0) should be prefetched
        high_prob_prefetches = [
            e for e in schedule
            if e[0] == 0 and e[1] == 0 and e[2] == 1 and e[3] == 0
        ]
        assert len(high_prob_prefetches) > 0, "high-prob edge not prefetched"

    def test_coverage_analysis(self):
        """Coverage analyzer should produce valid coverage ratio."""
        profile = make_realistic_profile()
        graph = build_graph_manual(profile)

        pipe = CompilerPipeline()
        pipe.run(graph)
        schedule = pipe.get_prefetch_schedule()

        analyzer = CoverageAnalyzer(graph, schedule)
        coverage = analyzer.compute_coverage()
        assert 0.0 <= coverage <= 1.0

        report = analyzer.coverage_report()
        assert "coverage_ratio" in report
        assert "interpretation" in report

    def test_profile_save_load_compile(self):
        """Test: save profile -> load -> Rust graph builder -> compile."""
        profile = make_realistic_profile()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            profile.save(path)
            loaded = RoutingProfile.load(path)
            graph = py_build_routing_graph(loaded)

            pipe = CompilerPipeline()
            pipe.run(graph)
            assert pipe.get_quant_plan()
            assert pipe.get_layout_plan()
        finally:
            Path(path).unlink()

    def test_ablation_quant_only(self):
        """Ablation: run only quantization pass."""
        profile = make_realistic_profile()
        graph = build_graph_manual(profile)

        pipe = CompilerPipeline()
        pipe.run_selective(graph, layout=False, quant=True, specialize=False, prefetch=False)

        assert len(pipe.get_quant_plan()) == 24
        assert pipe.get_prefetch_entry_count() == 0
        assert len(pipe.get_layout_plan()) == 0

    def test_ablation_layout_plus_quant(self):
        """Ablation: run layout + quant, no prefetch."""
        profile = make_realistic_profile()
        graph = build_graph_manual(profile)

        pipe = CompilerPipeline()
        pipe.run_selective(graph, layout=True, quant=True, specialize=False, prefetch=False)

        assert len(pipe.get_quant_plan()) == 24
        assert len(pipe.get_layout_plan()) == 24
        assert pipe.get_prefetch_entry_count() == 0
