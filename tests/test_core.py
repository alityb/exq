"""Tests for the Rust core types exposed via PyO3."""

import tempfile
from pathlib import Path

import pytest

from exq._core import (
    ExpertStats,
    LayerProfile,
    RoutingProfile,
    RoutingGraph,
    RoutingGraphNode,
    RoutingGraphEdge,
    CompilerPipeline,
)


# ---------------------------------------------------------------------------
# Profile types
# ---------------------------------------------------------------------------

class TestExpertStats:
    def test_create(self):
        s = ExpertStats(0)
        assert s.expert_id == 0
        assert s.activation_count == 0
        assert s.activation_freq == 0.0

    def test_set_count(self):
        s = ExpertStats(3)
        s.activation_count = 42
        assert s.activation_count == 42

    def test_repr(self):
        s = ExpertStats(1)
        assert "ExpertStats" in repr(s)


class TestLayerProfile:
    def test_create(self):
        lp = LayerProfile(0, 8, 2)
        assert lp.layer_idx == 0
        assert lp.n_experts == 8
        assert lp.top_k == 2
        assert lp.routing_entropy == 0.0

    def test_increment_expert(self):
        lp = LayerProfile(0, 4, 2)
        lp.increment_expert(0)
        lp.increment_expert(0)
        lp.increment_expert(1)
        counts = lp.get_activation_counts()
        assert counts[0] == 2
        assert counts[1] == 1
        assert counts[2] == 0

    def test_increment_out_of_range(self):
        lp = LayerProfile(0, 4, 2)
        with pytest.raises(IndexError):
            lp.increment_expert(10)

    def test_finalize(self):
        lp = LayerProfile(0, 4, 2)
        lp.set_expert_count(0, 40)
        lp.set_expert_count(1, 30)
        lp.set_expert_count(2, 20)
        lp.set_expert_count(3, 10)
        lp.finalize()
        freqs = lp.get_activation_freqs()
        assert abs(sum(freqs) - 1.0) < 1e-10
        assert freqs[0] == 0.4
        assert lp.routing_entropy > 0

    def test_co_activation(self):
        lp = LayerProfile(0, 4, 2)
        lp.add_co_activation(0, 1)
        lp.add_co_activation(0, 1)
        lp.add_co_activation(0, 2)
        # After finalize, co-activations should be normalized
        lp.set_expert_count(0, 100)
        lp.finalize()
        # Just verify it doesn't crash -- exact co-act values
        # are tested in Rust


class TestRoutingProfile:
    def test_create(self):
        p = RoutingProfile("test-model", 100)
        assert p.model_id == "test-model"
        assert p.calibration_samples == 100
        assert p.n_layers == 0

    def test_add_get_layer(self):
        p = RoutingProfile("test", 50)
        lp = LayerProfile(0, 8, 2)
        lp.set_expert_count(0, 50)
        lp.finalize()
        p.add_layer(lp)
        assert p.n_layers == 1
        assert p.moe_layer_indices() == [0]
        retrieved = p.get_layer(0)
        assert retrieved.n_experts == 8

    def test_get_missing_layer(self):
        p = RoutingProfile("test", 50)
        with pytest.raises(KeyError):
            p.get_layer(99)

    def test_json_roundtrip(self):
        p = RoutingProfile("test-model", 200)
        lp = LayerProfile(0, 4, 2)
        lp.set_expert_count(0, 40)
        lp.set_expert_count(1, 30)
        lp.set_expert_count(2, 20)
        lp.set_expert_count(3, 10)
        lp.finalize()
        p.add_layer(lp)

        json_str = p.to_json()
        p2 = RoutingProfile.from_json(json_str)
        assert p2.model_id == "test-model"
        assert p2.n_layers == 1

    def test_file_roundtrip(self):
        p = RoutingProfile("file-test", 100)
        lp = LayerProfile(0, 4, 1)
        lp.set_expert_count(0, 100)
        lp.finalize()
        p.add_layer(lp)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        p.save(path)
        p2 = RoutingProfile.load(path)
        assert p2.model_id == "file-test"
        Path(path).unlink()

    def test_validate_good(self):
        p = RoutingProfile("test", 50)
        lp = LayerProfile(0, 4, 2)
        for i in range(4):
            lp.set_expert_count(i, 25)
        lp.finalize()
        p.add_layer(lp)
        warnings = p.validate()
        assert len(warnings) == 0

    def test_validate_bad(self):
        p = RoutingProfile("test", 50)
        lp = LayerProfile(0, 4, 2)
        # Don't finalize -> freqs won't sum to 1
        p.add_layer(lp)
        warnings = p.validate()
        assert len(warnings) > 0


# ---------------------------------------------------------------------------
# Routing Graph
# ---------------------------------------------------------------------------

class TestRoutingGraph:
    def _make_graph(self):
        g = RoutingGraph("test")
        for layer in range(2):
            for expert in range(4):
                freq = [0.35, 0.30, 0.20, 0.15][expert]
                g.add_node(RoutingGraphNode(layer, expert, freq))
        g.add_edge(RoutingGraphEdge(0, 0, 1, 0, 0.80))
        g.add_edge(RoutingGraphEdge(0, 0, 1, 1, 0.20))
        g.add_edge(RoutingGraphEdge(0, 1, 1, 2, 0.50))
        return g

    def test_node_count(self):
        g = self._make_graph()
        assert g.n_nodes == 8

    def test_edge_count(self):
        g = self._make_graph()
        assert g.n_edges == 3

    def test_layer_indices(self):
        g = self._make_graph()
        assert g.layer_indices() == [0, 1]

    def test_hot_experts(self):
        g = self._make_graph()
        hot = g.hot_experts(0.10)
        assert len(hot) > 0
        for layer_idx, expert_idx, freq in hot:
            del layer_idx, expert_idx
            assert freq >= 0.10

    def test_cold_experts(self):
        g = self._make_graph()
        cold = g.cold_experts(0.20)
        assert len(cold) > 0
        for layer_idx, expert_idx, freq in cold:
            del layer_idx, expert_idx
            assert freq < 0.20

    def test_high_prob_edges(self):
        g = self._make_graph()
        high = g.high_prob_edges(0.60)
        assert len(high) == 1
        assert high[0][4] >= 0.60  # conditional_prob

    def test_json_roundtrip(self):
        g = self._make_graph()
        json_str = g.to_json()
        g2 = RoutingGraph.from_json(json_str)
        assert g2.n_nodes == g.n_nodes
        assert g2.n_edges == g.n_edges

    def test_repr(self):
        g = self._make_graph()
        r = repr(g)
        assert "RoutingGraph" in r
        assert "nodes=8" in r


# ---------------------------------------------------------------------------
# Compiler Pipeline
# ---------------------------------------------------------------------------

class TestCompilerPipeline:
    def _make_graph(self):
        g = RoutingGraph("pipeline-test")
        for layer in range(3):
            for expert in range(8):
                freq = max(0.01, 0.25 - expert * 0.03)
                g.add_node(RoutingGraphNode(
                    layer, expert, freq,
                    weight_size_bytes=2_000_000,
                    routing_entropy=1.5,
                ))
            if layer < 2:
                g.add_edge(RoutingGraphEdge(layer, 0, layer + 1, 0, 0.75))
                g.add_edge(RoutingGraphEdge(layer, 0, layer + 1, 1, 0.25))
                g.add_edge(RoutingGraphEdge(layer, 1, layer + 1, 1, 0.60))
                g.add_edge(RoutingGraphEdge(layer, 1, layer + 1, 2, 0.40))
        return g

    def test_pipeline_not_run(self):
        pipe = CompilerPipeline()
        assert "not run" in repr(pipe)

    def test_pipeline_run(self):
        g = self._make_graph()
        pipe = CompilerPipeline()
        pipe.run(g)
        assert "not run" not in repr(pipe)

    def test_quant_plan(self):
        g = self._make_graph()
        pipe = CompilerPipeline()
        pipe.run(g)
        plan = pipe.get_quant_plan()
        assert len(plan) == 24  # 8 experts × 3 layers
        # Expert 0 (freq=0.25) should be BF16
        assert plan[(0, 0)] == "BF16"

    def test_layout_plan(self):
        g = self._make_graph()
        pipe = CompilerPipeline()
        pipe.run(g)
        layout = pipe.get_layout_plan()
        assert len(layout) == 24

    def test_specialization_plan(self):
        g = self._make_graph()
        pipe = CompilerPipeline()
        pipe.run(g)
        spec = pipe.get_specialization_plan()
        assert len(spec) == 3  # 3 layers

    def test_prefetch_schedule(self):
        g = self._make_graph()
        pipe = CompilerPipeline()
        pipe.run(g)
        count = pipe.get_prefetch_entry_count()
        assert count > 0
        schedule = pipe.get_prefetch_schedule()
        assert len(schedule) == count
        # Each entry is (src_l, src_e, dst_l, dst_e, priority, size)
        for entry in schedule:
            assert len(entry) == 6
            assert entry[5] > 0  # prefetch size > 0

    def test_selective_quant_only(self):
        g = self._make_graph()
        pipe = CompilerPipeline()
        pipe.run_selective(g, layout=False, quant=True, specialize=False, prefetch=False)
        plan = pipe.get_quant_plan()
        assert len(plan) > 0
        layout = pipe.get_layout_plan()
        assert len(layout) == 0  # not run
        assert pipe.get_prefetch_entry_count() == 0

    def test_full_pipeline_deterministic(self):
        """Two runs on the same graph should produce identical results."""
        g = self._make_graph()
        pipe1 = CompilerPipeline()
        pipe1.run(g)
        pipe2 = CompilerPipeline()
        pipe2.run(g)
        assert pipe1.get_quant_plan() == pipe2.get_quant_plan()
        assert pipe1.get_prefetch_entry_count() == pipe2.get_prefetch_entry_count()
