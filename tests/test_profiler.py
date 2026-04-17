"""Tests for the Python-side routing profiler.

These tests use a mock MoE model (no real weights) to verify that the
profiler correctly discovers MoE layers and collects routing statistics.
"""

import torch
import torch.nn as nn
import pytest

from rpgo.profiler.routing_profiler import RoutingProfiler, _find_moe_layers


# ---------------------------------------------------------------------------
# Mock MoE model for testing
# ---------------------------------------------------------------------------

class MockGate(nn.Module):
    """Simulates an MoE router gate: Linear(hidden, n_experts)."""

    def __init__(self, hidden_size: int, n_experts: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, n_experts, bias=False)
        # Initialize with known pattern for deterministic routing
        with torch.no_grad():
            # Expert 0 and 1 will be "hot" (large weights)
            self.linear.weight.fill_(0.0)
            self.linear.weight[0, :] = 1.0
            self.linear.weight[1, :] = 0.8

    @property
    def out_features(self):
        return self.linear.out_features

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)


class MockMoEBlock(nn.Module):
    """Simulates an MoE MLP block with a gate."""

    def __init__(self, hidden_size: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = MockGate(hidden_size, n_experts)
        self.num_experts_per_tok = top_k
        # Simplified expert computation
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(n_experts)
        ])

    def forward(self, x):
        # Just return gate logits (profiler hooks on the gate)
        gate_logits = self.gate(x)
        # Simplified: just pass through
        return x


class MockQwen3MoeSparseMoeBlock(MockMoEBlock):
    """Name triggers 'Qwen3Moe' detection pattern."""
    pass


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.mlp = MockQwen3MoeSparseMoeBlock(hidden_size, n_experts, top_k)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)
        return x


class MockMoEModel(nn.Module):
    """A minimal mock MoE model with the structure RoutingProfiler expects."""

    def __init__(self, n_layers: int = 4, hidden_size: int = 32,
                 n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerLayer(hidden_size, n_experts, top_k)
            for _ in range(n_layers)
        ])
        self.hidden_size = hidden_size

    def forward(self, input_ids=None, **kwargs):
        # Simulate: create hidden states from input_ids shape
        if input_ids is not None:
            batch, seq_len = input_ids.shape
            x = torch.randn(batch, seq_len, self.hidden_size)
        else:
            x = torch.randn(1, 10, self.hidden_size)
        for layer in self.model.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindMoELayers:
    def test_finds_all_layers(self):
        model = MockMoEModel(n_layers=4, n_experts=8)
        layers = _find_moe_layers(model)
        assert len(layers) == 4

    def test_correct_n_experts(self):
        model = MockMoEModel(n_layers=2, n_experts=16)
        layers = _find_moe_layers(model)
        for _, _, _, n_experts in layers:
            assert n_experts == 16

    def test_no_moe_model(self):
        # A model with no MoE layers
        model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))
        layers = _find_moe_layers(model)
        assert len(layers) == 0


class TestRoutingProfiler:
    def test_init(self):
        model = MockMoEModel(n_layers=4, n_experts=8)
        profiler = RoutingProfiler(model, model_id="test-mock")
        assert profiler.n_moe_layers == 4

    def test_init_no_moe(self):
        model = nn.Sequential(nn.Linear(32, 32))
        with pytest.raises(ValueError, match="No MoE layers found"):
            RoutingProfiler(model)

    def test_start_stop(self):
        model = MockMoEModel(n_layers=2, n_experts=4)
        profiler = RoutingProfiler(model)
        profiler.start()
        assert profiler._started
        profiler.stop()
        assert not profiler._started

    def test_collect_activations(self):
        model = MockMoEModel(n_layers=2, n_experts=4, top_k=2)
        profiler = RoutingProfiler(model, model_id="test")
        profiler.start()

        # Run a forward pass
        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            model(input_ids=input_ids)

        profiler.stop()
        profile = profiler.build_profile(calibration_samples=1)

        assert profile.n_layers == 2
        assert profile.calibration_tokens > 0

        # Check that activation counts are non-zero
        for layer_idx in profile.moe_layer_indices():
            lp = profile.get_layer(layer_idx)
            counts = lp.get_activation_counts()
            assert sum(counts) > 0, f"Layer {layer_idx} has zero activations"

    def test_frequencies_sum_to_one(self):
        model = MockMoEModel(n_layers=2, n_experts=8, top_k=2)
        profiler = RoutingProfiler(model, model_id="test")
        profiler.start()

        for _ in range(5):
            input_ids = torch.randint(0, 100, (2, 20))
            with torch.no_grad():
                model(input_ids=input_ids)

        profiler.stop()
        profile = profiler.build_profile(calibration_samples=10)

        for layer_idx in profile.moe_layer_indices():
            lp = profile.get_layer(layer_idx)
            freqs = lp.get_activation_freqs()
            freq_sum = sum(freqs)
            assert abs(freq_sum - 1.0) < 1e-6, (
                f"Layer {layer_idx}: freqs sum to {freq_sum}"
            )

    def test_entropy_positive(self):
        model = MockMoEModel(n_layers=2, n_experts=8, top_k=2)
        profiler = RoutingProfiler(model, model_id="test")
        profiler.start()

        for _ in range(10):
            input_ids = torch.randint(0, 100, (2, 20))
            with torch.no_grad():
                model(input_ids=input_ids)

        profiler.stop()
        profile = profiler.build_profile()

        for layer_idx in profile.moe_layer_indices():
            lp = profile.get_layer(layer_idx)
            assert lp.routing_entropy >= 0.0

    def test_profile_json_roundtrip(self):
        model = MockMoEModel(n_layers=2, n_experts=4, top_k=2)
        profiler = RoutingProfiler(model, model_id="json-test")
        profiler.start()

        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            model(input_ids=input_ids)

        profiler.stop()
        profile = profiler.build_profile(calibration_samples=1)

        json_str = profile.to_json()
        profile2 = type(profile).from_json(json_str)
        assert profile2.model_id == "json-test"
        assert profile2.n_layers == 2

    def test_reset(self):
        model = MockMoEModel(n_layers=2, n_experts=4, top_k=2)
        profiler = RoutingProfiler(model, model_id="test")
        profiler.start()

        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            model(input_ids=input_ids)

        profiler.stop()
        profiler.reset()

        assert profiler._total_tokens == 0
