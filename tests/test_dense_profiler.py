"""Tests for dense attention profiling."""

from __future__ import annotations

import torch
import torch.nn as nn

from rpgo.profiler.attention_profiler import AttentionProfiler, _find_attention_layers


class MockLlamaAttention(nn.Module):
    def __init__(self, hidden_size: int = 16, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        head_dim = self.hidden_size // self.n_heads
        scales = torch.tensor([4.0, 2.0, 1.0, 0.5], device=x.device, dtype=x.dtype)
        heads = []
        for scale in scales:
            heads.append(x[..., :head_dim] * scale)
        concat = torch.cat(heads, dim=-1)
        return self.o_proj(concat)


class MockDenseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockLlamaAttention()

    def forward(self, x):
        return self.self_attn(x)


class MockDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_attention_heads": 4})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockDenseLayer(), MockDenseLayer()])

    def forward(self, input_ids=None, use_cache=False, **kwargs):
        del use_cache, kwargs
        batch, seq_len = input_ids.shape
        x = torch.randn(batch, seq_len, 16)
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_find_attention_layers():
    model = MockDenseModel()
    layers = _find_attention_layers(model)
    assert len(layers) == 2
    assert layers[0][2] == 4


def test_attention_profiler_collects_head_norms():
    model = MockDenseModel()
    profiler = AttentionProfiler(model, model_id="mock-dense")
    for _ in range(3):
        with torch.no_grad():
            model(input_ids=torch.randint(0, 100, (2, 8)))
    profile = profiler.build_profile(calibration_samples=3)
    assert profile.calibration_tokens == 48
    assert len(profile.layers) == 2
    layer0 = profile.layers[0]
    assert layer0.avg_head_norms[0] > layer0.avg_head_norms[1] > layer0.avg_head_norms[2] > layer0.avg_head_norms[3]
