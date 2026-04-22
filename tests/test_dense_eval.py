"""Tests for dense mixed-precision attention application."""

from __future__ import annotations

import torch
import torch.nn as nn

from exq.compiler.dense_quant_planner import DenseQuantPlan, HeadQuantPlan
from exq.eval.dense_quant_apply import ColumnMixedPrecisionLinear, HeadMixedPrecisionLinear, apply_dense_quant, build_uniform_dense_plan
from exq.eval.modeling import apply_precision_to_model


class TinyAttention(nn.Module):
    def __init__(self, hidden_size: int = 8, n_heads: int = 2):
        super().__init__()
        self.num_heads = n_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q + k + v)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()

    def forward(self, x):
        return self.self_attn(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_apply_dense_quant_replaces_attention_projections():
    model = TinyModel().to(torch.float16)
    plan = DenseQuantPlan(
        model_id="tiny",
        layer_plans={0: HeadQuantPlan(layer_idx=0, assignments={0: "BF16", 1: "INT4"}, estimated_memory_ratio=0.625)},
    )
    quantized = apply_dense_quant(model, plan)
    attn = quantized.model.layers[0].self_attn
    assert isinstance(attn.q_proj, HeadMixedPrecisionLinear)
    assert isinstance(attn.o_proj, ColumnMixedPrecisionLinear)

    x = torch.randn(2, 4, 8, dtype=torch.float16)
    with torch.no_grad():
        out = quantized(x)
    assert torch.isfinite(out).all()


def test_apply_precision_to_model_uses_dense_uniform_int4_for_dense_models():
    model = TinyModel().to(torch.float16)
    stats = apply_precision_to_model(model, "int4")
    assert stats["INT4"] == 2
    assert isinstance(model.model.layers[0].self_attn.q_proj, HeadMixedPrecisionLinear)


def test_build_uniform_dense_plan_uses_all_heads():
    model = TinyModel().to(torch.float16)
    plan = build_uniform_dense_plan(model, precision="INT4")
    assert plan.layer_plans[0].assignments == {0: "INT4", 1: "INT4"}
