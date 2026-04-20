"""Apply a DenseQuantPlan to a loaded transformer model."""

from __future__ import annotations

import logging
from collections import Counter

import torch
import torch.nn as nn

from rpgo.compiler.dense_quant_planner import DenseQuantPlan, HeadQuantPlan
from rpgo.eval.quant_shim import quantize_tensor_int4, quantize_tensor_int8
from rpgo.profiler.attention_profiler import _find_attention_layers

logger = logging.getLogger(__name__)

_PRECISION_ORDER = {"INT4": 0, "INT8": 1, "BF16": 2}


class HeadMixedPrecisionLinear(nn.Module):
    """Linear projection with independently quantized row slices."""

    def __init__(self, slices: list[nn.Module], bias: torch.Tensor | None = None):
        super().__init__()
        self.slices = nn.ModuleList(slices)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [slice_module(x) for slice_module in self.slices]
        result = torch.cat(outputs, dim=-1)
        if self.bias is not None:
            result = result + self.bias
        return result


class ColumnMixedPrecisionLinear(nn.Module):
    """Linear projection with independently quantized column slices."""

    def __init__(self, slices: list[nn.Module], bias: torch.Tensor | None = None):
        super().__init__()
        self.slices = nn.ModuleList(slices)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split_sizes = [slice_module.in_features for slice_module in self.slices]
        x_slices = torch.split(x, split_sizes, dim=-1)
        result = None
        for x_slice, slice_module in zip(x_slices, self.slices):
            current = slice_module(x_slice)
            result = current if result is None else result + current
        if result is None:
            raise RuntimeError("ColumnMixedPrecisionLinear has no slices")
        if self.bias is not None:
            result = result + self.bias
        return result


def _quantize_weight_slice(weight_slice: torch.Tensor, precision: str) -> torch.Tensor:
    """Fake-quantize a weight slice while preserving module compatibility."""
    if precision == "BF16":
        return weight_slice.to(weight_slice.dtype)
    if precision == "INT8":
        return quantize_tensor_int8(weight_slice)
    return quantize_tensor_int4(weight_slice)


def _build_linear_module(
    weight_slice: torch.Tensor,
    precision: str,
    bias: torch.Tensor | None = None,
) -> nn.Linear:
    """Create a standard Linear module from a quantized weight slice."""
    out_features, in_features = weight_slice.shape
    linear = nn.Linear(in_features, out_features, bias=bias is not None)
    quantized_weight = _quantize_weight_slice(weight_slice, precision)
    linear.weight = nn.Parameter(quantized_weight)
    if bias is not None:
        linear.bias = nn.Parameter(bias.to(weight_slice.dtype))
    return linear


def _find_transformer_layers(model: nn.Module):
    """Find the main transformer layer container."""
    for attr in ("model.layers", "transformer.h", "model.model.layers"):
        try:
            current = model
            for part in attr.split("."):
                current = getattr(current, part)
            return current
        except AttributeError:
            continue
    raise RuntimeError("Could not find transformer layers")


def _find_attention_module(layer: nn.Module) -> nn.Module | None:
    """Return the attention submodule for a layer if present."""
    return (
        getattr(layer, "self_attn", None)
        or getattr(layer, "attention", None)
        or getattr(layer, "attn", None)
    )


def _find_projection(attn: nn.Module, *names: str) -> tuple[str | None, nn.Module | None]:
    """Return the first projection module matching any candidate name."""
    for name in names:
        projection = getattr(attn, name, None)
        if projection is not None:
            return name, projection
    return None, None


def _kv_precisions(head_assignments: dict[int, str], n_kv_heads: int) -> list[str]:
    """Collapse query-head precisions onto KV heads for GQA/MQA models."""
    n_heads = len(head_assignments)
    if n_kv_heads <= 0:
        return []
    if n_kv_heads >= n_heads:
        return [head_assignments[idx] for idx in range(n_heads)]

    group_size = n_heads // n_kv_heads
    precisions: list[str] = []
    for kv_head_idx in range(n_kv_heads):
        start = kv_head_idx * group_size
        end = min(start + group_size, n_heads)
        query_precisions = [head_assignments[idx] for idx in range(start, end)]
        precisions.append(max(query_precisions, key=lambda precision: _PRECISION_ORDER[precision]))
    return precisions


def _replace_row_projection(
    attn: nn.Module,
    attr_name: str,
    projection: nn.Module,
    precisions: list[str],
) -> Counter:
    """Replace a row-partitioned projection such as Q/K/V."""
    counts: Counter[str] = Counter()
    weight = projection.weight.data
    bias = projection.bias.data if getattr(projection, "bias", None) is not None else None
    head_dim = weight.shape[0] // len(precisions)
    slices: list[nn.Module] = []

    for head_idx, precision in enumerate(precisions):
        row_start = head_idx * head_dim
        row_end = (head_idx + 1) * head_dim
        bias_slice = bias[row_start:row_end].clone() if bias is not None else None
        slices.append(_build_linear_module(weight[row_start:row_end, :].clone(), precision, bias_slice))
        counts[precision] += 1

    setattr(attn, attr_name, HeadMixedPrecisionLinear(slices))
    return counts


def _replace_output_projection(
    attn: nn.Module,
    attr_name: str,
    projection: nn.Module,
    precisions: list[str],
) -> Counter:
    """Replace a column-partitioned output projection such as O."""
    counts: Counter[str] = Counter()
    weight = projection.weight.data
    bias = projection.bias.data.clone() if getattr(projection, "bias", None) is not None else None
    head_dim = weight.shape[1] // len(precisions)
    slices: list[nn.Module] = []

    for head_idx, precision in enumerate(precisions):
        col_start = head_idx * head_dim
        col_end = (head_idx + 1) * head_dim
        slices.append(_build_linear_module(weight[:, col_start:col_end].clone(), precision))
        counts[precision] += 1

    setattr(attn, attr_name, ColumnMixedPrecisionLinear(slices, bias=bias))
    return counts


def apply_dense_quant(model: nn.Module, plan: DenseQuantPlan) -> nn.Module:
    """Apply a DenseQuantPlan to a model's attention projections."""
    counts: Counter[str] = Counter()
    layers_container = _find_transformer_layers(model)

    for layer_idx, layer in enumerate(layers_container):
        if layer_idx not in plan.layer_plans:
            continue

        attn = _find_attention_module(layer)
        if attn is None:
            continue

        head_plan = plan.layer_plans[layer_idx]
        n_heads = len(head_plan.assignments)
        if n_heads == 0:
            continue

        query_precisions = [head_plan.assignments[head_idx] for head_idx in sorted(head_plan.assignments)]

        q_name, q_proj = _find_projection(attn, "q_proj", "query")
        if q_name is not None and hasattr(q_proj, "weight"):
            counts.update(_replace_row_projection(attn, q_name, q_proj, query_precisions))

        k_name, k_proj = _find_projection(attn, "k_proj", "key")
        v_name, v_proj = _find_projection(attn, "v_proj", "value")
        if k_name is not None and hasattr(k_proj, "weight"):
            head_dim = q_proj.weight.shape[0] // n_heads if q_proj is not None else (k_proj.weight.shape[0] // n_heads)
            n_kv_heads = k_proj.weight.shape[0] // head_dim
            kv_precisions = _kv_precisions(head_plan.assignments, n_kv_heads)
            counts.update(_replace_row_projection(attn, k_name, k_proj, kv_precisions))
        if v_name is not None and hasattr(v_proj, "weight"):
            head_dim = q_proj.weight.shape[0] // n_heads if q_proj is not None else (v_proj.weight.shape[0] // n_heads)
            n_kv_heads = v_proj.weight.shape[0] // head_dim
            kv_precisions = _kv_precisions(head_plan.assignments, n_kv_heads)
            counts.update(_replace_row_projection(attn, v_name, v_proj, kv_precisions))

        o_name, o_proj = _find_projection(attn, "o_proj", "out_proj", "dense", "c_proj")
        if o_name is not None and hasattr(o_proj, "weight"):
            counts.update(_replace_output_projection(attn, o_name, o_proj, query_precisions))

    logger.info("Dense quant applied: %s", dict(counts))
    return model


def build_uniform_dense_plan(model: nn.Module, precision: str = "INT4") -> DenseQuantPlan:
    """Build a uniform dense-head quantization plan directly from a model."""
    layer_plans = {}
    for layer_idx, _attn, n_heads in _find_attention_layers(model):
        layer_plans[layer_idx] = HeadQuantPlan(
            layer_idx=layer_idx,
            assignments={head_idx: precision for head_idx in range(n_heads)},
            estimated_memory_ratio=1.0,
        )
    return DenseQuantPlan(model_id=getattr(getattr(model, "config", None), "_name_or_path", "unknown"), layer_plans=layer_plans)
