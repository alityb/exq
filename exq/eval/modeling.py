"""Model-loading helpers for ExQ evaluation scripts.

Thin wrappers around rpgo.model_utils that add quantization plan compilation.
"""

from __future__ import annotations

from pathlib import Path

from exq._core import CompilerPipeline, RoutingProfile, py_build_routing_graph
from exq.eval.dense_quant_apply import apply_dense_quant, build_uniform_dense_plan
from exq.eval.quant_shim import apply_quant_plan_to_model, apply_uniform_int4
from exq.model_utils import (
    load_model_and_tokenizer,
    model_slug,
    resolve_offload_folder,
)


def compile_quant_plan(profile_path: str | Path) -> dict[tuple[int, int], str]:
    """Compile an auto-configured quantization plan from a routing profile."""
    profile = RoutingProfile.load(str(profile_path))
    layer_indices = profile.moe_layer_indices()
    if not layer_indices:
        raise ValueError("profile contains no MoE layers")

    first_layer = profile.get_layer(layer_indices[0])
    graph = py_build_routing_graph(profile)
    pipeline = CompilerPipeline()
    pipeline.run_auto(graph, first_layer.n_experts, first_layer.top_k)
    return pipeline.get_quant_plan()


def apply_precision_to_model(
    model,
    precision: str,
    *,
    profile_path: str | Path | None = None,
) -> dict[str, int]:
    """Apply an evaluation precision mode to an already-loaded model."""
    has_moe_experts = any(
        hasattr(getattr(layer, "mlp", None), "experts")
        for layer in getattr(getattr(model, "model", None), "layers", [])
    )
    if precision == "fp16":
        return {"fp16": 1, "total": 1}
    if precision == "int4":
        if not has_moe_experts:
            plan = build_uniform_dense_plan(model, precision="INT4")
            apply_dense_quant(model, plan)
            return plan.summary
        return apply_uniform_int4(model)
    if precision == "rpgo":
        if profile_path is None:
            raise ValueError("profile_path is required for rpgo precision")
        quant_plan = compile_quant_plan(profile_path)
        return apply_quant_plan_to_model(model, quant_plan)
    raise ValueError(f"unsupported precision '{precision}'")
