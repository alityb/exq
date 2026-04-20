"""Model-loading helpers for R-PGO evaluation scripts.

This module keeps evaluation orchestration in Python while delegating all
compiler decisions to the Rust core.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from rpgo._core import CompilerPipeline, RoutingProfile, py_build_routing_graph
from rpgo.hf_compat import patch_transformers_remote_code_compat
from rpgo.eval.dense_quant_apply import apply_dense_quant, build_uniform_dense_plan
from rpgo.eval.quant_shim import apply_quant_plan_to_model, apply_uniform_int4


def model_slug(model_id: str) -> str:
    """Convert a Hugging Face model id into a stable filesystem slug."""
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def resolve_offload_folder(offload_folder: str | None = None) -> str:
    """Resolve the accelerate offload directory for model loading."""
    resolved = (
        offload_folder
        or os.environ.get("RPGO_OFFLOAD_DIR")
        or str(Path(tempfile.gettempdir()) / "rpgo_offload")
    )
    Path(resolved).mkdir(parents=True, exist_ok=True)
    return resolved


def load_model_and_tokenizer(
    model_id: str,
    *,
    load_in_4bit: bool = False,
    dtype: Any = torch.float16,
    offload_folder: str | None = None,
):
    """Load a causal LM and tokenizer for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    patch_transformers_remote_code_compat()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_offload = resolve_offload_folder(offload_folder)
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "offload_folder": resolved_offload,
        "trust_remote_code": True,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        if torch.cuda.is_available():
            load_kwargs["device_map"] = {"": 0}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        load_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


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
