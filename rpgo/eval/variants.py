"""Shared helpers for constructing evaluated model variants."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from transformers import AutoTokenizer

from rpgo.compiler.dense_quant_planner import DenseQuantPlan, HeadQuantPlan
from rpgo.eval.dense_quant_apply import apply_dense_quant
from rpgo.eval.modeling import apply_precision_to_model, load_model_and_tokenizer


def load_dense_plan(artifact_path: str, model_id: str) -> DenseQuantPlan:
    with open(artifact_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    layer_heads: dict[int, dict[int, str]] = defaultdict(dict)
    for key, precision in artifact["quant_assignments"].items():
        layer_idx, head_idx = map(int, key.split(":"))
        layer_heads[layer_idx][head_idx] = precision
    return DenseQuantPlan(
        model_id=artifact.get("model_id", model_id),
        layer_plans={
            idx: HeadQuantPlan(layer_idx=idx, assignments=heads, estimated_memory_ratio=1.0)
            for idx, heads in layer_heads.items()
        },
    )


def load_model_variant(
    model_id: str,
    precision: str,
    *,
    profile: str | None = None,
    quant_plan: str | None = None,
    awq_calib_samples: int = 64,
    awq_calib_seq_len: int = 512,
    awq_group_size: int = 128,
) -> tuple[Any, Any]:
    """Load a model/tokenizer pair for a given evaluation precision variant."""
    if precision == "awq_controlled":
        from awq import AutoAWQForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        wrapper = AutoAWQForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
        wrapper.quantize(
            tokenizer,
            quant_config={
                "zero_point": True,
                "q_group_size": awq_group_size,
                "w_bit": 4,
                "version": "GEMM",
            },
            calib_data="pileval",
            max_calib_samples=awq_calib_samples,
            max_calib_seq_len=awq_calib_seq_len,
            n_parallel_calib_samples=8,
            duo_scaling=False,
            apply_clip=True,
        )
        model = wrapper.model
        model = model.to("cuda")
        model.eval()
        return model, tokenizer

    model, tokenizer = load_model_and_tokenizer(model_id)
    if precision == "rpgo_dense":
        if quant_plan is None:
            raise ValueError("quant_plan is required for rpgo_dense")
        plan = load_dense_plan(quant_plan, model_id)
        model = apply_dense_quant(model, plan)
    else:
        apply_precision_to_model(model, precision, profile_path=profile)
    return model, tokenizer
