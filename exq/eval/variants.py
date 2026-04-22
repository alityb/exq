"""Shared helpers for constructing evaluated model variants."""

from __future__ import annotations

from typing import Any

from exq.compiler.dense_quant_planner import DenseQuantPlan
from exq.eval.dense_quant_apply import apply_dense_quant
from exq.eval.modeling import apply_precision_to_model
from exq.model_utils import fix_tokenizer, load_model_and_tokenizer


def load_dense_plan(artifact_path: str, model_id: str) -> DenseQuantPlan:
    return DenseQuantPlan.from_artifact(artifact_path, model_id=model_id)


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
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        fix_tokenizer(tokenizer)
        wrapper = AutoAWQForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto",
        )
        wrapper.quantize(
            tokenizer,
            quant_config={
                "zero_point": True, "q_group_size": awq_group_size,
                "w_bit": 4, "version": "GEMM",
            },
            calib_data="pileval",
            max_calib_samples=awq_calib_samples,
            max_calib_seq_len=awq_calib_seq_len,
            n_parallel_calib_samples=8,
            duo_scaling=False,
            apply_clip=True,
        )
        model = wrapper.model.to("cuda")
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
