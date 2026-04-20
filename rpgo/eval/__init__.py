"""Evaluation utilities: benchmarks, quality metrics, coverage analysis."""

from rpgo.eval.coverage import CoverageAnalyzer
from rpgo.eval.dense_quant_apply import apply_dense_quant, build_uniform_dense_plan
from rpgo.eval.modeling import apply_precision_to_model, compile_quant_plan, load_model_and_tokenizer, model_slug, resolve_offload_folder
from rpgo.eval.quality import append_eval_result, compute_perplexity, resolve_benchmark

__all__ = [
    "CoverageAnalyzer",
    "append_eval_result",
    "apply_dense_quant",
    "build_uniform_dense_plan",
    "apply_precision_to_model",
    "compile_quant_plan",
    "compute_perplexity",
    "load_model_and_tokenizer",
    "model_slug",
    "resolve_offload_folder",
    "resolve_benchmark",
]
