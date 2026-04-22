"""Evaluation utilities: benchmarks, quality metrics, coverage analysis."""

from exq.eval.bench import compute_recovery, measure_tpot, parse_eval_log, summarize_latencies
from exq.eval.coverage import CoverageAnalyzer
from exq.eval.dense_quant_apply import apply_dense_quant, build_uniform_dense_plan
from exq.eval.modeling import apply_precision_to_model, compile_quant_plan
from exq.eval.quality import append_eval_result, compute_kl_divergence, compute_perplexity, resolve_benchmark
from exq.eval.variants import load_dense_plan, load_model_variant
from exq.model_utils import load_model_and_tokenizer, model_slug, resolve_offload_folder

__all__ = [
    "CoverageAnalyzer",
    "append_eval_result",
    "apply_dense_quant",
    "apply_precision_to_model",
    "build_uniform_dense_plan",
    "compile_quant_plan",
    "compute_kl_divergence",
    "compute_perplexity",
    "compute_recovery",
    "load_dense_plan",
    "load_model_and_tokenizer",
    "load_model_variant",
    "measure_tpot",
    "model_slug",
    "parse_eval_log",
    "resolve_benchmark",
    "resolve_offload_folder",
    "summarize_latencies",
]
