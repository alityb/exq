#!/usr/bin/env python3
"""Evaluate perplexity for fp16, ExQ, or uniform INT4."""

import argparse
import json
import logging

import torch

from exq.eval import (
    append_eval_result,
    apply_dense_quant,
    apply_precision_to_model,
    compute_perplexity,
    resolve_benchmark,
)
from exq.eval.quant_shim import apply_quant_plan_to_model
from exq.compiler.dense_quant_planner import DenseQuantPlan
from exq.model_utils import load_model_and_tokenizer, load_artifact, parse_moe_assignments

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="ExQ: Evaluate perplexity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", required=True, choices=["fp16", "rpgo", "rpgo_dense", "int4"])
    parser.add_argument("--benchmark", choices=["wikitext2", "c4"])
    parser.add_argument("--dataset", choices=["wikitext2", "c4"])
    parser.add_argument("--profile", help="Routing profile path (rpgo without --quant-plan)")
    parser.add_argument("--quant-plan", help="Pre-built artifact path (rpgo_dense, or rpgo with artifact)")
    parser.add_argument("--log-path", default="results/eval_log.txt")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    benchmark_name = args.benchmark or args.dataset
    if benchmark_name is None:
        parser.error("one of --benchmark or --dataset is required")
    benchmark = resolve_benchmark(benchmark_name)

    if args.precision == "rpgo_dense":
        if args.quant_plan is None:
            parser.error("--quant-plan is required for rpgo_dense")
        plan = DenseQuantPlan.from_artifact(args.quant_plan, model_id=args.model)
        model, tokenizer = load_model_and_tokenizer(args.model, device_map="cpu")
        model = apply_dense_quant(model, plan)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        quant_stats = plan.summary

    elif args.precision == "rpgo" and args.quant_plan:
        artifact = load_artifact(args.quant_plan)
        assignments = parse_moe_assignments(artifact)
        model, tokenizer = load_model_and_tokenizer(args.model)
        quant_stats = apply_quant_plan_to_model(model, assignments)

    else:
        model, tokenizer = load_model_and_tokenizer(args.model)
        quant_stats = apply_precision_to_model(model, args.precision, profile_path=args.profile)

    results = compute_perplexity(
        model, tokenizer,
        dataset_name=benchmark["dataset_name"],
        dataset_config=benchmark["dataset_config"],
        split=benchmark["split"],
        text_field=benchmark["text_field"],
        max_length=args.max_length,
        stride=args.stride,
        max_samples=args.max_samples,
        model_kwargs={"use_cache": False},
        streaming=benchmark.get("streaming", False),
    )
    append_eval_result(args.log_path, args.model, args.precision, benchmark_name, results["perplexity"])

    payload = {
        "model_id": args.model,
        "precision": args.precision,
        "benchmark": benchmark_name,
        "quant_stats": quant_stats,
        **results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
