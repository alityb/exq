#!/usr/bin/env python3
"""CLI: Evaluate GSM8K accuracy for fp16, ExQ, or uniform INT4."""

import argparse
import json
import logging

from exq.eval import append_eval_result, apply_precision_to_model, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="ExQ: Evaluate GSM8K")
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--precision", required=True, choices=["fp16", "rpgo", "int4"])
    parser.add_argument("--profile", help="Routing profile path required for rpgo")
    parser.add_argument("--log-path", default="results/eval_log.txt", help="Benchmark log file")
    parser.add_argument("--limit", type=int, help="Optional sample limit")
    parser.add_argument("--num-fewshot", type=int, default=5, help="Few-shot examples")
    args = parser.parse_args()

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    model, tokenizer = load_model_and_tokenizer(args.model)
    quant_stats = apply_precision_to_model(model, args.precision, profile_path=args.profile)
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False

    original_generate = model.generate

    def generate_without_cache(*generate_args, **generate_kwargs):
        generate_kwargs["use_cache"] = False
        return original_generate(*generate_args, **generate_kwargs)

    model.generate = generate_without_cache

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    eval_kwargs = {
        "model": lm,
        "tasks": ["gsm8k"],
        "num_fewshot": args.num_fewshot,
        "batch_size": 1,
    }
    if args.limit is not None:
        eval_kwargs["limit"] = args.limit

    results = evaluator.simple_evaluate(
        **eval_kwargs,
    )

    task_results = results["results"]["gsm8k"]
    accuracy = None
    for metric_name, metric_value in task_results.items():
        if metric_name.startswith("exact_match") or metric_name == "exact_match":
            accuracy = metric_value
            break
    if accuracy is None:
        raise RuntimeError(f"could not find exact-match metric in gsm8k results: {task_results}")

    append_eval_result(args.log_path, args.model, args.precision, "gsm8k", accuracy)

    payload = {
        "model_id": args.model,
        "precision": args.precision,
        "benchmark": "gsm8k",
        "accuracy": accuracy,
        "quant_stats": quant_stats,
        "raw_results": task_results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
