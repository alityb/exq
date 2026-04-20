#!/usr/bin/env python3
"""CLI: Evaluate perplexity for fp16, R-PGO, or uniform INT4."""

import argparse
import json
import logging

from rpgo.eval import (
    append_eval_result,
    apply_precision_to_model,
    compute_perplexity,
    load_model_and_tokenizer,
    resolve_benchmark,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="R-PGO: Evaluate perplexity")
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--precision", required=True, choices=["fp16", "rpgo", "int4"])
    parser.add_argument("--benchmark", required=True, choices=["wikitext2", "c4"])
    parser.add_argument("--profile", help="Routing profile path required for rpgo")
    parser.add_argument("--log-path", default="results/eval_log.txt", help="Benchmark log file")
    parser.add_argument("--max-samples", type=int, default=200, help="Maximum text samples")
    parser.add_argument("--max-length", type=int, default=512, help="Sliding window length")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride")
    args = parser.parse_args()

    benchmark = resolve_benchmark(args.benchmark)
    model, tokenizer = load_model_and_tokenizer(args.model)
    quant_stats = apply_precision_to_model(model, args.precision, profile_path=args.profile)

    results = compute_perplexity(
        model,
        tokenizer,
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
    append_eval_result(
        args.log_path,
        args.model,
        args.precision,
        args.benchmark,
        results["perplexity"],
    )

    payload = {
        "model_id": args.model,
        "precision": args.precision,
        "benchmark": args.benchmark,
        "quant_stats": quant_stats,
        **results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
