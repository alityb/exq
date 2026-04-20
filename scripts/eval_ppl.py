#!/usr/bin/env python3
"""CLI: Evaluate perplexity for fp16, R-PGO, or uniform INT4."""

import argparse
import json
import logging

import torch

from rpgo.eval import (
    append_eval_result,
    apply_dense_quant,
    apply_precision_to_model,
    compute_perplexity,
    load_model_and_tokenizer,
    resolve_benchmark,
    resolve_offload_folder,
)
from rpgo.hf_compat import patch_transformers_remote_code_compat

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="R-PGO: Evaluate perplexity")
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--precision", required=True, choices=["fp16", "rpgo", "rpgo_dense", "int4"])
    parser.add_argument("--benchmark", choices=["wikitext2", "c4"])
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], help="Alias for --benchmark")
    parser.add_argument("--profile", help="Routing profile path required for rpgo")
    parser.add_argument("--quant-plan", help="Dense quant-plan artifact path for rpgo_dense")
    parser.add_argument("--log-path", default="results/eval_log.txt", help="Benchmark log file")
    parser.add_argument("--max-samples", type=int, default=200, help="Maximum text samples")
    parser.add_argument("--max-length", type=int, default=512, help="Sliding window length")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride")
    args = parser.parse_args()

    benchmark_name = args.benchmark or args.dataset
    if benchmark_name is None:
        parser.error("one of --benchmark or --dataset is required")

    benchmark = resolve_benchmark(benchmark_name)

    if args.precision == "rpgo_dense":
        if args.quant_plan is None:
            parser.error("--quant-plan is required for rpgo_dense")
        from rpgo.compiler.dense_quant_planner import DenseQuantPlan, HeadQuantPlan

        patch_transformers_remote_code_compat()
        with open(args.quant_plan, encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("type") != "dense_head_quant":
            raise ValueError(
                "Use rpgo precision for MoE artifacts and rpgo_dense for dense artifacts"
            )

        from collections import defaultdict
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        layer_heads: dict[int, dict[int, str]] = defaultdict(dict)
        for key, precision in artifact["quant_assignments"].items():
            layer_idx, head_idx = map(int, key.split(":"))
            layer_heads[layer_idx][head_idx] = precision
        layer_plans = {
            layer_idx: HeadQuantPlan(
                layer_idx=layer_idx,
                assignments=heads,
                estimated_memory_ratio=1.0,
            )
            for layer_idx, heads in layer_heads.items()
        }
        plan = DenseQuantPlan(
            model_id=artifact.get("model_id", args.model),
            layer_plans=layer_plans,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cpu",
            offload_folder=resolve_offload_folder(),
            trust_remote_code=True,
        )
        model = apply_dense_quant(model, plan)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        quant_stats = plan.summary
    else:
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
        benchmark_name,
        results["perplexity"],
    )

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
