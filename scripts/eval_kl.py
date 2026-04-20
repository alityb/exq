#!/usr/bin/env python3
"""Compute KL divergence between a reference fp16 model and a quantized model."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="R-PGO KL divergence evaluation")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-precision", required=True, choices=["fp16", "int4", "rpgo", "rpgo_dense"])
    parser.add_argument("--profile")
    parser.add_argument("--quant-plan")
    parser.add_argument("--benchmark", choices=["wikitext2", "c4"], required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    from rpgo.eval import compute_kl_divergence, load_model_and_tokenizer, apply_precision_to_model, apply_dense_quant, resolve_benchmark

    ref_model, tokenizer = load_model_and_tokenizer(args.reference_model)
    cand_model, _ = load_model_and_tokenizer(args.candidate_model)

    if args.candidate_precision == "rpgo_dense":
        if args.quant_plan is None:
            raise ValueError("--quant-plan is required for rpgo_dense")
        from rpgo.compiler.dense_quant_planner import DenseQuantPlan, HeadQuantPlan
        from collections import defaultdict

        with open(args.quant_plan, encoding="utf-8") as handle:
            artifact = json.load(handle)
        layer_heads = defaultdict(dict)
        for key, precision in artifact["quant_assignments"].items():
            layer_idx, head_idx = map(int, key.split(":"))
            layer_heads[layer_idx][head_idx] = precision
        plan = DenseQuantPlan(
            model_id=artifact.get("model_id", args.candidate_model),
            layer_plans={
                idx: HeadQuantPlan(layer_idx=idx, assignments=heads, estimated_memory_ratio=1.0)
                for idx, heads in layer_heads.items()
            },
        )
        cand_model = apply_dense_quant(cand_model, plan)
    else:
        apply_precision_to_model(cand_model, args.candidate_precision, profile_path=args.profile)

    benchmark = resolve_benchmark(args.benchmark)
    result = compute_kl_divergence(
        ref_model,
        cand_model,
        tokenizer,
        dataset_name=benchmark["dataset_name"],
        dataset_config=benchmark["dataset_config"],
        split=benchmark["split"],
        max_length=args.max_length,
        max_samples=args.max_samples,
        text_field=benchmark["text_field"],
        streaming=benchmark.get("streaming", False),
    )
    payload = {
        "reference_model": args.reference_model,
        "candidate_model": args.candidate_model,
        "candidate_precision": args.candidate_precision,
        "benchmark": args.benchmark,
        **result,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
