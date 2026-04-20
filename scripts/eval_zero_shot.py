#!/usr/bin/env python3
"""Run the standard 5-task zero-shot evaluation suite via lm-eval.

Tasks: ARC-Easy, ARC-Challenge, PIQA, HellaSwag, WinoGrande.
Outputs per-task accuracy and the average.
"""

from __future__ import annotations

import argparse
import json


DEFAULT_TASKS = [
    "arc_easy",
    "arc_challenge",
    "piqa",
    "hellaswag",
    "winogrande",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="R-PGO zero-shot accuracy suite")
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", required=True, choices=["fp16", "int4", "rpgo", "rpgo_dense"])
    parser.add_argument("--profile")
    parser.add_argument("--quant-plan")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task example limit")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from lm_eval import simple_evaluate
    from rpgo.eval import apply_precision_to_model, load_model_and_tokenizer, apply_dense_quant

    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.precision == "rpgo_dense":
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
            model_id=artifact.get("model_id", args.model),
            layer_plans={
                idx: HeadQuantPlan(layer_idx=idx, assignments=heads, estimated_memory_ratio=1.0)
                for idx, heads in layer_heads.items()
            },
        )
        model = apply_dense_quant(model, plan)
    else:
        apply_precision_to_model(model, args.precision, profile_path=args.profile)

    results = simple_evaluate(
        model="hf",
        model_args={
            "pretrained": model,
            "tokenizer": tokenizer,
        },
        tasks=DEFAULT_TASKS,
        num_fewshot=0,
        limit=args.limit,
        log_samples=False,
    )

    task_scores = {}
    for task in DEFAULT_TASKS:
        task_result = results["results"][task]
        score = task_result.get("acc,none")
        if score is None:
            score = task_result.get("acc_norm,none")
        task_scores[task] = score

    avg = sum(v for v in task_scores.values() if v is not None) / max(1, len([v for v in task_scores.values() if v is not None]))
    payload = {
        "model": args.model,
        "precision": args.precision,
        "tasks": task_scores,
        "average_accuracy": avg,
        "limit": args.limit,
    }
    print(json.dumps(payload, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
