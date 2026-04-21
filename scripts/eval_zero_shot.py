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
    parser.add_argument("--precision", required=True, choices=["fp16", "int4", "rpgo", "rpgo_dense", "awq_controlled"])
    parser.add_argument("--profile")
    parser.add_argument("--quant-plan")
    parser.add_argument("--awq-calib-samples", type=int, default=64)
    parser.add_argument("--awq-calib-seq-len", type=int, default=512)
    parser.add_argument("--awq-group-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-task example limit")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from rpgo.eval import load_model_variant

    model, tokenizer = load_model_variant(
        args.model,
        args.precision,
        profile=args.profile,
        quant_plan=args.quant_plan,
        awq_calib_samples=args.awq_calib_samples,
        awq_calib_seq_len=args.awq_calib_seq_len,
        awq_group_size=args.awq_group_size,
    )

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        trust_remote_code=True,
        device="cuda",
    )

    results = simple_evaluate(
        model=lm,
        model_args=None,
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
