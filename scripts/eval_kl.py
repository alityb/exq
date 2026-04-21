#!/usr/bin/env python3
"""Compute KL divergence between a reference fp16 model and a quantized model."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="R-PGO KL divergence evaluation")
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-precision", required=True, choices=["fp16", "int4", "rpgo", "rpgo_dense", "awq_controlled"])
    parser.add_argument("--profile")
    parser.add_argument("--quant-plan")
    parser.add_argument("--awq-calib-samples", type=int, default=64)
    parser.add_argument("--awq-calib-seq-len", type=int, default=512)
    parser.add_argument("--awq-group-size", type=int, default=128)
    parser.add_argument("--benchmark", choices=["wikitext2", "c4"], required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    from rpgo.eval import compute_kl_divergence, load_model_and_tokenizer, load_model_variant, resolve_benchmark

    ref_model, tokenizer = load_model_and_tokenizer(args.reference_model)
    cand_model, _ = load_model_variant(
        args.candidate_model,
        args.candidate_precision,
        profile=args.profile,
        quant_plan=args.quant_plan,
        awq_calib_samples=args.awq_calib_samples,
        awq_calib_seq_len=args.awq_calib_seq_len,
        awq_group_size=args.awq_group_size,
    )

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
