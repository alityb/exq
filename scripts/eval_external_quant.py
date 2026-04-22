#!/usr/bin/env python3
"""Evaluate an external quantized checkpoint against ExQ results.

This is used to compare against production quantizers like AWQ/GPTQ without
changing the core ExQ pipeline. It loads a pre-quantized checkpoint and
measures perplexity on the same benchmarks used for ExQ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate external quantized checkpoint")
    parser.add_argument("--model", required=True, help="Quantized checkpoint id/path")
    parser.add_argument("--tokenizer", required=True, help="Base tokenizer id/path")
    parser.add_argument("--provider", choices=["awq"], required=True)
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from exq.eval import compute_perplexity
from exq.model_utils import fix_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    if args.provider == "awq":
        from awq import AutoAWQForCausalLM

        wrapper = AutoAWQForCausalLM.from_quantized(
            args.model,
            device_map="auto",
            trust_remote_code=True,
        )
        model = wrapper.model
    else:
        raise ValueError(f"unsupported provider {args.provider}")

    model.eval()

    if args.dataset == "wikitext2":
        ds_name = "wikitext"
        ds_config = "wikitext-2-raw-v1"
        ds_split = "test"
    else:
        ds_name = "allenai/c4"
        ds_config = "en"
        ds_split = "validation"

    result = compute_perplexity(
        model,
        tokenizer,
        dataset_name=ds_name,
        dataset_config=ds_config,
        split=ds_split,
        max_length=512,
        stride=256,
        max_samples=args.max_samples,
        streaming=args.streaming,
    )

    payload = {
        "provider": args.provider,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "streaming": args.streaming,
        **result,
    }

    print(json.dumps(payload, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
