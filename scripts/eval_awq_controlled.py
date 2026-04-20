#!/usr/bin/env python3
"""Run a controlled in-process AWQ baseline from the same base checkpoint.

This script quantizes a base model with AutoAWQ in the current environment,
then evaluates perplexity on the same benchmarks used for R-PGO. Unlike the
external-checkpoint comparison, this is a same-checkpoint baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _resolve_dataset(name: str) -> tuple[str, str, str, bool]:
    if name == "wikitext2":
        return "wikitext", "wikitext-2-raw-v1", "test", False
    if name == "c4":
        return "allenai/c4", "en", "validation", True
    raise ValueError(f"unsupported dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled AWQ baseline evaluation")
    parser.add_argument("--model", required=True, help="Base model id")
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-seq-len", type=int, default=512)
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--duo-scaling", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--eval-device", default=None, help="Force final evaluation onto one device (e.g. cuda)")
    parser.add_argument("--save-dir", default=None, help="Optional path to save the quantized model")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    from rpgo.eval import compute_perplexity

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    wrapper = AutoAWQForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }

    wrapper.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data="pileval",
        max_calib_samples=args.calib_samples,
        max_calib_seq_len=args.calib_seq_len,
        n_parallel_calib_samples=8,
        duo_scaling=args.duo_scaling,
        apply_clip=not args.no_clip,
    )

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        wrapper.save_quantized(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

    eval_device = args.eval_device
    if eval_device is None:
        eval_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model = wrapper.model
    model = model.to(eval_device)
    model.eval()

    ds_name, ds_config, ds_split, streaming = _resolve_dataset(args.dataset)
    result = compute_perplexity(
        model,
        tokenizer,
        dataset_name=ds_name,
        dataset_config=ds_config,
        split=ds_split,
        max_length=512,
        stride=256,
        max_samples=args.max_samples,
        streaming=streaming,
        model_kwargs={"use_cache": False},
    )

    payload = {
        "provider": "awq_controlled",
        "model": args.model,
        "dataset": args.dataset,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "w_bit": args.w_bit,
        "group_size": args.group_size,
        "duo_scaling": args.duo_scaling,
        "apply_clip": not args.no_clip,
        **result,
    }
    print(json.dumps(payload, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
