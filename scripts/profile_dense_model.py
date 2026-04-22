#!/usr/bin/env python3
"""CLI: Collect attention head profile from a dense transformer model."""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from exq.hf_compat import patch_transformers_remote_code_compat
from exq.model_utils import fix_tokenizer
from exq.profiler.attention_profiler import AttentionProfiler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _model_device(model) -> torch.device:
    """Return a usable device for model inputs under simple device_map setups."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExQ: Collect attention head profile (dense models)"
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", default="profiles/dense_profile.json")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Install: pip install transformers datasets accelerate")
        sys.exit(1)

    patch_transformers_remote_code_compat()
    logging.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    profiler = AttentionProfiler(model, model_id=args.model)

    logging.info("Loading dataset: %s/%s", args.dataset, args.dataset_config)
    dataset = load_dataset(args.dataset, args.dataset_config, split="train")
    texts = [text for text in dataset["text"] if text.strip()]
    device = _model_device(model)

    n_collected = 0
    with torch.no_grad():
        for text in texts:
            if n_collected >= args.samples:
                break
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            if inputs["input_ids"].shape[1] < 8:
                continue
            model(**inputs, use_cache=False)
            n_collected += 1
            if n_collected % 50 == 0:
                logging.info("  %s/%s samples", n_collected, args.samples)

    profile = profiler.build_profile(calibration_samples=n_collected)
    for warning in profile.validate():
        logging.warning(warning)

    summary = profile.summary()
    logging.info(
        "Profile: %s layers, %s total heads, avg entropy: %.3f, %s token activations",
        summary["n_layers"],
        summary["total_heads"],
        summary["avg_normalized_entropy"],
        f"{profile.calibration_tokens:,}",
    )

    profile.save(args.output)
    logging.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
