"""Perplexity evaluation for R-PGO quantization plans.

Measures WikiText-103 perplexity under different quantization configurations:
  1. Uniform INT4 (baseline -- what everyone else does)
  2. R-PGO frequency-stratified (BF16 hot + INT8 warm + INT4 cold)
  3. fp16 (quality ceiling, if memory permits)

The key comparison: R-PGO stratified should match fp16 quality at INT4 memory.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "wikitext2": {
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "text_field": "text",
        "streaming": False,
    },
    "c4": {
        "dataset_name": "allenai/c4",
        "dataset_config": "en",
        "split": "validation",
        "text_field": "text",
        "streaming": True,
    },
}


def resolve_benchmark(benchmark: str) -> dict[str, Any]:
    """Return dataset configuration for a named benchmark."""
    try:
        return BENCHMARK_SPECS[benchmark].copy()
    except KeyError as exc:
        supported = ", ".join(sorted(BENCHMARK_SPECS))
        raise ValueError(
            f"unsupported benchmark '{benchmark}'; expected one of: {supported}"
        ) from exc


def append_eval_result(
    log_path: str | Path,
    model_id: str,
    precision: str,
    benchmark: str,
    value: float,
) -> None:
    """Append a single benchmark result line to the eval log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{model_id}\t{precision}\t{benchmark}\t{value}\n")


def compute_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "test",
    max_length: int = 512,
    stride: int = 256,
    max_samples: int | None = None,
    device: str | None = None,
    text_field: str = "text",
    model_kwargs: dict[str, Any] | None = None,
    streaming: bool = False,
) -> dict[str, float]:
    """Compute perplexity of a model on a dataset using sliding window.

    Uses the standard sliding-window approach: for each position, the model
    sees max_length tokens of context and predicts the next token. Stride
    controls overlap between windows.

    Returns:
        {"perplexity": float, "loss": float, "n_tokens": int}
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)

    # Concatenate all text with newlines
    texts: list[str] = []
    for item in ds:
        text = item[text_field]
        if not text.strip():
            continue
        texts.append(text)
        if max_samples is not None and len(texts) >= max_samples:
            break
    full_text = "\n\n".join(texts)

    # Tokenize the full text
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings["input_ids"][0]
    seq_len = input_ids.size(0)

    logger.info(f"Evaluating perplexity: {seq_len} tokens, "
                f"max_length={max_length}, stride={stride}")

    nlls = []
    n_tokens = 0
    model.eval()
    model_kwargs = model_kwargs or {"use_cache": False}

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        target_begin = max(begin, begin + max_length - stride) if begin > 0 else 0

        input_chunk = input_ids[begin:end].unsqueeze(0).to(device)
        target_chunk = input_chunk.clone()
        # Mask tokens we don't want to compute loss on (the overlap region)
        target_chunk[0, :target_begin - begin] = -100

        with torch.no_grad():
            outputs = model(input_chunk, **model_kwargs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Shift logits and targets for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = target_chunk[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

            # Count non-masked tokens
            valid_tokens = (shift_labels != -100).sum().item()
            if valid_tokens > 0:
                nlls.append(loss.item())
                n_tokens += valid_tokens

        if end >= seq_len:
            break

        if len(nlls) % 20 == 0:
            running_ppl = math.exp(sum(nlls) / n_tokens) if n_tokens > 0 else float("inf")
            logger.info(f"  {n_tokens} tokens evaluated, running PPL={running_ppl:.2f}")

    avg_nll = sum(nlls) / n_tokens if n_tokens > 0 else float("inf")
    ppl = math.exp(avg_nll)

    logger.info(f"Final: PPL={ppl:.2f}, loss={avg_nll:.4f}, tokens={n_tokens}")

    return {
        "perplexity": ppl,
        "loss": avg_nll,
        "n_tokens": n_tokens,
    }


def compute_kl_divergence(
    reference_model,
    candidate_model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_length: int = 256,
    max_samples: int | None = 200,
    device: str | None = None,
    text_field: str = "text",
    streaming: bool = False,
) -> dict[str, float]:
    """Compute KL divergence between reference and candidate logits.

    Returns token-level KL statistics to capture both average and outlier drift.
    """
    from datasets import load_dataset

    if device is None:
        device = next(reference_model.parameters()).device

    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)

    kl_values: list[float] = []
    n_samples = 0
    reference_model.eval()
    candidate_model.eval()

    for item in ds:
        text = item[text_field]
        if not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            ref_logits = reference_model(input_ids, use_cache=False).logits[:, :-1, :]
            cand_logits = candidate_model(input_ids, use_cache=False).logits[:, :-1, :]

        p = F.softmax(ref_logits.float(), dim=-1)
        q_log = F.log_softmax(cand_logits.float(), dim=-1)
        kl = F.kl_div(q_log, p, reduction="none").sum(dim=-1)
        kl_values.extend(kl.reshape(-1).cpu().tolist())

        n_samples += 1
        if max_samples is not None and n_samples >= max_samples:
            break

    if not kl_values:
        return {"mean": float("nan"), "max": float("nan"), "p99": float("nan"), "count": 0}

    kl_values.sort()
    p99_idx = min(len(kl_values) - 1, int(0.99 * len(kl_values)))
    return {
        "mean": sum(kl_values) / len(kl_values),
        "max": max(kl_values),
        "p99": kl_values[p99_idx],
        "count": len(kl_values),
    }


def run_perplexity_comparison(
    model_id: str,
    quant_plan: dict[tuple[int, int], str] | None = None,
    max_length: int = 512,
    stride: int = 256,
    max_samples: int | None = 200,
) -> dict[str, Any]:
    """Run perplexity comparison between uniform INT4 and R-PGO stratified.

    This is the Milestone 3 benchmark. Loads the model with CPU offloading
    and measures perplexity on WikiText-103 test set.

    For now, we measure perplexity of the fp16+offload model as the baseline.
    The R-PGO comparison requires actually applying mixed quantization to
    individual expert weights, which needs the runtime shim (Milestone 4+).

    What we CAN measure right now:
    - fp16 baseline perplexity (quality ceiling)
    - This establishes the target the R-PGO plan must match
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {model_id} for perplexity eval...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        offload_folder="/tmp/rpgo_offload",
        trust_remote_code=True,
        dtype=torch.float16,
    )

    results = compute_perplexity(
        model,
        tokenizer,
        max_length=max_length,
        stride=stride,
        max_samples=max_samples,
    )

    return {
        "model_id": model_id,
        "config": "fp16_offload",
        **results,
    }
