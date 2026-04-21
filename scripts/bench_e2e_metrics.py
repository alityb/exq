#!/usr/bin/env python3
"""End-to-end inference benchmark for R-PGO.

Measures real generation metrics on a fixed GPU using actual `generate()` calls:
  - TTFT: time to first token
  - TPOT: time per output token
  - Throughput: output tokens/sec

Compares three conditions:
  A) Baseline model
  B) Runtime predictor hook (per-token overhead)
  C) R-PGO compiled static runtime (artifact-patched model)

This avoids simulation language: every metric comes from real inference on a
real model with the actual Hugging Face generation stack.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpgo.runtime import CompiledInference
from scripts.bench_latency import RuntimePrefetchHook


def _prepare_inputs(tokenizer, prompt: str, batch_size: int, device: torch.device):
    prompts = [prompt for _ in range(batch_size)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in inputs.items()}


def _measure_ttft(model, inputs: dict[str, torch.Tensor], n_runs: int, warmup: int) -> list[float]:
    times: list[float] = []
    with torch.no_grad():
        for i in range(warmup + n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= warmup:
                times.append(elapsed)
    return times


def _measure_generation(model, inputs: dict[str, torch.Tensor], n_tokens: int, n_runs: int, warmup: int) -> tuple[list[float], list[float]]:
    """Return per-run TPOT ms/token and throughput tokens/sec."""
    tpot_values: list[float] = []
    throughput_values: list[float] = []
    input_len = inputs["input_ids"].shape[1]
    batch_size = inputs["input_ids"].shape[0]
    with torch.no_grad():
        for i in range(warmup + n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            generated = out.shape[1] - input_len
            total_generated = generated * batch_size
            if i >= warmup:
                tpot_values.append(elapsed / max(generated, 1) * 1000)
                throughput_values.append(total_generated / max(elapsed, 1e-9))
    return tpot_values, throughput_values


def _summary(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    n = len(values)
    if n == 0:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "mean": float("nan")}
    return {
        "p50": values[n // 2],
        "p95": values[min(n - 1, int(0.95 * (n - 1)))],
        "p99": values[min(n - 1, int(0.99 * (n - 1)))],
        "mean": statistics.mean(values),
    }


def _benchmark_condition(model, inputs, n_tokens: int, n_runs: int, warmup: int) -> dict[str, dict[str, float]]:
    ttft = _measure_ttft(model, inputs, n_runs=n_runs, warmup=warmup)
    tpot, throughput = _measure_generation(model, inputs, n_tokens=n_tokens, n_runs=n_runs, warmup=warmup)
    return {
        "ttft_ms": _summary(ttft),
        "tpot_ms": _summary(tpot),
        "throughput_toks_per_s": _summary(throughput),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R-PGO end-to-end metrics benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prompt", default="The following is a detailed analysis of")
    parser.add_argument("--output", default="results/e2e_metrics.json")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    inputs = _prepare_inputs(tokenizer, args.prompt, args.batch_size, device)

    print(f"GPU: {gpu_name}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output tokens: {args.n_tokens}")

    # A) Baseline
    print("\n=== A: Baseline ===")
    baseline = _benchmark_condition(model, inputs, args.n_tokens, args.n_runs, args.warmup)
    print(f"TTFT p50: {baseline['ttft_ms']['p50']:.1f}ms")
    print(f"TPOT p50: {baseline['tpot_ms']['p50']:.1f}ms/token")
    print(f"Throughput p50: {baseline['throughput_toks_per_s']['p50']:.1f} tok/s")

    # B) Runtime predictor
    print("\n=== B: Runtime predictor ===")
    predictor = RuntimePrefetchHook(args.profile, model)
    runtime_pred = _benchmark_condition(model, inputs, args.n_tokens, args.n_runs, args.warmup)
    predictor.remove()
    print(f"TTFT p50: {runtime_pred['ttft_ms']['p50']:.1f}ms")
    print(f"TPOT p50: {runtime_pred['tpot_ms']['p50']:.1f}ms/token")
    print(f"Throughput p50: {runtime_pred['throughput_toks_per_s']['p50']:.1f} tok/s")

    # C) Compiled static
    print("\n=== C: R-PGO compiled static ===")
    engine = CompiledInference.from_artifact(args.artifact, model, tokenizer)
    compiled = _benchmark_condition(model, inputs, args.n_tokens, args.n_runs, args.warmup)
    print(f"TTFT p50: {compiled['ttft_ms']['p50']:.1f}ms")
    print(f"TPOT p50: {compiled['tpot_ms']['p50']:.1f}ms/token")
    print(f"Throughput p50: {compiled['throughput_toks_per_s']['p50']:.1f} tok/s")

    results = {
        "gpu": gpu_name,
        "model_id": args.model,
        "batch_size": args.batch_size,
        "n_tokens": args.n_tokens,
        "n_runs": args.n_runs,
        "baseline": baseline,
        "runtime_predictor": runtime_pred,
        "rpgo_compiled": compiled,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
