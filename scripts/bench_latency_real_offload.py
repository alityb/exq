#!/usr/bin/env python3
"""Real CPU-offload latency benchmark using Accelerate offloaded layers.

This benchmark forces actual CPU offloading by loading OLMoE in fp32 with a
restricted GPU memory budget. It then compares:

  A) Baseline auto-offloaded model
  B) Runtime predictor hook overhead on the offloaded model
  C) Static layer-prefetch experiment that pre-triggers the next CPU layer's
     Accelerate offload hook while the current layer computes

All metrics are measured with real `generate()` calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.bench_latency import RuntimePrefetchHook


class LayerPrefetchPatcher:
    """Prefetch next offloaded layer by calling its Accelerate hook early."""

    def __init__(self, model):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.handles = []
        self.pending: dict[int, Future] = {}
        self.prefetch_attempts = 0
        self.prefetch_waits = 0

        self.layers = list(model.model.layers)
        self.layer_devices = {
            i: model.hf_device_map.get(f"model.layers.{i}")
            for i in range(len(self.layers))
        }

    def patch(self) -> int:
        patched = 0
        for idx, layer in enumerate(self.layers):
            # Wait for any pending prefetch for this layer right before execution
            self.handles.append(layer.register_forward_pre_hook(self._make_wait_hook(idx), with_kwargs=True))
            # Trigger prefetch for the next offloaded layer after current layer starts
            if idx + 1 < len(self.layers) and self.layer_devices.get(idx + 1) == "cpu":
                self.handles.append(layer.register_forward_hook(self._make_prefetch_hook(idx + 1)))
                patched += 1
        return patched

    def _make_wait_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            future = self.pending.pop(layer_idx, None)
            if future is not None:
                self.prefetch_waits += 1
                future.result()
            return args, kwargs
        return hook

    def _make_prefetch_hook(self, next_idx: int):
        def hook(module, inputs, output):
            next_layer = self.layers[next_idx]
            hf_hook = getattr(next_layer, "_hf_hook", None)
            if hf_hook is None or not hasattr(hf_hook, "pre_forward"):
                return
            # Launch CPU->GPU weight load for the next layer in a background thread.
            # The dummy tensor is only used so the hook has a device/context to work with.
            dummy = output[0] if isinstance(output, tuple) else output
            if not isinstance(dummy, torch.Tensor):
                return
            self.prefetch_attempts += 1
            self.pending[next_idx] = self.executor.submit(hf_hook.pre_forward, next_layer, dummy)
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.executor.shutdown(wait=True)


def _summary(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    n = len(values)
    return {
        "p50": values[n // 2],
        "p95": values[min(n - 1, int(0.95 * (n - 1)))],
        "p99": values[min(n - 1, int(0.99 * (n - 1)))],
        "mean": statistics.mean(values),
    }


def _measure(model, tokenizer, prompt: str, batch_size: int, n_tokens: int, n_runs: int, warmup: int) -> dict:
    prompts = [prompt for _ in range(batch_size)]
    device = next(iter(p.device for p in model.parameters() if p.device.type == "cuda"), torch.device("cuda"))
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # TTFT
    ttft = []
    with torch.no_grad():
        for i in range(warmup + n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()
            if i >= warmup:
                ttft.append((time.perf_counter() - t0) * 1000)

    # TPOT / throughput
    tpot = []
    throughput = []
    with torch.no_grad():
        for i in range(warmup + n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            generated = out.shape[1] - input_len
            if i >= warmup:
                tpot.append(elapsed / max(generated, 1) * 1000)
                throughput.append((generated * batch_size) / max(elapsed, 1e-9))

    return {
        "ttft_ms": _summary(ttft),
        "tpot_ms": _summary(tpot),
        "throughput_toks_per_s": _summary(throughput),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real CPU-offload benchmark")
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--gpu-memory", default="18GiB")
    parser.add_argument("--cpu-memory", default="48GiB")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-tokens", type=int, default=16)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prompt", default="The following is a detailed analysis of")
    parser.add_argument("--output", default="results/real_offload_metrics.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder="/tmp/rpgo_real_offload",
        trust_remote_code=True,
    )
    model.eval()

    device_counts = {}
    for dev in model.hf_device_map.values():
        device_counts[str(dev)] = device_counts.get(str(dev), 0) + 1
    print(f"device map counts: {device_counts}")

    print("\n=== A: Baseline auto-offload ===")
    baseline = _measure(model, tokenizer, args.prompt, args.batch_size, args.n_tokens, args.n_runs, args.warmup)
    print(f"TTFT p50: {baseline['ttft_ms']['p50']:.1f}ms")
    print(f"TPOT p50: {baseline['tpot_ms']['p50']:.1f}ms")

    print("\n=== B: Runtime predictor on offloaded model ===")
    predictor = RuntimePrefetchHook(args.profile, model)
    runtime_pred = _measure(model, tokenizer, args.prompt, args.batch_size, args.n_tokens, args.n_runs, args.warmup)
    predictor.remove()
    print(f"TTFT p50: {runtime_pred['ttft_ms']['p50']:.1f}ms")
    print(f"TPOT p50: {runtime_pred['tpot_ms']['p50']:.1f}ms")

    print("\n=== C: Static next-layer prefetch on real offload ===")
    patcher = LayerPrefetchPatcher(model)
    patched = patcher.patch()
    static_prefetch = _measure(model, tokenizer, args.prompt, args.batch_size, args.n_tokens, args.n_runs, args.warmup)
    patcher.remove()
    print(f"TTFT p50: {static_prefetch['ttft_ms']['p50']:.1f}ms")
    print(f"TPOT p50: {static_prefetch['tpot_ms']['p50']:.1f}ms")

    payload = {
        "model_id": args.model,
        "gpu_memory": args.gpu_memory,
        "cpu_memory": args.cpu_memory,
        "device_map_counts": device_counts,
        "patched_layers": patched,
        "prefetch_attempts": patcher.prefetch_attempts,
        "prefetch_waits": patcher.prefetch_waits,
        "baseline": baseline,
        "runtime_predictor": runtime_pred,
        "static_prefetch": static_prefetch,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
