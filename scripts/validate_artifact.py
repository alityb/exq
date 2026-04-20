#!/usr/bin/env python3
"""Validate a compiled R-PGO artifact and generated kernel module.

Checks:
1. Artifact schema sanity
2. Generated kernel module parses/imports
3. Runtime patching can be constructed for the target model
4. Deterministic codegen: emitting twice yields identical bytes
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import py_compile
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate compiled artifact and codegen")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model")
    args = parser.parse_args()

    from rpgo.codegen import emit_prefetch_kernels

    artifact_path = Path(args.artifact)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    required = ["model_id", "quant_assignments", "layout_placements", "specialization_decisions", "prefetch_entry_count"]
    missing = [k for k in required if k not in artifact]
    if missing:
        raise SystemExit(f"artifact missing required keys: {missing}")

    print("artifact schema: OK")

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        p1 = emit_prefetch_kernels(artifact_path, tmp1)
        p2 = emit_prefetch_kernels(artifact_path, tmp2)

        py_compile.compile(str(p1), doraise=True)
        print("generated module syntax: OK")

        mod_name = "rpgo_generated_validation"
        spec = importlib.util.spec_from_file_location(mod_name, p1)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        print("generated module import: OK")

        bytes1 = Path(p1).read_bytes()
        bytes2 = Path(p2).read_bytes()
        if bytes1 != bytes2:
            raise SystemExit("codegen is not deterministic")
        print("codegen determinism: OK")

    if args.model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from rpgo.runtime import CompiledInference
        import torch

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        engine = CompiledInference.from_artifact(artifact_path, model, tokenizer)
        print(f"runtime patching: OK ({len(engine.prefetch_table)} layers with schedules)")


if __name__ == "__main__":
    main()
