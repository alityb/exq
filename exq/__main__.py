"""
exq CLI — profile, compile, serve.

Three commands, one tool:

    exq profile --model Qwen/Qwen3-30B-A3B
    exq compile --profile profiles/Qwen_Qwen3-30B-A3B.json
    exq serve   --model Qwen/Qwen3-30B-A3B --port 30000

`exq serve` applies the ExQ INT4 patch to SGLang's MoE kernel and
then starts an OpenAI-compatible server.  The user never touches SGLang
directly.

Invoked via:
    exq <subcommand> [args...]      # installed entry point
    python -m exq <subcommand>     # direct module execution
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _default_profile_path(model_id: str) -> str:
    safe = model_id.replace("/", "_")
    return f"profiles/{safe}.json"


def _default_artifact_path(model_id: str) -> str:
    safe = model_id.replace("/", "_")
    return f"artifacts/{safe}.json"


# ── subcommand: profile ───────────────────────────────────────────────────────

def _cmd_profile(argv: list[str]) -> None:
    """Collect a routing profile from a HuggingFace MoE model."""
    # Delegate to the existing profile_model.py main() logic, but override
    # sys.argv so its argparse sees our arguments.
    import importlib.util, runpy
    script = Path(__file__).parent.parent / "scripts" / "profile_model.py"

    if not script.exists():
        # Installed package: look for the entry function directly
        from exq.profiler.calibration_runner import _profile_main as profile_main
        sys.argv = ["exq profile"] + argv
        profile_main()
        return

    # Dev install: run the script's main() with our argv
    sys.argv = ["exq profile"] + argv
    spec = importlib.util.spec_from_file_location("profile_model", script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def _cmd_compile(argv: list[str]) -> None:
    """Compile a routing profile into an ExQ artifact."""
    import importlib.util
    script = Path(__file__).parent.parent / "scripts" / "compile_model.py"

    if not script.exists():
        from exq.compiler._compile_main import compile_main
        sys.argv = ["exq compile"] + argv
        compile_main()
        return

    sys.argv = ["exq compile"] + argv
    spec = importlib.util.spec_from_file_location("compile_model", script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


# ── subcommand: serve ─────────────────────────────────────────────────────────

def _cmd_serve(argv: list[str]) -> None:
    """
    Serve a MoE model with ExQ's INT4 kernel via SGLang.

    ExQ profiles the model (if no profile exists), compiles an artifact
    (if none exists), applies the INT4 patch to SGLang's MoE dispatch, then
    starts an OpenAI-compatible server.

    All flags after --model / --port / --artifact are forwarded to SGLang's
    launch_server verbatim, so any sglang serve flag works here too.

    Example
    -------
        exq serve --model Qwen/Qwen3-30B-A3B
        exq serve --model Qwen/Qwen3-30B-A3B --port 30000 --mem-fraction-static 0.85
        exq serve --model Qwen/Qwen3-30B-A3B --artifact artifacts/custom.json
    """
    p = argparse.ArgumentParser(
        prog="exq serve",
        description=_cmd_serve.__doc__,
        # Pass unknown args through to SGLang
        add_help=True,
    )
    p.add_argument(
        "--model", required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen3-30B-A3B)"
    )
    p.add_argument(
        "--port", type=int, default=30000,
        help="Server port (default: 30000)"
    )
    p.add_argument(
        "--host", default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    p.add_argument(
        "--artifact",
        help="Path to compiled ExQ artifact JSON. "
             "If omitted, uses artifacts/<model_safe_name>.json, "
             "profiling and compiling first if needed."
    )
    p.add_argument(
        "--profile-samples", type=int, default=512,
        help="Calibration samples if profiling is needed (default: 512)"
    )
    p.add_argument(
        "--skip-exq", action="store_true",
        help="Start SGLang without the ExQ patch (useful for A/B timing)"
    )

    args, sglang_extra = p.parse_known_args(argv)

    model_id   = args.model
    profile_path  = Path(_default_profile_path(model_id))
    artifact_path = Path(args.artifact) if args.artifact else Path(_default_artifact_path(model_id))

    # ── Step 1: profile if needed ─────────────────────────────────────────────
    if not args.skip_exq:
        if not artifact_path.exists():
            if not profile_path.exists():
                print(f"[exq serve] No profile found at {profile_path}")
                print(f"[exq serve] Profiling {model_id} ({args.profile_samples} samples)...")
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                _cmd_profile([
                    "--model", model_id,
                    "--samples", str(args.profile_samples),
                    "--output", str(profile_path),
                ])
            else:
                print(f"[exq serve] Profile: {profile_path}  (exists, skipping)")

            # ── Step 2: compile if needed ─────────────────────────────────────
            print(f"[exq serve] Compiling routing profile → artifact...")
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            _cmd_compile([
                "--profile", str(profile_path),
                "--run-auto",
                "--output", str(artifact_path),
            ])
        else:
            print(f"[exq serve] Artifact: {artifact_path}  (exists, skipping)")

        # ── Step 3: apply ExQ patch ─────────────────────────────────────────
        print(f"[exq serve] Applying ExQ INT4 patch from {artifact_path}...")
        try:
            from exq.runtime.sglang_backend import patch_sglang
            backend = patch_sglang(str(artifact_path))
            print(f"[exq serve] Patch applied — {backend.n_layers} layers covered")
        except ImportError as exc:
            print(f"[exq serve] WARNING: SGLang not installed — serving without ExQ patch")
            print(f"             Install with: pip install sglang")
            print(f"             Error: {exc}")
    else:
        print("[exq serve] --skip-exq set: starting SGLang without ExQ patch")

    # ── Step 4: launch SGLang server ──────────────────────────────────────────
    # Build the argv list that SGLang's prepare_server_args expects.
    # We pass --model, --port, --host, plus any extra flags the user supplied.
    sglang_argv = [
        "--model", model_id,
        "--port", str(args.port),
        "--host", args.host,
    ] + sglang_extra

    print(f"[exq serve] Starting SGLang server on {args.host}:{args.port}")
    print(f"             Model: {model_id}")
    if sglang_extra:
        print(f"             Extra SGLang flags: {' '.join(sglang_extra)}")
    print()

    try:
        from sglang.srt.server_args import prepare_server_args
        from sglang.launch_server import run_server
    except ImportError as exc:
        print(f"ERROR: SGLang is not installed.")
        print(f"  Install: pip install sglang")
        print(f"  Details: {exc}")
        sys.exit(1)

    server_args = prepare_server_args(sglang_argv)
    run_server(server_args)


# ── top-level dispatcher ──────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the `exq` command."""
    # Minimal top-level parser: only intercepts --version and the subcommand name.
    # Everything else (including --help) is passed through to the subcommand so
    # `exq serve --help` shows serve's own help, not the top-level help.
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: exq <command> [options]")
        print()
        print("Commands:")
        print("  profile   Collect a routing profile from a MoE model")
        print("  compile   Compile a routing profile into an ExQ artifact")
        print("  serve     Serve a model with ExQ's INT4 kernel via SGLang")
        print()
        print("Run `exq <command> --help` for per-command options.")
        print()
        print("Example (three-command workflow):")
        print("  exq profile --model Qwen/Qwen3-30B-A3B")
        print("  exq compile --profile profiles/Qwen_Qwen3-30B-A3B.json")
        print("  exq serve   --model Qwen/Qwen3-30B-A3B --port 30000")
        print()
        print("Example (one-command):")
        print("  exq serve --model Qwen/Qwen3-30B-A3B")
        print("  (profiles and compiles automatically if no artifact exists)")
        sys.exit(0)

    if sys.argv[1] in ("--version", "-V"):
        from exq import __version__
        print(f"exq {__version__}")
        sys.exit(0)

    command = sys.argv[1]
    remainder = sys.argv[2:]

    dispatch = {
        "profile": _cmd_profile,
        "compile": _cmd_compile,
        "serve":   _cmd_serve,
    }

    if command not in dispatch:
        print(f"exq: unknown command '{command}'")
        print("Run `exq --help` for available commands.")
        sys.exit(1)

    dispatch[command](remainder)


if __name__ == "__main__":
    main()
