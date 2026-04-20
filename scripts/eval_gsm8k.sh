#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/../.venv/bin/python}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/eval_gsm8k.py" "$@"
