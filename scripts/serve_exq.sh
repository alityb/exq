#!/bin/bash
# ExQ: profile → compile → serve with SGLang
#
# Usage: bash scripts/serve_exq.sh [MODEL] [OPTIONS]
#   MODEL   HuggingFace model ID (default: allenai/OLMoE-1B-7B-0924)
#   --port  Port for SGLang server (default: 30000)
#   --batch Max batch size (default: 8)
#
# Examples:
#   bash scripts/serve_exq.sh
#   bash scripts/serve_exq.sh Qwen/Qwen3-30B-A3B
#   bash scripts/serve_exq.sh allenai/OLMoE-1B-7B-0924 --port 30001

set -e

MODEL=${1:-"allenai/OLMoE-1B-7B-0924"}
# Shift model arg; remaining args pass to sglang
if [ $# -gt 0 ]; then shift; fi

SAFE_NAME=$(echo "$MODEL" | tr '/' '_' | tr '-' '_')
PROFILE="profiles/${SAFE_NAME}.json"
ARTIFACT="artifacts/${SAFE_NAME}.json"
PORT=30000

# Parse remaining args for --port
for arg in "$@"; do
    if [[ "$arg" == --port=* ]]; then
        PORT="${arg#--port=}"
    fi
done

echo "========================================"
echo " ExQ: profile → compile → serve"
echo "========================================"
echo " Model:    $MODEL"
echo " Profile:  $PROFILE"
echo " Artifact: $ARTIFACT"
echo " Port:     $PORT"
echo ""

# ── Step 1: Profile ───────────────────────────────────────────────────────────
if [ ! -f "$PROFILE" ]; then
    echo "[1/3] Profiling $MODEL (512 calibration samples)..."
    python scripts/profile_model.py \
        --model "$MODEL" \
        --samples 512 \
        --max-length 128 \
        --output "$PROFILE"
    echo "      Profile saved: $PROFILE"
else
    echo "[1/3] Profile exists: $PROFILE  (skipping)"
fi

# ── Step 2: Compile ───────────────────────────────────────────────────────────
if [ ! -f "$ARTIFACT" ]; then
    echo "[2/3] Compiling routing profile..."
    python scripts/compile_model.py \
        --profile "$PROFILE" \
        --run-auto \
        --output "$ARTIFACT"
    echo "      Artifact saved: $ARTIFACT"
else
    echo "[2/3] Artifact exists: $ARTIFACT  (skipping)"
fi

# ── Print diagnostic ──────────────────────────────────────────────────────────
echo ""
python - <<EOF
from rpgo.kernels.rpgo_artifact import load_rpgo_artifact, print_profile_summary
try:
    p = load_rpgo_artifact('${ARTIFACT}', '${PROFILE}')
    print("ExQ artifact summary:")
    print_profile_summary(p)
except Exception as e:
    print(f"  (diagnostic skipped: {e})")
EOF

# ── Step 3: Serve ─────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Launching SGLang server with ExQ INT4 backend..."
echo ""
echo "  The ExQ patch will be applied at model load time via"
echo "  EXQ_ARTIFACT env var. The server is otherwise standard SGLang."
echo ""
echo "  Once running, test with:"
echo "    curl http://localhost:${PORT}/v1/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"${MODEL}\", \"prompt\": \"Hello\", \"max_tokens\": 32}'"
echo ""

# Export artifact path so the SGLang startup hook can pick it up
export EXQ_ARTIFACT="$ARTIFACT"

# SGLang server with ExQ backend activated via startup hook
python -c "
import os, sys
sys.path.insert(0, '.')

artifact = os.environ.get('EXQ_ARTIFACT')
if artifact:
    print(f'ExQ: activating INT4 backend from {artifact}')
    from rpgo.runtime.sglang_backend import patch_sglang
    backend = patch_sglang(artifact)
    print(f'ExQ: SGLang patched ({backend.n_layers} layers covered)')
" && \
python -m sglang.launch_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    "$@"
