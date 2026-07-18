#!/usr/bin/env bash
# Launch the J-Space Explorer web server.
#
#   ./run.sh                        # serve on 127.0.0.1:8000, load model eagerly
#   JSPACE_MODEL_PATH=/models/foo ./run.sh
#   JSPACE_PORT=8090 ./run.sh
#
# Environment:
#   JSPACE_ENV         conda env name (default: jspace)
#   JSPACE_MODEL_PATH  path to a local safetensors model directory
#   JSPACE_HOST        bind host (default: 127.0.0.1)
#   JSPACE_PORT        bind port (default: 8000)
#   JSPACE_LENS_CACHE  where fitted Jacobian operators are cached
set -euo pipefail

ENV_NAME="${JSPACE_ENV:-jspace}"
HOST="${JSPACE_HOST:-127.0.0.1}"
PORT="${JSPACE_PORT:-8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if command -v conda &>/dev/null; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u
  conda activate "${ENV_NAME}"
  set -u
elif [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "error: no conda and no .venv/bin/activate — run setup.sh or create a venv first" >&2
  exit 1
fi

# Avoid a deprecated/warning-noisy HF transfer path.
export HF_HUB_ENABLE_HF_TRANSFER=0
# Reduce CUDA fragmentation for large models + autograd.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec jspace serve --host "${HOST}" --port "${PORT}" --eager
