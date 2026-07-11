#!/usr/bin/env bash
# Launch the J-Space Explorer using your personal, gitignored config.
#
#   1) cp jspace.local.env.example jspace.local.env   # first time only
#   2) edit jspace.local.env for your machine
#   3) ./serve-local.sh
#
# You can point at a different config file:  JSPACE_CONFIG=other.env ./serve-local.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CONFIG="${JSPACE_CONFIG:-jspace.local.env}"
if [ ! -f "$CONFIG" ]; then
  echo "error: $CONFIG not found." >&2
  echo "Create it from the template:" >&2
  echo "    cp jspace.local.env.example jspace.local.env" >&2
  echo "then edit it (at least set JSPACE_MODEL_PATH)." >&2
  exit 1
fi

# Load the config (allexport so every assignment becomes an env var).
set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

ENV_NAME="${JSPACE_ENV:-jspace}"
HOST="${JSPACE_HOST:-127.0.0.1}"
PORT="${JSPACE_PORT:-8000}"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found on PATH." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda's activate hooks reference unbound vars; disable nounset around them.
set +u
conda activate "$ENV_NAME"
set -u

EAGER_FLAG=""
case "${JSPACE_EAGER:-1}" in
  1|true|yes|on) EAGER_FLAG="--eager" ;;
esac

echo "Model : ${JSPACE_MODEL_PATH:-<default>}"
echo "Serving on http://${HOST}:${PORT}  (env: ${ENV_NAME})"
exec jspace serve --host "$HOST" --port "$PORT" $EAGER_FLAG
