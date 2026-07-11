#!/usr/bin/env bash
# One-shot setup for the J-Space Explorer.
#
#   ./setup.sh                      # create conda env 'jspace' + install
#   JSPACE_ENV=myenv ./setup.sh     # use a different env name
#
# Requires conda (miniconda/anaconda) on PATH.
set -euo pipefail

ENV_NAME="${JSPACE_ENV:-jspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not found on PATH. Install Miniconda/Anaconda first:" >&2
  echo "  https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "==> Updating existing conda env '${ENV_NAME}'"
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo "==> Creating conda env '${ENV_NAME}' from environment.yml"
  # environment.yml pins name: jspace; honor a custom name if requested.
  if [ "${ENV_NAME}" = "jspace" ]; then
    conda env create -f environment.yml
  else
    conda env create -n "${ENV_NAME}" -f environment.yml
  fi
fi

echo "==> Installing the jspace package (editable)"
conda run -n "${ENV_NAME}" pip install -e .

cat <<EOF

✅ Setup complete.

Next:
  conda activate ${ENV_NAME}
  export JSPACE_MODEL_PATH=/path/to/your/model   # a local safetensors model dir
  ./run.sh                                        # or: jspace serve --eager

Then open http://127.0.0.1:8000  (a guided tour starts automatically).
EOF
