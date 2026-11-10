#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -z "${PYTHON:-}" && -x .venv-4090d/bin/python ]]; then
    export PYTHON="$PWD/.venv-4090d/bin/python"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
exec "${PYTHON:-python3}" scripts/quick_phasekv_long_output.py "$@"
