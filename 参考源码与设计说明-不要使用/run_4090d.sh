#!/usr/bin/env bash
# Run from anywhere; Python remains in the foreground so Ctrl-C stops the run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-300}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
export PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-10}"
export PIP_PREFER_BINARY="${PIP_PREFER_BINARY:-1}"
if [[ "${1:-}" == "--setup" ]]; then
    shift
    "${PYTHON:-python3}" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10, 3.11 or 3.12 for PyTorch 2.5.1"'
    if [[ ! -x .venv-4090d/bin/python ]]; then
        "${PYTHON:-python3}" -m venv .venv-4090d
    fi
    export PYTHON="$PWD/.venv-4090d/bin/python"
    "$PYTHON" -m pip install --upgrade pip \
        --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}"
    # General dependencies/build backends come from PyPI; pin the CUDA wheel explicitly.
    "$PYTHON" -m pip install 'torch==2.5.1+cu121' --prefer-binary \
        --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
        --extra-index-url https://download.pytorch.org/whl/cu121
    set -- --install-deps "$@"
elif [[ -z "${PYTHON:-}" && -x .venv-4090d/bin/python ]]; then
    export PYTHON="$PWD/.venv-4090d/bin/python"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
exec "${PYTHON:-python3}" scripts/benchmark_4090d.py "$@"
