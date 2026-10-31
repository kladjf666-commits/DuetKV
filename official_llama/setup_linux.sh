#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${PHASEKV_NATIVE_VENV:-$PWD/.venv}"
command -v nvcc >/dev/null || { echo 'CUDA development toolkit (nvcc) required for native KIVI/FlashAttention compilation.'; exit 1; }
"$PYTHON_BIN" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12"'
"$PYTHON_BIN" -m venv "$VENV"
PY="$VENV/bin/python"
# Calling the venv Python by absolute path does not activate its bin directory.
# PyTorch finds Ninja through PATH; otherwise it silently falls back to distutils.
export PATH="$VENV/bin:$PATH"
"$PY" -m pip install --upgrade pip setuptools wheel packaging ninja psutil
# Keep PyPI available for build dependencies (flit_core etc.).
"$PY" -m pip install 'torch==2.5.1+cu121' --extra-index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install 'transformers==4.46.3' 'accelerate==1.1.1' 'numpy==1.26.4' 'sentencepiece==0.2.0' 'protobuf==5.28.3'
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
"$PY" -c 'import os, shutil; from torch.utils.cpp_extension import is_ninja_available; print("Ninja:", shutil.which("ninja"), "MAX_JOBS:", os.environ["MAX_JOBS"]); assert is_ninja_available(), "Ninja must be executable through PATH; refusing a serial build"'
"$PY" prepare.py --verify
"$PY" -m pip install -v 'flash-attn==2.7.4.post1' --no-build-isolation
(cd runtime/KIVI/quant && "$PY" -m pip install -v . --no-build-isolation --no-deps)
"$PY" -c 'import torch, flash_attn, kivi_gemv; assert torch.cuda.is_available(); print(torch.__version__, flash_attn.__version__, torch.cuda.get_device_name())'
mkdir -p installation
"$PY" -m pip freeze > installation/pip-freeze.txt
nvidia-smi > installation/nvidia-smi.txt
nvcc --version > installation/nvcc.txt
"$PY" tests/test_native_contract.py
echo "Installation complete. Next: $PY run.py --tiny --methods full kivi snapkv scope scope_slide scope_adaptive phasekv --out results/tiny_validation"
