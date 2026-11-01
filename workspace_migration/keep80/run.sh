#!/usr/bin/env bash
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd -P)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/root/data/phasekv-triton-sm89
exec /root/data/phasekv-ws/venv/bin/python -u "$dir/campaign.py" "$@"
