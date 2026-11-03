#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
volume=$(dirname "$here")
py=${PHASEKV_PYTHON:-}
model=${PHASEKV_MODEL:-}
if [[ -z "$py" ]]; then
    for envdir in "${PHASEKV_NATIVE_VENV:-}" "$volume/phasekv-ws/venv" "$volume/phasekv-native-venv" /root/data/phasekv-ws/venv /data/phasekv-native-venv "${VIRTUAL_ENV:-}"; do
        if [[ -n "$envdir" && -x "$envdir/bin/python" ]]; then py="$envdir/bin/python"; break; fi
    done
    py=${py:-python3}
fi
if [[ -z "$model" ]]; then
    for parent in "$volume" /root/data /data; do
        candidate="$parent/Llama-3.1-8B-Instruct/snapshots/master"
        if [[ -f "$candidate/model.safetensors.index.json" ]]; then model="$candidate"; break; fi
    done
fi
if [[ -z "$model" ]]; then echo 'Set PHASEKV_MODEL to the directory containing the Llama-3.1-8B-Instruct weights.' >&2; exit 2; fi
export PATH="$(dirname "$(command -v "$py")"):$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$volume/phasekv-triton-sm89}"
out="$volume/phasekv-results/4090d_speed_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$out"
echo "Log and results: $out"
"$py" -u "$here/quick.py" --model "$model" --out "$out" "$@" 2>&1 | tee "$out/run.log"
