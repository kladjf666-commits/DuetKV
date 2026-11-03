#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
volume=$(dirname "$here")
(cd "$here" && sha256sum --quiet -c SHA256SUMS)
model=${PHASEKV_MODEL:-}
if [[ -z "$model" ]]; then
    for parent in "$volume" /data /root/data; do
        path="$parent/Llama-3.1-8B-Instruct/snapshots/master"
        if [[ -f "$path/model.safetensors.index.json" ]]; then model="$path"; break; fi
    done
fi
if [[ ! -f "$model/model.safetensors.index.json" ]]; then
    echo 'Set PHASEKV_MODEL to the Llama-3.1-8B-Instruct checkpoint directory.' >&2; exit 2
fi
py=${PHASEKV_PYTHON:-}
if [[ -z "$py" ]]; then
    for envdir in "$volume/phasekv-ws/venv" "$volume/phasekv-native-venv" /root/data/phasekv-ws/venv /data/phasekv-native-venv "${VIRTUAL_ENV:-}"; do
        if [[ -n "$envdir" && -x "$envdir/bin/python" ]]; then py="$envdir/bin/python"; break; fi
    done
    py=${py:-python3}
fi
export PATH="$(dirname "$(command -v "$py")"):$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$volume/phasekv-triton-sm89}"
exec 9>"$volume/phasekv-native-benchmark.lock"
flock -n 9 || { echo 'Benchmark GPU lock is busy.' >&2; exit 3; }
exec 8>"$volume/phasekv-native-validation.lock"
flock -n 8 || { echo 'Validation GPU lock is busy.' >&2; exit 3; }
mkdir -p "$volume/phasekv-results"
out=$(mktemp -d "$volume/phasekv-results/4090_cooling_retention_XXXXXXXX")
echo "Log and results: $out"
"$py" -u "$here/compare_cooling.py" --rtx4090-retention --project "$here/base" \
    --model "$model" --out "$out" "$@" 2>&1 | tee "$out/run.log"
