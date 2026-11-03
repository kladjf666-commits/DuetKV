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
[[ -f "$model/model.safetensors.index.json" ]] || { echo 'Set PHASEKV_MODEL to the checkpoint directory.' >&2; exit 2; }
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
entry=full_speed.py
label=4090_full_vs_phasekv
if [[ -f "$here/batch_speed.py" ]]; then
    entry=batch_speed.py
    label=4090_batch2_6
elif [[ -f "$here/long_decode.py" ]]; then
    entry=long_decode.py
    label=4090_long_decode
elif [[ -f "$here/prompt100_speed.py" ]]; then
    entry=prompt100_speed.py
    label=4090_prompt100_speed_2cases
fi
if [[ "$entry" == batch_speed.py && -n "${PHASEKV_OUT:-}" ]]; then
    out=$PHASEKV_OUT
    mkdir -p "$out"
else
    out=$(mktemp -d "$volume/phasekv-results/${label}_XXXXXXXX")
fi
echo "Log and results: $out"
"$py" -u "$here/$entry" --model "$model" --out "$out" "$@" 2>&1 | tee -a "$out/run.log"
