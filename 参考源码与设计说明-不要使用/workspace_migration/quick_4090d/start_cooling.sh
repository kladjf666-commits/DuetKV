#!/usr/bin/env bash
set -euo pipefail
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export PATH=/data/phasekv-native-venv/bin:$PATH
export TRITON_CACHE_DIR=/data/phasekv-triton-sm89
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ -f "$dir/measure.ready" ]]; then
    (cd "$dir" && sha256sum -c SHA256SUMS)
    mv "$dir/measure.ready" "$dir/measure.py"
fi
if [[ ! -f "$dir/measure.py" ]]; then echo 'Missing measure.py' >&2; exit 2; fi
pattern="/data/phasekv-native-venv/bin/python -u $dir/launch.py $dir"
if pgrep -f -x -- "$pattern" >/dev/null; then
    echo 'The existing model loader will pick up measure.py.'
elif [[ -f "$dir/cooling_summary.json" ]]; then
    echo "A result is already present. Inspect $dir/cooling_summary.json and $dir/run.log."
    exit 1
else
    nohup flock -n /data/phasekv-native-benchmark.lock flock -n /data/phasekv-native-validation.lock /data/phasekv-native-venv/bin/python -u "$dir/launch.py" "$dir" >> "$dir/run.log" 2>&1 </dev/null &
    echo "Started loader PID=$!"
fi
echo "Results: $dir/cooling_summary.json"
tail -n 12 -f "$dir/run.log"
