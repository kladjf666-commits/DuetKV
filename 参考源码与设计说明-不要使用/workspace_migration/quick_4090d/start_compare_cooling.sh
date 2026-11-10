#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
(cd "$here" && sha256sum -c SHA256SUMS)
export PATH=/data/phasekv-native-venv/bin:$PATH
export TRITON_CACHE_DIR=/data/phasekv-triton-sm89
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec 9>/data/phasekv-native-benchmark.lock
flock -n 9 || { echo 'Benchmark GPU lock is busy.' >&2; exit 3; }
exec 8>/data/phasekv-native-validation.lock
flock -n 8 || { echo 'Validation GPU lock is busy.' >&2; exit 3; }
mkdir -p /data/phasekv-results
out=$(mktemp -d /data/phasekv-results/4090d_cooling_ab_XXXXXXXX)
echo "Log and results: $out"
/data/phasekv-native-venv/bin/python -u "$here/compare_cooling.py" --out "$out" "$@" 2>&1 | tee "$out/run.log"
