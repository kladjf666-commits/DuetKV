#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
py=/data/phasekv-native-venv/bin/python
if [[ ! -x "$py" ]]; then echo "Missing benchmark Python: $py" >&2; exit 2; fi
(cd "$here" && sha256sum -c SHA256SUMS)
export PATH=/data/phasekv-native-venv/bin:$PATH
export TRITON_CACHE_DIR=/data/phasekv-triton-sm89
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec 9>/data/phasekv-native-benchmark.lock
if ! flock -n 9; then echo 'Another benchmark holds the GPU lock.' >&2; exit 3; fi
exec 8>/data/phasekv-native-validation.lock
if ! flock -n 8; then echo 'Another validation holds the GPU lock.' >&2; exit 3; fi
mkdir -p /data/phasekv-results
out=$(mktemp -d /data/phasekv-results/4090d_cooling_cpu_XXXXXXXX)
cp "$here/cooling_cpu.py" "$here/start.sh" "$here/README.txt" "$here/SHA256SUMS" "$out/"
echo "Log and results: $out"
"$py" -u "$out/cooling_cpu.py" --out "$out" "$@" 2>&1 | tee "$out/run.log"
