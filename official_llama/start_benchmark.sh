#!/usr/bin/env bash
# Run once. Reusing the same output directory resumes atomic completed records.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PHASEKV_NATIVE_VENV="${PHASEKV_NATIVE_VENV:-/data/phasekv-native-venv}"
export PATH="$PHASEKV_NATIVE_VENV/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
PY="$PHASEKV_NATIVE_VENV/bin/python"
OUT="${PHASEKV_BENCHMARK_OUT:-/data/phasekv-results/native_llama_20261029}"
MODEL="${PHASEKV_MODEL:-/data/Llama-3.1-8B-Instruct/snapshots/master}"
mkdir -p "$OUT"
# Same lock as the previously supplied validation command, shared for all GPUs here.
exec 9>/data/phasekv-native-validation.lock
if ! flock -n 9; then
    echo 'A validation/benchmark launcher is already running. No duplicate was started.'
    exit 75
fi
"$PY" verify_benchmark_bundle.py
# Pure-Python scoring wheels only; no network, CUDA build or torch reinstall.
"$PY" -m pip install --no-index --find-links bench_assets/wheels --no-deps \
  rouge==1.0.1 jieba==0.42.1 fuzzywuzzy==0.18.0 six==1.17.0
LOG="$OUT/launcher_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup "$PY" -u benchmark.py --model "$MODEL" --out "$OUT" "$@" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
printf '%s\n' "$LOG" > "$OUT/latest_log.txt"
echo "Formal benchmark launched: PID=$PID"
echo "Log: $LOG"
echo "Results: $OUT"
echo "Watch: tail -f \"$LOG\""
# Child inherits descriptor 9 and holds the launch lock until it exits.
