#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
VENV="${PHASEKV_VENV:-/data/phasekv-native-venv}"
PY="$VENV/bin/python"
OUT="${PHASEKV_OUT:-/data/phasekv-results/attention_ablation_20261031}"
MODEL="${PHASEKV_MODEL:-/data/Llama-3.1-8B-Instruct/snapshots/master}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
test -x "$PY"
mkdir -p "$OUT"
# Same launcher lock as the previous benchmark/validation: no overlapping jobs.
exec 9>"${PHASEKV_LOCK:-/data/phasekv-native-validation.lock}"
flock -n 9 || { echo 'Another validation/benchmark is running; no new job started.'; exit 75; }
"$PY" verify_attention_ablation_bundle.py
"$PY" attention_ablation.py --model "$MODEL" --out "$OUT" --prepare-only "$@"
LOG="$OUT/launcher_$(date +%Y%m%d_%H%M%S).log"
nohup "$PY" -u attention_ablation.py --model "$MODEL" --out "$OUT" "$@" >"$LOG" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
printf '%s\n' "$LOG" > "$OUT/latest_log.txt"
echo "Started PID=$PID; GPU correctness gates run before formal measurements."
echo "Progress: tail -f '$LOG'"
echo "Gate details: tail -f '$OUT/gpu_gate.log'"
echo "Results: $OUT/summary.csv and $OUT/ablation_pairs.json"
