#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
model_path=${1:?Usage: bash run_phasekv_optimizations.sh MODEL_PATH NEW_OUTPUT_DIR [full|smoke]}
output_dir=${2:?A fresh output directory is required}
mode=${3:-full}
if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite existing evidence: $output_dir" >&2
    exit 1
fi
if [[ "$mode" != full && "$mode" != smoke ]]; then
    echo "Mode must be full or smoke" >&2
    exit 1
fi
mkdir -p "$output_dir"
python scripts/test_phasekv_optimizations.py --require-cuda --output "$output_dir/tests.json"
python scripts/bench_phasekv_optimizations.py --prompts 1024 --cold-counts 0 7 128 \
    --hot-lengths 0 128 143 --head-layouts 28:4 --output "$output_dir/micro_smoke.json"
python scripts/analyze_phasekv_optimizations.py "$output_dir/micro_smoke.json" --output "$output_dir/micro_smoke.txt"
if [[ "$mode" == full ]]; then
    python scripts/bench_phasekv_optimizations.py --output "$output_dir/micro_full.json"
    python scripts/analyze_phasekv_optimizations.py "$output_dir/micro_full.json" --output "$output_dir/micro_full.txt"
    python scripts/compare_phasekv_optimizations.py --model-path "$model_path" --repeats 3 --out "$output_dir/e2e"
fi
