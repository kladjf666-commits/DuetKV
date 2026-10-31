#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PHASEKV_NATIVE_VENV:-$PWD/.venv}/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/validation_$STAMP"
"$PY" prepare.py --verify
"$PY" tests/test_native_contract.py
"$PY" tests/validate_cuda.py --out "$OUT/kernel_checks.json"
"$PY" run.py --tiny --methods full kivi snapkv scope scope_slide scope_adaptive phasekv --out "$OUT/tiny" --warmups 0
"$PY" check_run.py "$OUT/tiny"
if [[ $# -ge 1 ]]; then
    "$PY" check_model.py "$1"
    "$PY" run.py --model "$1" --methods full kivi snapkv scope scope_slide scope_adaptive phasekv --out "$OUT/llama31_8b"
    "$PY" check_run.py "$OUT/llama31_8b"
else
    echo 'Tiny-model/kernel gates passed. Real 8B deployment is still pending; rerun with the local model path.'
fi
echo "Validation artifacts: $PWD/$OUT"
