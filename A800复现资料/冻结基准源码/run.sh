#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PHASEKV_MODEL="${PHASEKV_MODEL:-/data/Llama-3.1-8B-Instruct/snapshots/master}"
export A800_OUT="${A800_OUT:-/data/phasekv-results/a800_s1_v1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
action=${1:-plan}; shift || true
if [[ $action == plan ]]; then exec python3 inventory.py; fi
mkdir -p "$A800_OUT"
if [[ $action == summary ]]; then exec python3 summary.py "$A800_OUT"; fi
exec 9>"/tmp/phasekv-a800-single-${CUDA_VISIBLE_DEVICES//\//_}.lock"
flock -n 9 || { echo "This GPU already has an A800 package run"; exit 1; }
exec > >(tee -a "$A800_OUT/run.log") 2>&1
echo "[$(date -Iseconds)] $action $*; GPU=$CUDA_VISIBLE_DEVICES"
sha256sum -c --quiet SHA256SUMS
envdir="${A800_ENVS:-$PWD/.envs}"
mainpy="${A800_MAIN_PYTHON:-$envdir/main/bin/python}"
milpy="${A800_MILLION_PYTHON:-$envdir/million/bin/python}"
clupy="${A800_CLUSTER_PYTHON:-$envdir/clusterkv/bin/python}"
if [[ $action == calibrate ]]; then exec "$milpy" calibrate.py; fi
case "$action" in
  check|speed|quality) methods="${A800_MAIN_METHODS:-full phasekv kivi snapkv scope pyramidkv}";;
  new-check|new-speed|new-quality) methods="${A800_NEW_METHODS:-million clusterkv}"; action=${action#new-};;
  *) echo "Actions: plan check speed quality calibrate new-check new-speed new-quality summary"; exit 2;;
esac
[[ -d $PHASEKV_MODEL ]] || { echo "Model directory missing: $PHASEKV_MODEL"; exit 1; }
rc=0
for m in $methods; do
  case "$m" in
    full|phasekv|kivi|snapkv|scope|pyramidkv) py=$mainpy;;
    million) py=$milpy;;
    clusterkv) py=$clupy;;
    *) echo "Unknown method: $m"; exit 2;;
  esac
  echo "[$(date -Iseconds)] $m $action"
  if [[ $action != check ]] && ! python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.exit(0 if p.is_file() and json.loads(p.read_text()).get("status")=="ok" else 1)' "$A800_OUT/$m/gate.json"; then
    echo "SKIP $m: no passing check"; rc=1; continue
  fi
  if [[ ! -x $py ]]; then echo "Missing $py. Run setup.sh for this environment."; fi
  if "$py" worker.py "$action" "$m" "$@"; then :; else
    code=$?; rc=1; echo "$m $action failed with exit code $code"
    if [[ $action == check ]]; then
      python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); old=json.loads(p.read_text()) if p.exists() else {}; old.update(status="failed",exit_code=int(sys.argv[2])); t=p.with_suffix(".tmp"); t.write_text(json.dumps(old)); t.replace(p)' "$A800_OUT/$m/gate.json" "$code"
    fi
  fi
done
python3 summary.py "$A800_OUT"
exit "$rc"
