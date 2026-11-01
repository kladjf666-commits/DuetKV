#!/usr/bin/env bash
set -euo pipefail
tools_dir=$(cd "$(dirname "$0")" && pwd -P)
volume=$(dirname "$tools_dir")
workspace="$volume/phasekv-ws"
project="$workspace/official_llama"
output="$volume/phasekv-results/rtx4090_b1_common_v1"

python "$tools_dir/install.py" --check-gpu --expected-gpus 1 --workspace "$workspace"
if [[ -e "$project/rtx4090_common.py" ]]; then
    cmp "$tools_dir/rtx4090_common.py" "$project/rtx4090_common.py"
else
    cp "$tools_dir/rtx4090_common.py" "$project/rtx4090_common.py"
fi
export PHASEKV_TRITON_CACHE_DIR="$volume/phasekv-triton-sm89"
exec "$workspace/venv/bin/python" -u "$project/rtx4090_common.py" \
    --gpus 1 --first-8k --profile-8k --skip-prior-oom --out "$output"
