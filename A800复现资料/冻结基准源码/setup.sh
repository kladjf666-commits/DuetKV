#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD
kind=$1
case "$kind" in main|million|clusterkv) ;; *) echo "Usage: bash setup.sh main|million|clusterkv"; exit 2;; esac
export A800_OUT="${A800_OUT:-/data/phasekv-results/a800_s1_v1}"
mkdir -p "$A800_OUT/install"
exec > >(tee -a "$A800_OUT/install/$kind.log") 2>&1
command -v nvcc >/dev/null || { echo 'CUDA development toolkit and nvcc required (ClusterKV upstream tests CUDA 12.8).'; exit 1; }
export MAX_JOBS="${MAX_JOBS:-4}" CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS:-4}" TORCH_CUDA_ARCH_LIST=8.0
envroot="${A800_ENVS:-$root/.envs}"
mkdir -p "$envroot"; envroot=$(realpath "$envroot"); venv=$envroot/$kind
py=$venv/bin/python
"${PYTHON_BIN:-python3}" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,11), "Use Python 3.10 or 3.11 (set PYTHON_BIN)"'
"${PYTHON_BIN:-python3}" -m venv "$venv"
export VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH"
"$py" -m pip install -U pip 'setuptools<81' wheel packaging ninja psutil
if [[ $kind == clusterkv ]]; then
  for bin in git /usr/bin/gcc-11 /usr/bin/g++-11; do command -v "$bin" >/dev/null || { echo "Missing $bin"; exit 1; }; done
  repo=$root/upstream/ClusterKV
  pinned=c7380819588771e8c5dadacdbd33942d7c53adea
  [[ -d $repo/.git ]] || git clone --no-checkout https://github.com/sjtu-zhao-lab/ClusterKV.git "$repo"
  git -C "$repo" checkout --detach "$pinned"
  git -C "$repo" submodule update --init --recursive --jobs 4
  "$py" "$root/patch_cluster.py" "$repo"
  "$py" -m pip install 'uv>=0.7.3'
  (cd "$repo"; UV_PROJECT_ENVIRONMENT="$venv" "$venv/bin/uv" sync --frozen --python "$py")
  "$py" -m pip install 'cmake>=3.26.4,<4' ninja packaging psutil wheel
  "$py" -m pip install flash-attn==2.6.3 --no-build-isolation
  (cd "$repo/3rdparty/raft"; export INSTALL_PREFIX="$venv" PARALLEL_LEVEL="$MAX_JOBS"; ./build.sh libraft)
  build=$repo/kernel/build-a800
  cmake -S "$repo/kernel" -B "$build" -GNinja -DCMAKE_PREFIX_PATH="$venv;$("$py" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  cmake --build "$build" --parallel "$MAX_JOBS"
  for ext in _clusterkv_knl _quest_knl; do cp "$build/$ext"*.so "$repo/clusterkv/"; done
  "$py" -m pip install --no-deps -e "$repo"
  "$py" -c 'import torch,clusterkv._quest_knl,clusterkv._clusterkv_knl,flashinfer'
else
  "$py" -m pip install 'torch==2.5.1+cu121' --extra-index-url https://download.pytorch.org/whl/cu121
  "$py" -m pip install 'transformers==4.46.3' 'accelerate==1.1.1' 'numpy==1.26.4' 'sentencepiece==0.2.0' 'protobuf==5.28.3'
  "$py" -m pip install 'flash-attn==2.7.4.post1' --no-build-isolation
  if [[ $kind == main ]]; then
    (cd native/runtime/KIVI/quant; "$py" -m pip install . --no-deps --no-build-isolation)
    "$py" -c 'import torch,flash_attn,kivi_gemv'
  else
    "$py" -m pip install 'faiss-cpu==1.9.0.post1' 'pykeops==2.2.3' pybind11 'pyarrow==17.0.0'
    mkdir -p "$envroot/million-build"
    cp -a upstream/MILLION/scripts/modeldb/bindings/. "$envroot/million-build/"
    "$py" million_quality.py "$envroot/million-build"
    (cd "$envroot/million-build"; "$py" -m pip install . --no-deps --no-build-isolation)
    "$py" -c 'import torch,bindings,faiss,pykeops; assert hasattr(bindings,"flash_decoding_allocated_buffer_f16u8_Ns32Lt128d128M64C256")'
  fi
fi
"$py" -m pip install 'rouge==1.0.1' 'fuzzywuzzy==0.18.0' 'jieba==0.42.1'
"$py" -m pip freeze > "$A800_OUT/install/$kind-freeze.txt"
nvcc --version; nvidia-smi; echo "Installed $kind at $py"
