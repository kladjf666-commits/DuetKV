"""Install the four-arm benchmark from a verified local wheel overlay, offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import site
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack_archives(folder, dest, names):
    for name in names:
        with zipfile.ZipFile(folder/name) as z:
            for entry in z.infolist():
                path = (dest/entry.filename).resolve()
                if not path.is_relative_to(dest/'official_llama'):
                    raise RuntimeError(f'Unexpected archive path: {entry.filename}')
            z.extractall(dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/data/phasekv-ws'))
    args = p.parse_args()
    target = args.workspace.resolve()
    if sys.platform != 'linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 10):
        raise RuntimeError('Use Linux x86_64 and the base image CPython 3.10 interpreter')
    libc, ver = platform.libc_ver()
    if libc != 'glibc' or tuple(map(int, ver.split('.')[:2])) < (2, 35):
        raise RuntimeError('This overlay targets Ubuntu 22.04/glibc 2.35 or newer')
    if not shutil.which('gcc'):
        raise RuntimeError('Triton JIT needs a host C compiler; select a development image with gcc')
    import sysconfig
    if not (Path(sysconfig.get_path('include'))/'Python.h').exists():
        raise RuntimeError('Select a development image with CPython 3.10 headers for Triton JIT')
    # Check the actual base before allocating or installing anything.
    import torch
    if (torch.__version__.split('+')[0] != '2.5.1' or torch.version.cuda != '12.1'
            or torch._C._GLIBCXX_USE_CXX11_ABI or not torch.cuda.is_available()):
        raise RuntimeError('Base must supply working torch 2.5.1 CUDA 12.1, CXX11 ABI=False')
    del torch
    target.mkdir(parents=True, exist_ok=True)
    for name in ('venv', 'official_llama'):
        if (target/name).exists():
            raise FileExistsError(f'Choose a new --workspace; {target/name} already exists')
    manifest = json.loads((HERE/'manifest.json').read_text('utf-8'))
    for key, folder in [('base_archives', 'base'), ('wheels', 'wheels')]:
        for name, sha in manifest[key].items():
            if digest(HERE/folder/name) != sha:
                raise ValueError('Hash mismatch: '+name)
    env = {**os.environ, 'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONNOUSERSITE': '1'}
    def run(*cmd, **kwargs):
        subprocess.run(list(map(str, cmd)), env=env, check=True, **kwargs)
    project = target/'official_llama'
    with tempfile.TemporaryDirectory(dir=target) as stage:
        stage = Path(stage)
        unpack_archives(HERE/'base', stage, manifest['base_archives'])
        work = stage/'official_llama'
        shutil.copy2(HERE/'parallel.py', work/'workspace_parallel.py')
        run(sys.executable, work/'verify_attention_ablation_bundle.py')
        run(sys.executable, work/'verify_benchmark_bundle.py')
        work.rename(project)
    venv = target/'venv'
    run(sys.executable, '-m', 'venv', '--system-site-packages', venv)
    py = venv/'bin/python'
    # Also handles images whose Python itself is a venv, rather than system Python.
    dest = venv/'lib/python3.10/site-packages/phasekv-base-image.pth'
    dest.write_text('\n'.join(site.getsitepackages())+'\n')
    # Install only the pinned binary artifacts into the new venv. No resolver,
    # network, extension build, or mutation of the image's torch/CUDA stack.
    run(py, '-m', 'pip', 'install', '--no-index', '--no-deps', '--ignore-installed',
        '--only-binary=:all:', *(HERE/'wheels'/w for w in manifest['wheels']))
    test = '''import torch, flash_attn, triton, transformers
assert torch.__version__.split('+')[0]=='2.5.1' and torch.version.cuda=='12.1'
assert not torch._C._GLIBCXX_USE_CXX11_ABI
assert (flash_attn.__version__, triton.__version__, transformers.__version__)==('2.7.4.post1','3.1.0','4.46.3')
from flash_attn import flash_attn_func
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        q=torch.randn(1,1,32,128,device='cuda',dtype=torch.float16); k=torch.randn(1,33,8,128,device='cuda',dtype=q.dtype)
        y=flash_attn_func(q,k,k,causal=True); assert y.shape==q.shape and torch.isfinite(y).all().item()
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i), 'FA2 GQA OK')'''
    run(py, '-c', test)
    print(f'Installed offline. Project: {project}; Python: {py}. Full GPU gates still required.')


if __name__ == '__main__':
    main()
