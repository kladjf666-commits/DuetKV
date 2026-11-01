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


GPU_TEST = '''import sys, torch
if torch.__version__.split('+')[0] != '2.5.1' or torch.version.cuda != '12.1' or torch._C._GLIBCXX_USE_CXX11_ABI:
    raise RuntimeError('GPU image must provide torch 2.5.1 CUDA 12.1 with CXX11 ABI=False')
count = int(sys.argv[1])
if torch.cuda.device_count() < count:
    raise RuntimeError('Fewer visible devices than --expected-gpus')
import flash_attn, triton, transformers
if (flash_attn.__version__, triton.__version__, transformers.__version__) != ('2.7.4.post1','3.1.0','4.46.3'):
    raise RuntimeError('GPU interpreter is loading a different binary environment')
from flash_attn import flash_attn_func
for i in range(count):
    with torch.cuda.device(i):
        q=torch.randn(1,1,32,128,device='cuda',dtype=torch.float16); k=torch.randn(1,33,8,128,device='cuda',dtype=q.dtype)
        y=flash_attn_func(q,k,k,causal=True)
        if y.shape!=q.shape or not torch.isfinite(y).all().item(): raise RuntimeError('FA2 smoke test failed')
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i), 'FA2 GQA OK')'''


def run(*cmd, **kwargs):
    env = {**os.environ, 'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONNOUSERSITE': '1'}
    subprocess.run(list(map(str, cmd)), env=env, check=True, **kwargs)


def model_check(project, model):
    run(sys.executable, project/'check_model.py', model)
    frozen = json.loads((project/'bench_assets/frozen/manifest.json').read_text())
    for name, expected in frozen['tokenizer_files'].items():
        if name != 'configuration.json' and digest(model/name) != expected:
            raise ValueError('Model/tokenizer file differs from frozen inputs: '+name)


def prepare_cpu(target, manifest, identity, model):
    project = target/'official_llama'
    if project.exists():
        check_prepared(project, identity)
        if model: model_check(project, model)
        return
    if (target/'venv').exists():
        raise FileExistsError('Choose a fresh workspace; this one already has an environment')
    for group, folder in [('base_archives', 'base'), ('wheels', 'wheels')]:
        for name, expected in manifest[group].items():
            if digest(HERE/folder/name) != expected:
                raise ValueError('Hash mismatch: '+name)
    with tempfile.TemporaryDirectory(dir=target) as stage:
        stage = Path(stage)
        unpack_archives(HERE/'base', stage, manifest['base_archives'])
        work = stage/'official_llama'
        shutil.copy2(HERE/'parallel.py', work/'workspace_parallel.py')
        run(sys.executable, work/'verify_attention_ablation_bundle.py')
        run(sys.executable, work/'verify_benchmark_bundle.py')
        if model: model_check(work, model)
        # Persist Linux wheels without binding an interpreter, torch or driver
        # path from the CPU instance into the final GPU environment.
        run(sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
            '--only-binary=:all:', '--target', work/'.cp310-site',
            *(HERE/'wheels'/name for name in manifest['wheels']))
        (work/'.cpu-prepared.json').write_text(json.dumps(identity)+'\n')
        work.rename(project)


def check_prepared(project, identity):
    marker = project/'.cpu-prepared.json'
    if (not marker.is_file() or json.loads(marker.read_text()) != identity
            or not (project/'.cp310-site').is_dir()
            or digest(project/'workspace_parallel.py') != identity['parallel_sha256']):
        raise RuntimeError('No matching CPU preparation in this workspace. Use --prepare-cpu with a new directory.')


def stage_cu121(project, *, offline=False):
    """Install the binary torch stack on the CPU instance; offline uses only uploaded files."""
    torch_site = project/'.torch-cu121'
    wheels = []
    lock = HERE/'runtime_manifest.json'
    if offline:
        if not lock.is_file():
            raise FileNotFoundError('Unzip the torch/cu121 offline supplement into the volume first')
        expected = json.loads(lock.read_text())
        if len(expected)!=13 or 'torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl' not in expected:
            raise RuntimeError('Incomplete Linux torch/CUDA wheel manifest')
        for name, checksum in expected.items():
            if Path(name).name!=name or not name.endswith('.whl'): raise ValueError('Invalid wheel name')
            path = HERE/'runtime_wheels'/name
            if not path.is_file() or digest(path)!=checksum: raise ValueError('Runtime wheel hash mismatch: '+name)
            wheels.append(path)
    probe = """import sys
from pathlib import Path
sys.path[:0] = sys.argv[1:3]
import torch
if not Path(torch.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve()):
    raise RuntimeError('torch did not load from the volume')
if torch.__version__.split('+')[0] != '2.5.1' or torch.version.cuda != '12.1' or torch._C._GLIBCXX_USE_CXX11_ABI:
    raise RuntimeError('Staged torch build mismatch')
print('Volume torch verified:', torch.__version__, torch.version.cuda)
"""
    if torch_site.exists():
        if offline and (not (torch_site/'.runtime-lock').is_file()
                        or (torch_site/'.runtime-lock').read_text()!=digest(lock)):
            raise RuntimeError('Existing torch site has a different provenance. Use a fresh workspace for the offline stack.')
        run(sys.executable, '-c', probe, project/'.cp310-site', torch_site)
        return
    with tempfile.TemporaryDirectory(dir=project.parent) as temp:
        dest = Path(temp)/'site'
        env = {**os.environ, 'TMPDIR':temp, 'PIP_DISABLE_PIP_VERSION_CHECK':'1'}
        base = [sys.executable,'-m','pip','--isolated','install','--no-cache-dir',
                '--ignore-installed','--only-binary=:all:','--no-compile','--target',str(dest)]
        if offline:
            env['PIP_NO_INDEX']='1'
            subprocess.run([*base,'--no-index','--no-deps',*map(str,wheels)],env=env,check=True)
            (dest/'.runtime-lock').write_text(digest(lock))
        else:
            env.pop('PIP_NO_INDEX',None)
            subprocess.run([*base,'--index-url','https://download.pytorch.org/whl/cu121',
                            '--find-links',str(HERE/'wheels'),'torch==2.5.1+cu121'],env=env,check=True)
        run(sys.executable, '-c', probe, project/'.cp310-site', dest)
        dest.rename(torch_site)


def check_gpu(target, identity, expected_gpus):
    project = target/'official_llama'
    check_prepared(project, identity)
    import sysconfig
    if not shutil.which('gcc') or not (Path(sysconfig.get_path('include'))/'Python.h').exists():
        raise RuntimeError('Select a GPU development image with gcc and CPython 3.10 headers for Triton JIT')
    venv = target/'venv'
    owner = venv/'.phasekv-overlay.json'
    if venv.exists() and (not owner.is_file() or json.loads(owner.read_text()) != identity):
        raise FileExistsError('The destination venv belongs to another installation; choose a new workspace')
    venv.mkdir(exist_ok=True)
    owner.write_text(json.dumps(identity)+'\n')
    run(sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', venv)
    # The GPU image supplies torch and its CUDA libraries; the overlay has priority.
    paths = [str(project/'.cp310-site')]
    if (project/'.torch-cu121').is_dir():
        paths.append(str(project/'.torch-cu121'))
    paths.extend(site.getsitepackages())
    dest = venv/'lib/python3.10/site-packages/phasekv-base-image.pth'
    dest.write_text('\n'.join(paths)+'\n')
    py = venv/'bin/python'
    run(py, '-m', 'pip', '--version')
    run(py, '-c', GPU_TEST, expected_gpus)
    print(f'GPU binary smoke passed. Python: {py}; formal numerical gates run at benchmark start.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=HERE.parent/'phasekv-ws')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--prepare-cpu', action='store_true', help='Stage project and wheels without any GPU')
    mode.add_argument('--check-gpu', action='store_true', help='Bind the prepared workspace to this GPU image')
    p.add_argument('--model', type=Path, help='Check uploaded checkpoint structure and frozen tokenizer/config hashes')
    p.add_argument('--expected-gpus', type=int, default=1, help='Minimum number of visible devices required by GPU smoke')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--with-cu121', action='store_true', help='Download pinned PyTorch CUDA wheels to the persistent volume during CPU preparation (requires internet)')
    source.add_argument('--offline-cu121', action='store_true', help='Install the uploaded torch/CUDA supplement on CPU without network access')
    args = p.parse_args()
    if args.expected_gpus < 1: p.error('--expected-gpus must be positive')
    if (args.with_cu121 or args.offline_cu121) and not args.prepare_cpu: p.error('Runtime installation requires --prepare-cpu')
    if sys.platform != 'linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 10):
        raise RuntimeError('Use Linux x86_64 with CPython 3.10')
    libc, ver = platform.libc_ver()
    if libc != 'glibc' or tuple(map(int, ver.split('.')[:2])) < (2, 35):
        raise RuntimeError('Use Ubuntu 22.04/glibc 2.35 or newer')
    import fcntl
    target = args.workspace.resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = HERE/'manifest.json'
    identity = {'schema':1, 'manifest_sha256':digest(manifest_path), 'parallel_sha256':digest(HERE/'parallel.py')}
    with (target/'.installation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.check_gpu:
            prepare_cpu(target, json.loads(manifest_path.read_text()), identity, args.model)
            if args.with_cu121 or args.offline_cu121:
                stage_cu121(target/'official_llama',offline=args.offline_cu121)
            print(f'CPU preparation complete: {target}.', flush=True)
        if not args.prepare_cpu:
            if args.check_gpu and args.model: model_check(target/'official_llama', args.model)
            check_gpu(target, identity, args.expected_gpus)


if __name__ == '__main__':
    main()
