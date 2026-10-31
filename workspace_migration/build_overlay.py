"""Build an offline Linux CPython 3.10 wheel overlay for the pinned PyTorch image.

Run on the originating Windows workstation; no Linux compiler is used. Read
the original successful run's freeze and verify downloads against PyPI SHA256.
"""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import zipfile
from email import message_from_bytes

import requests
from packaging.tags import cpython_tags, compatible_tags
from packaging.utils import parse_wheel_filename, canonicalize_name
from packaging.specifiers import SpecifierSet
from packaging.requirements import Requirement
from packaging.markers import default_environment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DELIVERY = ROOT/'official_llama/delivery'
CACHE = HERE/'wheelhouse'
BASE = ['PhaseKV_native_llama_20261028.zip', 'PhaseKV_formal_benchmark_20261029.zip',
        'PhaseKV_attention_ablation_20261031.zip']
FLASH_SHA = 'b267f80a08e516292cdd748056a2178a45b8abedf7fca123292eb17c21c8c87c'
PLATFORMS = [f'manylinux_2_{i}_x86_64' for i in range(35, 4, -1)] + [
    'manylinux2014_x86_64', 'manylinux2010_x86_64', 'manylinux1_x86_64', 'linux_x86_64']
TAGS = list(cpython_tags((3, 10), abis=['cp310'], platforms=PLATFORMS)) + list(
    compatible_tags((3, 10), interpreter='cp310', platforms=PLATFORMS))
RANK = {tag: i for i, tag in enumerate(TAGS)}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(pin):
    name, version = pin.split('==')
    r = requests.get(f'https://pypi.org/pypi/{name}/{version}/json', timeout=40)
    r.raise_for_status()
    candidates = []
    for entry in r.json()['urls']:
        if entry['packagetype'] != 'bdist_wheel' or entry.get('yanked'):
            continue
        _, _, _, tags = parse_wheel_filename(entry['filename'])
        rank = min((RANK[t] for t in tags if t in RANK), default=None)
        if rank is not None and SpecifierSet(entry.get('requires_python') or '').contains('3.10.19'):
            candidates.append((rank, entry['filename'], entry))
    if not candidates:
        raise ValueError(f'No cp310/x86_64 Linux wheel: {pin}')
    _, filename, entry = min(candidates)
    path = CACHE/filename
    expected = entry['digests']['sha256']
    if not path.exists() or sha(path) != expected:
        partial = path.with_suffix('.part')
        with requests.get(entry['url'], stream=True, timeout=(20, 100)) as response:
            response.raise_for_status()
            with partial.open('wb') as f:
                for chunk in response.iter_content(1 << 20):
                    f.write(chunk)
        if sha(partial) != expected:
            raise ValueError(f'Download hash mismatch: {filename}')
        partial.replace(path)
    print('verified', filename, flush=True)
    return path, expected


def wheels():
    CACHE.mkdir(exist_ok=True)
    previous = next(ROOT.glob('*/native_llama_20261029/pip-freeze.txt')).read_text('utf-8')
    local = {canonicalize_name(parse_wheel_filename(p.name)[0]): p for p in
             (ROOT/'official_llama/bench_assets/wheels').glob('*.whl')}
    flash = next((ROOT/'Env').glob('flash_attn-*.whl'))
    assert sha(flash) == FLASH_SHA
    table = {flash: FLASH_SHA}
    pins = []
    for line in previous.splitlines():
        if '==' not in line:  # Flash handled above; KIVI isn't used by these four arms.
            continue
        name, version = line.split('==')
        name = canonicalize_name(name)
        if name == 'torch' or name.startswith('nvidia-'):
            continue  # Supplied by the explicitly required base image.
        if name in local:
            p = local[name]
            assert str(parse_wheel_filename(p.name)[1]) == version
            table[p] = sha(p)
        else:
            pins.append(line)
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for path, digest in ex.map(fetch, pins):
            table[path] = digest
    return table


def build():
    table = wheels()
    # Check runtime dependencies for the *target* OS/interpreter rather than
    # resolving Linux wheels against the builder's Windows environment.
    env = {**default_environment(), 'sys_platform':'linux', 'os_name':'posix',
           'platform_system':'Linux','platform_machine':'x86_64','python_version':'3.10',
           'python_full_version':'3.10.19','extra':''}
    known = {'torch':'2.5.1+cu121'}
    requirements = []
    for path in table:
        with zipfile.ZipFile(path) as z:
            if z.testzip(): raise ValueError('Damaged wheel: '+path.name)
            metadata = message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
        known[canonicalize_name(metadata['Name'])] = metadata['Version']
        requirements.extend(Requirement(s) for s in metadata.get_all('Requires-Dist',[]))
    for req in requirements:
        if req.marker and not req.marker.evaluate(env): continue
        ver = known.get(canonicalize_name(req.name))
        if ver is None or ver not in req.specifier:
            raise ValueError(f'Unmet target dependency: {req}; present={ver}')
    manifest = {'schema': 1, 'base_image': 'Linux x86_64, glibc >=2.35, CPython 3.10, torch 2.5.1+cu121, ABI=False',
                'wheels': {p.name: v for p,v in sorted(table.items())}, 'base_archives': {}}
    for filename in BASE:
        path = DELIVERY/filename
        expected = path.with_suffix('.zip.sha256').read_text().split()[0]
        if sha(path) != expected:
            raise ValueError('Base archive hash mismatch: '+filename)
        manifest['base_archives'][filename] = expected
    out = DELIVERY/'PhaseKV_new_workspace_cp310_cu121.zip'
    tmp = out.with_suffix('.partial')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(table):
            z.write(p, 'migration/wheels/'+p.name)
        for name in BASE:
            z.write(DELIVERY/name, 'migration/base/'+name)
        for name in ('install.py', 'parallel.py', 'README.txt'):
            z.write(HERE/name, 'migration/'+name)
        z.writestr('migration/manifest.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
    tmp.replace(out)
    out.with_suffix('.zip.sha256').write_text(sha(out)+'  '+out.name+'\n', encoding='ascii')
    print(out, out.stat().st_size, sha(out))


if __name__ == '__main__':
    build()
