"""Bundle the frozen baseline algorithms, updated cache and fixed speed sample."""
import hashlib
import json
import zipfile
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[1]
delivery = root / 'PhaseKV与Full消融'
original_manifest = 'bcc631430218bd83a241faf6c852eddb11c8a19efbf900852e5545f4e0ae72d7'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    with zipfile.ZipFile(delivery / 'PhaseKV_4090D_speed_quick.zip') as source:
        prefix = 'phasekv_4090d_speed/'
        manifest = source.read(prefix + 'files.json')
        if digest(manifest) != original_manifest:
            raise RuntimeError('Unexpected frozen speed package')
        base = {}
        for name, expected in json.loads(manifest).items():
            data = source.read(prefix + name)
            if digest(data) != expected:
                raise RuntimeError(f'Corrupt frozen file: {name}')
            if name not in ('run.sh', 'README.txt'):
                base[name] = data
    # Adapt only the measurement driver in base/. The baseline algorithm files
    # above are identical to those in the previous speed package.
    base['quick.py'] = (here / 'quick.py').read_bytes().replace(b'\r\n', b'\n')
    base['files.json'] = json.dumps({name: digest(data) for name, data in base.items()}, indent=2).encode()
    files = {'base/' + name: data for name, data in base.items()}
    for name, path in (('start.sh', here / 'start_retention_4090.sh'),
                       ('compare_cooling.py', here / 'compare_cooling.py'),
                       ('cooling_cpu.py', here / 'cooling_cpu.py'),
                       ('phasekv_cache.py', root / 'PhaseKV/phasekv_cache.py'),
                       ('segmented_plan.py', root / 'PhaseKV/kernels/segmented_plan.py')):
        files[name] = path.read_bytes().replace(b'\r\n', b'\n')
    files['SHA256SUMS'] = ''.join(f'{digest(data)}  {name}\n' for name, data in files.items()).encode()
    target = delivery / 'PhaseKV_4090_cooling_retention.zip'
    temp = target.with_suffix('.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr('phasekv_4090_cooling_retention/' + name, data)
    temp.replace(target)
    print(target, target.stat().st_size)


if __name__ == '__main__':
    build()
