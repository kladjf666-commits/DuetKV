"""Bundle the exact runtime and optimized sources from the completed retention run."""
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parents[1] / 'PhaseKV与Full消融'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    result = json.loads((DELIVERY / '4090_cooling_retention_6Ed7ciwp/summary.json').read_text(encoding='utf-8'))
    if result['status'] != 'complete':
        raise RuntimeError('Previous retention run did not complete')
    protocol = result['base_protocol']
    with zipfile.ZipFile(DELIVERY / 'PhaseKV_4090_cooling_retention.zip') as source:
        prefix = 'phasekv_4090_cooling_retention/'
        manifest = source.read(prefix + 'base/files.json')
        if digest(manifest) != protocol['code_sha256']:
            raise RuntimeError('Base runtime differs from the completed run')
        base = {}
        for name, expected in json.loads(manifest).items():
            data = source.read(prefix + 'base/' + name)
            if digest(data) != expected:
                raise RuntimeError(f'Corrupt source file: {name}')
            base[name] = data
        for name, target in [('phasekv_cache.py', 'runtime/PhaseKV/phasekv_cache.py'),
                             ('segmented_plan.py', 'runtime/PhaseKV/kernels/segmented_plan.py')]:
            data = source.read(prefix + name)
            if digest(data) != result['candidate_sha256'][name]:
                raise RuntimeError(f'Optimized source differs from the completed run: {name}')
            base[target] = data
    base['files.json'] = json.dumps({k: digest(v) for k, v in base.items()}, indent=2).encode()
    files = {'base/' + name: data for name, data in base.items()}
    prior = dict(gpu_uuid=protocol['gpu'].split(',')[0], input_sha256=protocol['input_sha256'],
                 phase_output_sha256=result['reference_by_retention'],
                 optimized_source_sha256=result['candidate_sha256'])
    files['prior_run.json'] = json.dumps(prior, indent=2).encode()
    for target, source in [('start.sh', 'start_full_4090.sh'), ('full_speed.py', 'full_speed.py')]:
        files[target] = (HERE / source).read_bytes().replace(b'\r\n', b'\n')
    files['SHA256SUMS'] = ''.join(f'{digest(data)}  {name}\n' for name, data in files.items()).encode()
    target = DELIVERY / 'PhaseKV_4090_full_vs_phasekv.zip'
    tmp = target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr('phasekv_4090_full_vs_phasekv/' + name, data)
    tmp.replace(target)
    print(target, target.stat().st_size)


if __name__ == '__main__':
    build()
