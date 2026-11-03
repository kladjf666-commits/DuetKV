"""Extend the completed RTX 4090 Full comparison with two equal output lengths."""
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DELIVERY = ROOT / 'PhaseKV与Full消融'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    previous = json.loads((DELIVERY / '4090_full_vs_phasekv_li8K4ePQ/summary.json').read_text(encoding='utf-8'))
    if previous['status'] != 'complete':
        raise RuntimeError('The preceding same-card Full comparison is incomplete')
    with zipfile.ZipFile(DELIVERY / 'PhaseKV_4090_full_vs_phasekv.zip') as archive:
        prefix = 'phasekv_4090_full_vs_phasekv/base/'
        manifest = archive.read(prefix + 'files.json')
        if digest(manifest) != previous['protocol']['code_sha256']:
            raise RuntimeError('Frozen runtime does not match the completed 4090 run')
        base = {}
        for name, expected in json.loads(manifest).items():
            data = archive.read(prefix + name)
            if digest(data) != expected:
                raise RuntimeError(f'Corrupt baseline file: {name}')
            base[name] = data

    frozen = ROOT / 'official_llama/bench_assets/frozen'
    pool = frozen / 'samples.jsonl.gz'
    corpus_hash = digest(pool.read_bytes())
    if corpus_hash != json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))['samples_sha256']:
        raise RuntimeError('Frozen corpus checksum mismatch')
    ids = json.loads(base['sample.json'])['input_ids']
    targets = {'performance/1024+4096/0': 1024, 'performance/8192+2048/0': 8192}
    specs = {}
    with gzip.open(pool, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            source = row['id']
            if source not in targets:
                continue
            length = targets[source]
            if (source in specs or row['input_ids'] != ids[:length] or len(row['input_ids']) != length
                    or digest(json.dumps(row['input_ids']).encode()) != row['input_sha256']
                    or row['natural_eos'] or row['max_new_tokens'] < 2048):
                raise RuntimeError(f'Unexpected frozen prompt or budget: {source}')
            specs[source] = dict(source_id=source, id=f'performance/{length}+2048/0',
                                 input_tokens=length, input_sha256=row['input_sha256'])
    plan = dict(frozen_samples_sha256=corpus_hash, cases=[specs[k] for k in targets],
                preceding_run=dict(gpu_uuid=previous['protocol']['gpu'].split(',')[0],
                                   shape=previous['protocol']['shape'], code_sha256=previous['protocol']['code_sha256']))
    # Only the length checks in the timing driver need adaptation. All model
    # adapters, generation code and PhaseKV kernels come from the verified run.
    base['quick.py'] = (HERE / 'quick.py').read_bytes().replace(b'\r\n', b'\n')
    base['files.json'] = json.dumps({k: digest(v) for k, v in base.items()}, indent=2).encode()
    files = {'base/' + name: data for name, data in base.items()}
    files['long_cases.json'] = json.dumps(plan, indent=2).encode()
    for name, src in [('start.sh', 'start_full_4090.sh'), ('full_speed.py', 'full_speed.py'),
                      ('long_decode.py', 'long_decode.py')]:
        files[name] = (HERE / src).read_bytes().replace(b'\r\n', b'\n')
    files['SHA256SUMS'] = ''.join(f'{digest(data)}  {name}\n' for name, data in files.items()).encode()
    target = DELIVERY / 'PhaseKV_4090_long_decode.zip'
    tmp = target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as out:
        for name, data in files.items():
            out.writestr('phasekv_4090_long_decode/' + name, data)
    tmp.replace(target)
    print(target, target.stat().st_size)


if __name__ == '__main__':
    build()
