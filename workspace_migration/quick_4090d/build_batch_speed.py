"""Build the 4090 B2--B6 sweep from the completed two-shape B1 benchmark."""
import gzip
import json
import zipfile

from build_full_4090 import DELIVERY, HERE, digest


def build():
    run = json.loads((DELIVERY / '4090_prompt100_speed_2cases_VjjdG5Fk/summary.json').read_text(encoding='utf8'))
    shapes = ('9192+256', '8192+2048')
    if run['status'] != 'complete' or set(run['cases']) != set(shapes):
        raise ValueError('Expected the completed two-shape B1 reference')
    with zipfile.ZipFile(DELIVERY / 'PhaseKV_4090_prompt100_speed_2cases.zip') as archive:
        prefix = 'phasekv_4090_prompt100_speed_2cases/'
        files = {}
        for line in archive.read(prefix + 'SHA256SUMS').decode().splitlines():
            expected, name = line.split('  ', 1)
            if name.startswith('base/') or name in ('speed_inputs.json', 'quality_reference.json'):
                data = archive.read(prefix + name)
                if digest(data) != expected:
                    raise ValueError(f'Bad reference file {name}')
                files[name] = data
    base_hash = digest(files['base/files.json'])
    plan = json.loads(files.pop('speed_inputs.json'))
    pool = HERE.parents[1] / 'official_llama/bench_assets/frozen/samples.jsonl.gz'
    if digest(pool.read_bytes()) != plan['frozen_samples_sha256']:
        raise ValueError('Frozen input corpus mismatch')
    wanted = {s['source_sample_id'].rsplit('/', 1)[0] for s in plan['cases']}
    with gzip.open(pool, 'rt', encoding='utf8') as stream:
        sources = {r['id']: r for line in stream if (r := json.loads(line))['id'].rsplit('/', 1)[0] in wanted}
    for sample in plan['cases']:
        length = len(sample['input_ids'])
        baseline = run['cases'][sample['task']]
        if (baseline['protocol']['code_sha256'] != base_hash
                or baseline['protocol']['input_sha256'] != sample['input_sha256']):
            raise ValueError('Model runtime or input differs from B1')
        rows = []
        for i in range(3):
            source = sources[sample['source_sample_id'].rsplit('/', 1)[0] + '/' + str(i)]
            if digest(json.dumps(source['input_ids']).encode()) != source['input_sha256']:
                raise ValueError('Invalid frozen input')
            rows.append(source['input_ids'][:length])
        # Keep BOS and rotate the body once to supply three distinct equal-length requests.
        rows += [r[:1] + r[2:] + r[1:2] for r in rows]
        hashes = [digest(json.dumps(r).encode()) for r in rows]
        if len(set(hashes)) != 6 or rows[0] != sample.pop('input_ids') or any(len(r) != length for r in rows):
            raise ValueError('Batch rows must be distinct and include the original B1 prompt')
        sample.update(rows=rows, row_sha256=hashes)
    files['batch_inputs.json'] = json.dumps(plan, separators=(',', ':')).encode()
    hashes = json.loads(files['base/files.json'])
    for path, data in hashes.items():
        if digest(files['base/' + path]) != data:
            raise ValueError(f'Invalid frozen base file: {path}')
    files['base/ablation_attention.py'] = (HERE.parents[1] / 'official_llama/ablation_attention.py').read_bytes().replace(b'\r\n', b'\n')
    hashes['ablation_attention.py'] = digest(files['base/ablation_attention.py'])
    files['base/files.json'] = json.dumps(hashes, indent=2).encode()
    for target, src in [('batch_speed.py', 'batch_speed.py'), ('batch_check.py', 'batch_check.py'),
                        ('start.sh', 'start_full_4090.sh')]:
        files[target] = (HERE / src).read_bytes().replace(b'\r\n', b'\n')
    files['SHA256SUMS'] = ''.join(f'{digest(data)}  {name}\n' for name, data in files.items()).encode()
    target = DELIVERY / 'PhaseKV_4090_batch2_6.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr('phasekv_4090_batch2_6/' + name, data)
    target.with_suffix('.tmp').replace(target)
    print(target)


if __name__ == '__main__':
    build()
