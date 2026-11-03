"""Package two fixed-length speed cases on the completed prompt100 quality runtime."""
import gzip
import json
import zipfile

from build_full_4090 import DELIVERY, HERE, digest


def speed_inputs(base_ids):
    frozen = HERE.parents[1] / 'official_llama/bench_assets/frozen'
    corpus = frozen / 'samples.jsonl.gz'
    checksum = digest(corpus.read_bytes())
    if checksum != json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))['samples_sha256']:
        raise ValueError('Frozen speed corpus checksum mismatch')
    sources = [('performance/16384+256/0', 9192, 256), ('performance/8192+2048/0', 8192, 2048)]
    wanted = {source for source, _, _ in sources}
    with gzip.open(corpus, 'rt', encoding='utf-8') as stream:
        rows = {r['id']: r for line in stream if (r := json.loads(line))['id'] in wanted}
    samples = []
    for source, length, output in sources:
        row = rows[source]
        ids = row['input_ids'][:length]
        if (digest(json.dumps(row['input_ids']).encode()) != row['input_sha256']
                or len(ids) != length or ids[:len(base_ids)] != base_ids
                or row['natural_eos'] or row['max_new_tokens'] != output):
            raise ValueError(f'Unexpected frozen speed input: {source}')
        shape = f'{length}+{output}'
        ident = f'performance/{shape}/0'
        samples.append(dict(row, id=ident, task=shape, cluster_id=ident, length_budget=length,
                            input_ids=ids, input_sha256=digest(json.dumps(ids).encode()),
                            source_sample_id=source))
    return json.dumps(dict(frozen_samples_sha256=checksum, cases=samples), separators=(',', ':')).encode()


def build():
    run = json.loads((DELIVERY / '4090_prompt100_quality_v1/summary.json').read_text(encoding='utf-8'))
    protocol = run['protocol']
    if (run['status'] != 'complete' or run['generated_arms'] != ['prompt100']
            or digest(json.dumps(protocol, sort_keys=True).encode()) != run['fingerprint']):
        raise ValueError('Expected the completed prompt100 quality reference')
    with zipfile.ZipFile(DELIVERY / 'PhaseKV_4090_prompt100_quality.zip') as archive:
        prefix = 'phasekv_4090_prompt100_quality/'
        if digest(archive.read(prefix + 'SHA256SUMS')) != protocol['package_sha256']:
            raise ValueError('Quality archive differs from the completed run')
        manifest = archive.read(prefix + 'base/files.json')
        if digest(manifest) != protocol['code_sha256']:
            raise ValueError('Quality runtime manifest mismatch')
        files = {'base/files.json': manifest}
        hashes = json.loads(manifest)
        for name, checksum in hashes.items():
            data = archive.read(prefix + 'base/' + name)
            if digest(data) != checksum:
                raise ValueError(f'Corrupt quality runtime file {name}')
            files['base/' + name] = data
    # The 256-output driver asserted 255 steps. Use the length-aware driver
    # from the 2048-output benchmark and keep all timed model/kernel code frozen.
    files['base/quick.py'] = (HERE / 'quick.py').read_bytes().replace(b'\r\n', b'\n')
    hashes['quick.py'] = digest(files['base/quick.py'])
    files['base/files.json'] = json.dumps(hashes, indent=2).encode()
    files['speed_inputs.json'] = speed_inputs(json.loads(files['base/sample.json'])['input_ids'])
    prior = {key: protocol[key] for key in ('environment', 'code_sha256', 'options')}
    prior.update(gpu_uuid=protocol['gpu'].split(',')[0], quality_fingerprint=run['fingerprint'])
    prior['speed_runtime_sha256'] = digest(files['base/files.json'])
    files['quality_reference.json'] = json.dumps(prior, indent=2).encode()
    for name, source in [('start.sh', 'start_full_4090.sh'), ('full_speed.py', 'full_speed.py'),
                         ('prompt100_speed.py', 'prompt100_speed.py')]:
        files[name] = (HERE / source).read_bytes().replace(b'\r\n', b'\n')
    files['SHA256SUMS'] = ''.join(f'{digest(data)}  {name}\n' for name, data in files.items()).encode()
    target = DELIVERY / 'PhaseKV_4090_prompt100_speed_2cases.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr('phasekv_4090_prompt100_speed_2cases/' + name, data)
    target.with_suffix('.tmp').replace(target)
    print(target, target.stat().st_size)


if __name__ == '__main__':
    build()
