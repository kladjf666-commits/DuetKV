"""Package the B-local experiment with the exact frozen B runtime and paired controls."""
import gzip
import json
import zipfile
from build import DELIVERY, HERE, digest


def build():
    run = DELIVERY/'4090_block70_v1'
    summary = json.loads((run/'summary.json').read_bytes())
    protocol = json.loads((run/'protocol.json').read_bytes())
    fp = protocol['fingerprint']
    if (summary['status'] != 'complete' or summary['completed_prompts'] != 48 or summary['fingerprint'] != fp
            or digest(json.dumps(protocol['protocol'], sort_keys=True).encode()) != fp):
        raise ValueError('Expected complete and consistent B results')
    files = {}
    with zipfile.ZipFile(DELIVERY/'PhaseKV_4090_block70.zip') as src:
        prefix = 'phasekv_4090_block70/'
        manifest = src.read(prefix+'SHA256SUMS')
        if digest(manifest) != protocol['protocol']['package']:
            raise ValueError('B runtime package mismatch')
        for line in manifest.decode().splitlines():
            sha, name = line.split('  ', 1)
            data = src.read(prefix+name)
            if digest(data) != sha:
                raise ValueError(f'Corrupt {name}')
            files[name] = data
    ref = json.loads(gzip.decompress(files.pop('block_reference.json.gz')))
    for key, controls in ref['controls'].items():
        item = json.loads((run/'records'/(key.replace('/', '_')+'.json')).read_bytes())
        b = item['arms']['block70']
        if (item['id'] != key or item['fingerprint'] != fp or b['arm'] != 'block70'
                or b['input_sha256'] != controls['prompt70']['input_sha256']
                or b['retained_per_head'] != controls['prompt70']['retained_per_head']
                or any(item['arms'][a] != r for a,r in controls.items())
                or digest(json.dumps(b['output_ids']).encode()) != b['output_sha256']):
            raise ValueError(f'Invalid B record: {key}')
        controls['block70'] = {field:b[field] for field in controls['prompt70']}
    ref['fingerprints']['block70'] = fp
    ref['block70_gpu'] = protocol['protocol']['gpu']
    files['local_reference.json.gz'] = gzip.compress(json.dumps(ref).encode(), mtime=0)
    for name in ('future_oracle.py', 'block_local.py', 'start.sh'):
        files[name] = (HERE/name).read_bytes().replace(b'\r\n', b'\n')
    files['SHA256SUMS'] = ''.join(f'{digest(data)}  {name}\n' for name,data in files.items()).encode()
    target = DELIVERY/'PhaseKV_4090_block_local70.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():
            z.writestr('phasekv_4090_block_local70/'+name, data)
    target.with_suffix('.tmp').replace(target)
    print(target, target.stat().st_size)


if __name__ == '__main__':
    build()
