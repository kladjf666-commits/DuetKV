"""Build the offline future-access retest from the verified retention sweep."""
import gzip
import json
import zipfile
from build import DELIVERY, HERE, ROOT, digest


def build(*, block_budget=False):
    with zipfile.ZipFile(DELIVERY/'PhaseKV_4090_prompt_quality_scan.zip') as src:
        prefix = 'phasekv_4090_prompt_quality_scan/'
        manifest = src.read(prefix+'SHA256SUMS')
        scan = json.loads((DELIVERY/'4090_prompt_quality_scan/summary.json').read_text(encoding='utf8'))
        if scan['status'] != 'complete' or digest(manifest) != scan['protocol']['package_sha256']:
            raise ValueError('Source package differs from the completed retention scan')
        files = {}
        for line in manifest.decode().splitlines():
            sha, name = line.split('  ', 1)
            if name.startswith(('base/', 'vendor/')) or name in (
                    'quality_samples.json.gz', 'benchmark_metrics.py', 'quality_compare.py', 'scan_reference.json'):
                data = src.read(prefix+name)
                if digest(data) != sha:
                    raise ValueError(f'Corrupt source: {name}')
                files[name] = data
    ref = json.loads(files.pop('scan_reference.json'))
    if (scan['protocol']['reference_fingerprints'] != ref['fingerprints']
            or not scan['protocol']['same_gpu_as_endpoints']
            or scan['protocol']['base_options'] != ref['options']
            or scan['fingerprint'] != digest(json.dumps(scan['protocol'], sort_keys=True).encode())):
        raise ValueError('The archived controls have incompatible protocols')
    by_id = {s['id']: s for s in json.loads(gzip.decompress(files['quality_samples.json.gz']))}
    controls = {key: {} for key in by_id}
    rows = ref.pop('records') + [r for r in scan['records'] if r['ratio_pct'] == 85]
    fields = ('id', 'task', 'input_sha256', 'output_sha256', 'output_tokens', 'retained_per_head',
              'score', 'indexed_accuracy_diagnostic', 'completion_pct', 'severe_loop')
    for r in rows:
        key, pct = r['id'], r['ratio_pct']
        sample = by_id[key]
        arm = f'prompt{pct}'
        expected_name = 'original70' if pct == 70 else arm
        expected_fp = scan['fingerprint'] if pct == 85 else ref['fingerprints'][str(pct)]
        if (arm in controls[key] or r['arm'] != expected_name or r['fingerprint'] != expected_fp
                or r['input_sha256'] != sample['input_sha256'] or r['task'] != sample['task']
                or r['retained_per_head'] != round(len(sample['input_ids'])*(pct/100))):
            raise ValueError(f'Unmatched historical control: {key} {arm}')
        controls[key][arm] = {k: r[k] for k in fields}
        if pct == 100 and not block_budget:
            path = DELIVERY/'4090_prompt100_quality_v1/records'/(key.replace('/', '_')+'_prompt100.json')
            raw = json.loads(path.read_text(encoding='utf8'))
            ids = raw['output_ids']
            if (any(raw[k] != r[k] for k in fields) or raw['fingerprint'] != expected_fp
                    or len(ids) != r['output_tokens'] or digest(json.dumps(ids).encode()) != r['output_sha256']):
                raise ValueError(f'Corrupt archived teacher trajectory: {key}')
            controls[key]['teacher_ids'] = ids
    fields = {'prompt70','prompt85','prompt100'}
    if not block_budget:
        fields.add('teacher_ids')
    if len(controls) != 48 or any(set(c) != fields for c in controls.values()):
        raise ValueError('Incomplete paired controls')
    ref.update(controls=controls, scan_fingerprint=scan['fingerprint'])
    reference = 'block_reference.json.gz' if block_budget else 'oracle_reference.json.gz'
    files[reference] = gzip.compress(json.dumps(ref, separators=(',', ':')).encode(), mtime=0)
    experiment = 'block70.py' if block_budget else 'future_attention.py'
    for name in ('future_oracle.py', experiment, 'start.sh'):
        files[name] = (HERE/name).read_bytes().replace(b'\r\n', b'\n')
    if block_budget:
        files['block_budget.py'] = (ROOT/'PhaseKV/block_budget.py').read_bytes().replace(b'\r\n', b'\n')
    files['SHA256SUMS'] = ''.join(f'{digest(b)}  {n}\n' for n,b in files.items()).encode()
    label = 'block70' if block_budget else 'future70'
    target = DELIVERY/f'PhaseKV_4090_{label}.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as dst:
        for name, data in files.items():
            dst.writestr(f'phasekv_4090_{label}/'+name, data)
    target.with_suffix('.tmp').replace(target)
    print(target, target.stat().st_size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-budget', action='store_true', help='Package only the 128-token block budget arm')
    build(block_budget=parser.parse_args().block_budget)
