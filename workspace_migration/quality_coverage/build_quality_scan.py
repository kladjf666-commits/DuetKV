"""Freeze the two measured B1 endpoints, scorer and GPU runtime for the retention sweep."""
import gzip
import json
import zipfile

from build import DELIVERY, HERE, digest
from retention_curve import GRID, ORDER, long_repeat


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def build():
    folders = {70: DELIVERY/'4090_question_coverage_v1', 100: DELIVERY/'4090_prompt100_quality_v1'}
    runs = {pct: load(d/'summary.json') for pct, d in folders.items()}
    before, after = (runs[p]['protocol'] for p in (70,100))
    if (runs[100]['generated_arms'] != ['prompt100'] or after['baseline_fingerprint'] != runs[70]['fingerprint']
            or after['options'] != dict(before['options'],prefill_keep_ratio=1.)
            or before['options']['prefill_keep_ratio'] != .7
            or any(before[k] != after[k] for k in ('gpu','model','environment','code_sha256','batch',
                                                   'dtype','greedy','natural_eos','max_output'))):
        raise ValueError('Incompatible archived endpoints')
    files = {}
    with zipfile.ZipFile(DELIVERY/'PhaseKV_4090_prompt100_quality.zip') as archive:
        prefix = 'phasekv_4090_prompt100_quality/'
        manifest = archive.read(prefix+'SHA256SUMS')
        if digest(manifest) != after['package_sha256']:
            raise ValueError('Archive is not the tested quality package')
        for line in manifest.decode().splitlines():
            sha, name = line.split('  ',1)
            data = archive.read(prefix+name)
            if digest(data) != sha:
                raise ValueError(f'Corrupt {name}')
            if (name.startswith(('base/','vendor/')) or name in ('quality_samples.json.gz','benchmark_metrics.py')):
                files[name] = data
    if digest(files['base/files.json']) != after['code_sha256']:
        raise ValueError('Frozen model runtime mismatch')
    by_id = {s['id']: s for s in json.loads(gzip.decompress(files['quality_samples.json.gz']))}
    records = []
    for pct, run in runs.items():
        arm = 'original70' if pct == 70 else 'prompt100'
        rows = [r for r in run['records'] if r['arm'] == arm]
        if (run['status'] != 'complete' or digest(json.dumps(run['protocol'],sort_keys=True).encode()) != run['fingerprint']
                or len(rows) != 48 or {r['id'] for r in rows} != set(by_id)):
            raise ValueError(f'Incomplete or invalid {pct}% reference')
        for r in rows:
            key = r['id']
            s = by_id[key]
            raw = load(folders[pct]/'records'/(key.replace('/','_')+'_'+arm+'.json'))
            if (r['fingerprint'] != run['fingerprint'] or any(raw[k]!=v for k,v in r.items())
                    or s['input_sha256'] != r['input_sha256'] or r['task'] != s['task']
                    or r['retained_per_head'] != round(len(s['input_ids'])*(pct/100))
                    or r['total_questions'] != len(s['answers'])
                    or digest(json.dumps(raw['output_ids']).encode()) != r['output_sha256']):
                raise ValueError(f'Unmatched reference: {arm} {key}')
            records.append(dict(r,ratio_pct=pct,long_repeat=long_repeat(raw['output_ids'])))
    reference = dict(grid=list(GRID),order=list(ORDER),gpu=after['gpu'],environment=after['environment'],
                     code_sha256=after['code_sha256'],options=before['options'],
                     tokenizer_sha256=after['manifest']['tokenizer_sha256'],
                     fingerprints={p:r['fingerprint'] for p,r in runs.items()},records=records)
    files['scan_reference.json']=json.dumps(reference,separators=(',',':')).encode()
    for name in ('start.sh','quality_compare.py','quality_scan.py','retention_curve.py'):
        files[name]=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
    files['SHA256SUMS']=''.join(f'{digest(data)}  {name}\n' for name,data in files.items()).encode()
    target=DELIVERY/'PhaseKV_4090_prompt_quality_scan.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():
            z.writestr('phasekv_4090_prompt_quality_scan/'+name,data)
    target.with_suffix('.tmp').replace(target)
    print(target, target.stat().st_size)


if __name__=='__main__':
    build()
