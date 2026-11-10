"""Package one new condition and the input-matched, completed original70 records."""
import gzip
import json
import zipfile

from build import DELIVERY, HERE, digest


def build():
    result_dir = DELIVERY / '4090_question_coverage_v1'
    previous = json.loads((result_dir/'summary.json').read_text(encoding='utf-8'))
    if previous['status']!='complete' or previous['paired']['all']['complete_pairs']!=48:
        raise ValueError('Expected a completed 48-prompt reference')
    protocol = previous['protocol']
    if digest(json.dumps(protocol,sort_keys=True).encode()) != previous['fingerprint']:
        raise ValueError('Invalid baseline protocol fingerprint')
    with zipfile.ZipFile(DELIVERY/'PhaseKV_4090_question_coverage.zip') as source:
        prefix = 'phasekv_4090_question_coverage/'
        manifest = source.read(prefix+'SHA256SUMS')
        if digest(manifest)!=protocol['package_sha256']:
            raise ValueError('Source package differs from the previous quality test')
        files = {}
        for line in manifest.decode().splitlines():
            checksum, name = line.split('  ',1)
            data = source.read(prefix+name)
            if digest(data)!=checksum:
                raise ValueError(f'Corrupt frozen file {name}')
            if name!='coverage_policy.py':
                files[name]=data
    cases = json.loads(gzip.decompress(files['quality_samples.json.gz']))
    by_id = {s['id']:s for s in cases}
    rows = [r for r in previous['records'] if r['arm']=='original70']
    if len(rows)!=48 or {r['id'] for r in rows}!=set(by_id):
        raise ValueError('Missing original70 measurements')
    for r in rows:
        sample = by_id[r['id']]
        path = result_dir/'records'/(r['id'].replace('/','_')+'_original70.json')
        raw = json.loads(path.read_text(encoding='utf-8'))
        if (r['fingerprint']!=previous['fingerprint'] or any(raw[k]!=v for k,v in r.items())
                or digest(json.dumps(raw['output_ids']).encode())!=r['output_sha256']
                or sample['input_sha256']!=r['input_sha256']
                or r['retained_per_head']!=round(.7*len(sample['input_ids']))):
            raise ValueError(f'Invalid baseline for {r["id"]}')
    ref = dict(fingerprint=previous['fingerprint'],gpu=protocol['gpu'],environment=protocol['environment'],
               runtime_sha256=protocol['code_sha256'],options=protocol['options'],records=rows)
    files['original70_reference.json']=json.dumps(ref,separators=(',',':')).encode()
    design = json.loads(files['quality_manifest.json'])
    design.update(generation_arm='prompt100',policy='Retain all prompt positions at 4 bits. '
                  'Original70 is an archived reference. Only prompt100 produces new generations.')
    files['quality_manifest.json']=json.dumps(design,indent=2).encode()
    for name in ('start.sh','quality_compare.py'):
        files[name]=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
    files['SHA256SUMS']=''.join(f'{digest(b)}  {n}\n' for n,b in files.items()).encode()
    target=DELIVERY/'PhaseKV_4090_prompt100_quality_v2.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():
            z.writestr('phasekv_4090_prompt100_quality/'+name,data)
    target.with_suffix('.tmp').replace(target)
    print(target,target.stat().st_size)


if __name__=='__main__':
    build()
