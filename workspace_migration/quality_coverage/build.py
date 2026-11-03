"""Build the question coverage comparison on the validated RTX 4090 runtime."""
import collections
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DELIVERY=ROOT/'PhaseKV与Full消融'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    previous=json.loads((DELIVERY/'4090_full_vs_phasekv_li8K4ePQ/summary.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(DELIVERY/'PhaseKV_4090_full_vs_phasekv.zip') as z:
        prefix='phasekv_4090_full_vs_phasekv/base/'
        manifest=z.read(prefix+'files.json')
        if previous['status']!='complete' or digest(manifest)!=previous['protocol']['code_sha256']:
            raise RuntimeError('Expected the validated 4090 runtime')
        files={'base/files.json':manifest}
        for name,sha in json.loads(manifest).items():
            data=z.read(prefix+name)
            if digest(data)!=sha:
                raise RuntimeError(f'Corrupt runtime file {name}')
            files['base/'+name]=data
    frozen=ROOT/'official_llama/bench_assets/frozen'
    index=json.loads((frozen/'manifest.json').read_text(encoding='utf-8'))
    corpus=frozen/'samples.jsonl.gz'
    if digest(corpus.read_bytes())!=index['samples_sha256']:
        raise RuntimeError('Frozen corpus checksum mismatch')
    with gzip.open(corpus,'rt',encoding='utf-8') as f:
        samples=[r for line in f if (r:=json.loads(line))['suite']=='longgenbench']
    if collections.Counter(s['task'] for s in samples)!={'csqa':16,'gsm8k':16,'mmlu':16}:
        raise ValueError('Incorrect tasks')
    audit=json.loads((ROOT/'results/longgen_quality_probe/archive_audit.json').read_text(encoding='utf-8'))['rows']
    samples.sort(key=lambda s:(not audit[s['id']]['keep80']['severe'],s['task'],int(s['document_id'])))
    data=dict(frozen_samples_sha256=index['samples_sha256'],ordered_ids=[s['id'] for s in samples],
              tokenizer_sha256={k:v for k,v in index['tokenizer_files'].items()
                                if k in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')},
              policy='Content tokens: ceil(2*n/3) or at least 6 for stems and 3 for choices, capped at n. '
                     'Protect last 32. Equal 70% budget in each head and layer. Only visible prompt defines spans.')
    assert 'tokenizer.json' in data['tokenizer_sha256']
    files['quality_manifest.json']=json.dumps(data,indent=2).encode()
    files['quality_samples.json.gz']=gzip.compress(json.dumps(samples,separators=(',',':')).encode(),mtime=0)
    for name,path in [('quality_compare.py',HERE/'quality_compare.py'),('coverage_policy.py',HERE/'coverage_policy.py'),
                      ('start.sh',HERE/'start.sh'),('benchmark_metrics.py',ROOT/'official_llama/benchmark_metrics.py'),
                      ('vendor/SCOPE/eval_longgenbench.py',ROOT/'official_llama/vendor/SCOPE/eval_longgenbench.py')]:
        files[name]=path.read_bytes().replace(b'\r\n',b'\n')
    files['SHA256SUMS']=''.join(f'{digest(b)}  {n}\n' for n,b in files.items()).encode()
    target=DELIVERY/'PhaseKV_4090_question_coverage_v2.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
        for name,b in files.items():
            z.writestr('phasekv_4090_question_coverage/'+name,b)
    target.with_suffix('.tmp').replace(target)
    print(target,target.stat().st_size)


if __name__=='__main__':
    build()
