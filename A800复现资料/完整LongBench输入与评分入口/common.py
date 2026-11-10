from pathlib import Path
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
import threading
from functools import lru_cache

HERE=Path(__file__).resolve().parent
METHODS=('full','phasekv','kivi','clusterkv','snapkv','pyramidkv','million','scope')
BATCHES=(1,2,4,8)
SHAPES=('8192+256','8192+2048','32768+256','32768+2048')
BASE_MANIFEST='180f71ceff70ed6a046bd8ae7ad1f521181c8160d81d5d455194b04dfa539ef2'
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read(path,default=None):
    return json.loads(Path(path).read_text(encoding='utf8')) if Path(path).is_file() else default
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(path)
def verify(root):
    root=Path(root).resolve()
    for line in (root/'SHA256SUMS').read_text().splitlines():
        h,name=line.split('  ',1);p=(root/name).resolve()
        if not p.is_relative_to(root) or sha(p)!=h:raise ValueError('Changed file: '+str(p))
    return sha(root/'SHA256SUMS')
@lru_cache(maxsize=1)
def rows():
    with gzip.open(HERE/'data/fulltext.jsonl.gz','rt',encoding='utf8') as f:
        result=[json.loads(line) for line in f]
    assert len(result)==320 and sum(r['pilot'] for r in result)==10
    for r in result:
        assert not r['truncated_input'] and hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest()==r['input_sha256']
    return result
def path_for(out,method,row):return Path(out)/'results'/method/'quality'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
def result(out,method,row):
    r=read(path_for(out,method,row),{})
    if r and (r.get('id')!=row['id'] or r.get('input_sha256')!=row['input_sha256']):raise ValueError('Result/input mismatch: '+row['id'])
    return r
def terminal(r):return r.get('status') in ('ok','oom','unsupported')
def pending(out,method,pilot=False):return [r for r in rows() if (not pilot or r['pilot']) and not terminal(result(out,method,r))]
def weight_identity(model):
    model=Path(model)
    for name,h in read(HERE/'data/model_metadata.json').items():
        if name!='configuration.json' and sha(model/name)!=h:raise ValueError('Model metadata differs: '+name)
    index=read(model/'model.safetensors.index.json')
    return {n:[(model/n).stat().st_size,(model/n).stat().st_mtime_ns] for n in sorted(set(index['weight_map'].values()))}

@lru_cache(maxsize=4)
def speed_matrix(bench):
    with gzip.open(Path(bench)/'data/speed_rows.jsonl.gz','rt',encoding='utf8') as f:
        bank=[json.loads(line)['input_ids'] for line in f]
    if len(bank)<8 or any(len(r)<32768 for r in bank[:8]):raise ValueError('Insufficient frozen speed inputs')
    cases=[]
    for shape in SHAPES:
        length,limit=map(int,shape.split('+'))
        hashes=[hashlib.sha256(json.dumps(r[:length]).encode()).hexdigest() for r in bank[:8]]
        if len(set(hashes))!=8:raise ValueError('Speed batch must contain distinct frozen requests')
        cases.append(dict(shape=shape,input_tokens=length,max_new_tokens=limit,input_sha256=hashes))
    return cases

def speed_cases(bench,method):
    return [(r,b) for r in speed_matrix(str(bench)) for b in ((1,) if method in ('million','clusterkv') else BATCHES)]

def speed_result(old,method,case,batch):
    p=Path(old)/'results'/method/'speed'/f"{case['shape']}_B{batch}_r3.json"
    r=read(p,{})
    if not r:return r
    if r.get('shape')!=case['shape'] or r.get('batch')!=batch or r.get('input_sha256')!=case['input_sha256'][:batch]:
        raise ValueError('Speed identity differs: '+str(p))
    if r.get('status')=='ok':
        runs=r.get('runs',[])
        if len(runs)!=3 or any(x.get('batch')!=batch or x.get('input_tokens')!=case['input_tokens'] or
            x.get('output_tokens_per_row')!=case['max_new_tokens'] or x.get('decode_steps')!=case['max_new_tokens']-1 for x in runs):
            raise ValueError('Speed measurement is incomplete: '+str(p))
    return r
