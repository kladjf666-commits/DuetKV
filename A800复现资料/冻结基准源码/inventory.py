"""Inspect the exact matrix, with no Python packages, model, or GPU required."""
from collections import Counter
from pathlib import Path
import gzip,hashlib,json
HERE=Path(__file__).resolve().parent
MAIN='full phasekv kivi snapkv scope pyramidkv'.split()
NEW=['million','clusterkv']
SHAPES=['9192+256','8192+2048','32768+256','32768+2048']
def sha(b):return hashlib.sha256(b).hexdigest()
def rows(name):
    with gzip.open(HERE/'data'/name,'rt',encoding='utf8') as f:
        yield from map(json.loads,f)
def collect(check=False):
    ids={};counts=Counter();lengths={}
    for file in ['legacy.jsonl.gz','quality32.jsonl.gz','longbenchv2_slice.jsonl.gz']:
        for r in rows(file):
            if r['suite']=='performance':continue
            suite='ruler32' if file=='quality32.jsonl.gz' else r['suite'];key=suite+'/'+r['task'];ident=r['id'];size=len(r['input_ids'])
            assert ident not in ids and r['answers'] and size+r['max_new_tokens']<=131072
            if check:assert sha(json.dumps(r['input_ids']).encode())==r['input_sha256']
            ids[ident]=key;counts[key]+=1;lo,hi=lengths.get(suite,(size,size));lengths[suite]=(min(lo,size),max(hi,size))
    assert len(ids)==1150
    return ids,counts,lengths
if __name__=='__main__':
    for l in (HERE/'SHA256SUMS').read_text().splitlines():
        h,n=l.split('  ',1)
        if sha((HERE/n).read_bytes())!=h:raise ValueError('Corrupt '+n)
    ids,c,ls=collect(True)
    pool=list(rows('speed_rows.jsonl.gz'))
    assert len(pool)==8 and len(set(r['input_sha256'] for r in pool))==8
    for r in pool:assert len(r['input_ids'])==32768 and sha(json.dumps(r['input_ids']).encode())==r['input_sha256']
    wiki=HERE/'data/wikitext2_train.parquet';raw=wiki.read_bytes();assert raw[:4]==raw[-4:]==b'PAR1'
    print(json.dumps(dict(main_b1_to_8=MAIN,separate_b1=NEW,shapes=SHAPES,quality_rows_per_method=len(ids),quality_counts=c,input_length_range=ls,calibration='WikiText-2 raw train, 65536 vectors per K/V codebook'),indent=2))
