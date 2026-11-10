"""CPU-only input, identity, resume and selection helpers."""
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip, hashlib, json, os, statistics, time

HERE=Path(__file__).resolve().parent
BASE='180f71ceff70ed6a046bd8ae7ad1f521181c8160d81d5d455194b04dfa539ef2'
RECOVERY='9273fa97c5b8717d99178e3d2bc04d9abae0edadafed8f6a08f7b59e24629fdf'
FOUR='fb686e32bb4243ffcd48484d74488337531e44f3e5b27655dab0f9d5b3033475'
OPT=dict(fused_quantize=True,reuse_workspace=True,paged_cold=True,page_slots=32,cold_group=8,
         specialized_4bit=True,gqa_reuse=True,hot_ring=True,channel_contiguous_k=True,a800_prompt_mma=True)

def now():return datetime.now(timezone.utc).isoformat()

def read(p,default=None):return json.loads(Path(p).read_text(encoding='utf-8')) if Path(p).is_file() else default

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def key(x):return hashlib.sha256(x.encode()).hexdigest()+'.json'

def write(p,x,*,new=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        with tmp.open('x',encoding='utf-8') as f:
            json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        if new and os.name=='nt':os.rename(tmp,p)  # Windows rename refuses an existing target.
        elif new:os.link(tmp,p)  # Linux atomic create without replacing a completed record.
        else:os.replace(tmp,p)
    finally:tmp.unlink(missing_ok=True)

def verify(root,expected=None):
    root=Path(root).resolve();actual=sha(root/'SHA256SUMS')
    if expected and actual!=expected:raise ValueError('Manifest differs: '+str(root))
    for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        h,name=line.split('  ',1);p=(root/name).resolve()
        if not p.is_relative_to(root) or sha(p)!=h:raise ValueError('File changed: '+str(p))
    return actual

@contextmanager
def lock(p):
    import fcntl
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Runner/GPU lock busy: '+str(p))
        try:yield f
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def options(p):
    return dict({k:v for k,v in p.items() if k!='id'},group_size=32,lazy_g=16,sink_tokens=0,optimizations=dict(OPT))

def load_rows(cfg):
    selected={r['id']:r for r in read(Path(cfg['kit'])/'matrix.json')['quality'] if r['group'].startswith('longgenbench/')}
    rows=[]
    with gzip.open(Path(cfg['bench'])/'data/legacy.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['id'] not in selected:continue
            s=selected[r['id']]
            h=hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest()
            assert r['suite']=='longgenbench' and not r.get('truncated_input',False) and r['natural_eos']
            assert h==r['input_sha256']==s['input_sha256'] and len(r['input_ids'])==s['input_tokens']
            assert r['max_new_tokens']==s['max_new_tokens']==4096
            rows.append(r)
    assert len(rows)==len(selected)==48 and len({r['id'] for r in rows})==48
    assert Counter(r['task'] for r in rows)==Counter(gsm8k=16,mmlu=16,csqa=16)
    # Interleave tasks. All profiles use exactly the same full 48 inputs.
    return [r for i in range(16) for task in ('gsm8k','mmlu','csqa') for r in [sorted([s for s in rows if s['task']==task],key=lambda s:s['id'])[i]]]

def meta(row):return {**{k:row[k] for k in ('id','suite','task','input_sha256','max_new_tokens')},'input_tokens':len(row['input_ids'])}

def source_check(args):
    cfg=read(args.config)
    if not cfg or cfg['manifest']!=RECOVERY or cfg['bench_manifest']!=BASE:raise ValueError('Need preserved full-text recovery config')
    verify(HERE);verify(args.recovery_kit,RECOVERY);verify(cfg['bench'],BASE);verify(cfg['kit'],FOUR)
    model=Path(cfg['model'])
    for name,h in read(args.recovery_kit/'data/model_metadata.json').items():
        if name!='configuration.json' and sha(model/name)!=h:raise ValueError('Model metadata differs: '+name)
    idx=read(model/'model.safetensors.index.json')
    weights={n:[(model/n).stat().st_size,(model/n).stat().st_mtime_ns] for n in sorted(set(idx['weight_map'].values()))}
    if weights!=cfg['weights']:raise ValueError('Preserve checkpoint files and timestamps')
    original=read(Path(cfg['old_out'])/'results/phasekv/protocol.json')
    if original['weights']!=weights or original['manifest']!=BASE or Path(original['model']).resolve()!=model.resolve():raise ValueError('Prior PhaseKV protocol differs')
    if read(Path(cfg['old_out'])/'results/phasekv/gate.json')['status']!='ok':raise ValueError('Prior PhaseKV gate missing')
    return cfg,load_rows(cfg),original

def preserved(saved):
    for p,h in saved.items():
        if not Path(p).is_file() or sha(p)!=h:raise ValueError('A previous result changed: '+p)

def profile():return read(HERE/'profile.json')

def quality_rows(cfg):
    with gzip.open(Path(cfg['recovery_kit'])/'data/fulltext.jsonl.gz','rt',encoding='utf-8') as f:
        lb=[json.loads(x) for x in f]
    with gzip.open(Path(cfg['bench'])/'data/quality32.jsonl.gz','rt',encoding='utf-8') as f:
        ruler=[json.loads(x) for x in f]
    selected={x['id']:x for x in read(Path(cfg['kit'])/'matrix.json')['quality'] if x['group'].startswith('ruler32/')}
    assert len(lb)==320 and len(ruler)==len(selected)==64 and sum(r['pilot'] for r in lb)==10
    for r in lb+ruler:
        assert not r.get('truncated_input',False)
        assert hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest()==r['input_sha256']
        assert r['stop_ids'] and r['max_new_tokens']>0
    for r in ruler:
        s=selected[r['id']]
        assert r['natural_eos'] and s['input_sha256']==r['input_sha256'] and s['input_tokens']==len(r['input_ids']) and s['max_new_tokens']==r['max_new_tokens']
    return sorted(lb,key=lambda r:not r['pilot'])+ruler

def speed_bank(cfg):
    with gzip.open(Path(cfg['bench'])/'data/speed_rows.jsonl.gz','rt',encoding='utf-8') as f:
        bank=[json.loads(next(f))['input_ids'] for _ in range(8)]
    assert all(len(r)>=32768 for r in bank)
    return bank

def speed_cases(cfg):
    bank=speed_bank(cfg);cases=[]
    for n,t in ((8192,256),(8192,2048),(32768,256),(32768,2048)):
        hashes=[hashlib.sha256(json.dumps(r[:n]).encode()).hexdigest() for r in bank]
        assert len(set(hashes))==8
        for b in (1,2,4,8):cases.append(dict(shape=f'{n}+{t}',batch=b,input_tokens=n,max_new_tokens=t,input_sha256=hashes[:b]))
    return cases

def quality_path(out,row):return Path(out)/'results/phasekv/quality'/key(row['id'])

def quality_record(out,row,allow_error=False):
    p=quality_path(out,row);v=read(p,{})
    if v:
        plan=read(Path(out)/'plan.json')
        if v.get('experiment')!=plan['experiment'] or v.get('profile')!=profile() or any(v.get(k)!=row[k] for k in ('id','input_sha256','max_new_tokens')):
            raise ValueError('Quality identity changed: '+str(p))
        if v.get('status') not in (('ok','oom','error') if allow_error else ('ok','oom')):raise ValueError('Saved error needs review: '+str(p))
    return v

def speed_path(out,case,index=None):
    suffix=f"{case['shape']}_B{case['batch']}_r{index if index is not None else 3}.json"
    return Path(out)/'results/phasekv'/('speed_runs' if index is not None else 'speed')/suffix

def speed_record(out,case,index=None,allow_error=False):
    path=speed_path(out,case,index);v=read(path,{})
    if v:
        plan=read(Path(out)/'plan.json')
        if v.get('experiment')!=plan['experiment'] or v.get('profile')!=profile() or any(v.get(k)!=x for k,x in case.items()):raise ValueError('Speed identity changed: '+str(path))
        if v.get('status') not in (('ok','oom','error') if allow_error else ('ok','oom')):raise ValueError('Saved speed error needs review: '+str(path))
        if v['status']=='ok':
            for r in ([v] if index else v['runs']):
                if r['output_tokens_per_row']!=case['max_new_tokens'] or r['decode_steps']!=case['max_new_tokens']-1:raise ValueError('Speed output shortened')
            if index is None and len(v['runs'])!=3:raise ValueError('Expected 3 formal runs')
    return v

def finish_speed(out,case):
    saved=speed_record(out,case)
    if saved:return saved
    runs=[speed_record(out,case,i) for i in (1,2,3)]
    if any(r.get('status')=='oom' for r in runs):
        bad=next(r for r in runs if r.get('status')=='oom')
        result=dict(case,experiment=read(Path(out)/'plan.json')['experiment'],profile=profile(),status='oom',runs=[r for r in runs if r],gpu=bad['gpu'])
    elif all(r.get('status')=='ok' for r in runs):
        result=dict(case,experiment=read(Path(out)/'plan.json')['experiment'],profile=profile(),status='ok',runs=runs,gpu=runs[0]['gpu'])
        for key_name,field in [('median_total_tps','aggregate_tokens_per_s'),('median_sequence_tps','per_sequence_tokens_per_s'),('median_ttft_s','ttft_s'),('median_final_kv_bytes','final_kv_bytes')]:result[key_name]=statistics.median(r[field] for r in runs)
        result['max_peak_allocated_bytes']=max(r['peak_allocated_bytes'] for r in runs)
    else:return {}
    write(speed_path(out,case),result,new=True);return result
