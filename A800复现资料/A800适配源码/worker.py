import argparse,collections,gc,gzip,hashlib,json,os,platform,statistics,sys,time,traceback
from pathlib import Path
import torch,fcntl
from engine import Engine,run,HERE
from transformers import AutoTokenizer
from benchmark_metrics import score

OUT=Path(os.environ['A800_OUT']);MODEL=Path(os.environ['PHASEKV_MODEL']);DATA=HERE/'data'
def digest(b):return hashlib.sha256(b).hexdigest()
def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False),encoding='utf8');tmp.replace(p)
def load(name):
    with gzip.open(DATA/name,'rt',encoding='utf8') as f:
        yield from (json.loads(l) for l in f)
def hw():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one A800 via CUDA_VISIBLE_DEVICES')
    gpu=torch.cuda.get_device_properties(0)
    if 'A800' not in gpu.name or torch.cuda.get_device_capability()!=(8,0) or gpu.total_memory<75*2**30:raise RuntimeError(f'Expected A800 80G SM80; got {gpu}')
    return dict(gpu=gpu.name,uuid=str(getattr(gpu,'uuid',os.environ.get('CUDA_VISIBLE_DEVICES'))),capacity=gpu.total_memory)
def oom_memory():
    return dict(allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated())

def quality_score(s,text):
    if s['suite']=='longbenchv2':
        import re
        m=re.findall(r'correct answer is\s*[:锛歖?\s*\(?([A-D])\b',text,re.I)
        if not m:m=re.findall(r'^\s*\(?([A-D])\)?[.!]?\s*$',text,re.M)
        return dict(score=100*int(bool(m) and m[-1].upper()==s['answers'][0]),metric='local_direct_answer_MC',parsed=m[-1] if m else None)
    return score(s,text)
def gate(e,bank,maxb):
    from batch_check import check_cooling
    from batch_backend import packed_prompt
    test=[r[:2112] for r in bank[:maxb]]
    assert len(set(tuple(row) for row in test))==maxb
    def probe(rows):
        ids=torch.tensor(rows,device='cuda');first=e.start(ids,2048,False,{})
        post=e.step(ids[:,-8:-7].contiguous());res=torch.cat([first,post],1).float().cpu()
        if res.shape[:2]!=(len(rows),2) or not torch.isfinite(res).all():raise ValueError('Invalid logits')
        e.release();return res
    independent=torch.cat([probe([row]) for row in test])
    errors={}
    for b in range(1,maxb+1):
        current=probe(test[:b]);err=((current-independent[:b]).norm(dim=-1)/independent[:b].norm(dim=-1).clamp_min(1e-6)).max().item()
        if err>.04:raise RuntimeError(f'B{b} request independence relative L2 {err:.5f}')
        errors[b]=err
    length=322 if e.name=='clusterkv' else 146
    smoke=run(e,test[:min(maxb,2)],length,[],False,{})
    if smoke['decode_steps']!=length-1:raise ValueError('Incomplete smoke')
    t=smoke['trace']
    if e.name=='phasekv':
        assert t['cold_segments']==1 and t['prompt_tokens']==2112
        smoke['packed_cooling_error']=check_cooling(packed_prompt,dict(e.options,prefill_keep_ratio=1.))
    if e.name=='kivi':assert t['key_quantized'] and t['value_quantized']
    if e.name in ('snapkv','scope'):assert t['physical_tokens']<2112+145
    if e.name=='pyramidkv':assert min(t['layer_lengths'])<max(t['layer_lengths'])<=2112+145
    if e.name=='clusterkv':
        from patch_cluster import verify_rope
        from cluster_paper import verify_rank
        smoke['rope_32k_max_error']=verify_rope(e.model)
        smoke['ranking_409_and_scan']=verify_rank()
        assert t['clusters']==2112//80+4
    smoke.pop('output_ids');e.release()
    quality_probe=run(e,[test[0]],3,[],True,{})
    smoke['quality_probe']=quality_probe['trace']
    if e.name=='million':assert quality_probe['trace']['pq_tokens']==2114
    if e.name=='phasekv':assert 0<quality_probe['trace']['prompt_tokens']<2112
    e.release()
    return dict(batch_logit_errors=errors,smoke=smoke)
def work(args):
    gpu=hw()
    # File identity and codebook are part of the resume protocol.
    manifest=json.loads((DATA/'old_manifest.json').read_text())
    for n,h in manifest['tokenizer_files'].items():
        if digest((MODEL/n).read_bytes())!=h: raise RuntimeError(f'Checkpoint metadata mismatch: {n}')
    idx=json.loads((MODEL/'model.safetensors.index.json').read_text())
    weights={n:[(MODEL/n).stat().st_size,(MODEL/n).stat().st_mtime_ns] for n in sorted(set(idx['weight_map'].values()))}
    import transformers,triton,flash_attn
    protocol=dict(phasekv='A800 SM80 MMA; quality95/speed100; four_gpu_v1',manifest=digest((HERE/'SHA256SUMS').read_bytes()),gpu=gpu,weights=weights,
        model=str(MODEL.resolve()),method=args.method,torch=str(torch.__version__),transformers=transformers.__version__,
        flash=flash_attn.__version__,triton=triton.__version__,python=platform.python_version(),speed_prompt_pct=100,quality_prompt_pct=95,execution='four independent single-GPU workers',
        omp_threads=os.environ.get('OMP_NUM_THREADS'),
        kivi_fp16_reduced_precision_reduction=False if args.method=='kivi' else None)
    if args.method=='million':protocol['codebook']=digest((OUT/'codebook/meta.json').read_bytes())
    dest=OUT/args.method
    prot=dest/'protocol.json'
    if prot.exists() and json.loads(prot.read_text())!=protocol:raise ValueError('Protocol changed. Set A800_OUT to a fresh directory.')
    write(prot,protocol)
    bank=[r['input_ids'] for r in load('speed_rows.jsonl.gz')]
    maxb=1 if args.method in ('million','clusterkv') else 8
    gatefile=dest/'gate.json'
    if args.action!='check' and (not gatefile.exists() or json.loads(gatefile.read_text())['status']!='ok'):
        raise RuntimeError('No passing gate for this method; run check first')
    torch.manual_seed(20261029);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.action=='check': write(gatefile,dict(status='running'))
    # Limit simultaneous CPU copies of the 8B checkpoint; inference remains concurrent.
    with open('/tmp/phasekv-a800-model-load.lock','a') as loading:
        fcntl.flock(loading,fcntl.LOCK_EX)
        e=Engine(args.method,MODEL)
    if args.action=='check':
        details=gate(e,bank,maxb)
        if args.method=='phasekv':
            from phase_gate import check_phase
            details['a800_phase']=check_phase(e,bank[0])
        write(gatefile,dict(status='ok',**details));print('GATE PASS',args.method);return
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    if args.action=='speed':
        def synthetic():
            for shape in args.shapes:
                length,limit=map(int,shape.split('+'))
                yield shape,[r[:length] for r in bank],limit,None
        from real_workloads import workloads
        cases=workloads(args.data_groups) if args.data_groups else synthetic()
        for shape,rows,limit,source in cases:
            for b in (args.batches or range(1,maxb+1)):
                if b>maxb:raise ValueError('Batch exceeds native method support')
                p=dest/'speed'/f'{shape}_B{b}_r{args.rounds}.json'
                if p.exists() and json.loads(p.read_text())['status'] in ('ok','oom'):continue
                progress={};measurements=[]
                try:
                    for i in range(args.rounds+1):
                        row=run(e,rows[:b],limit,[],False,progress)
                        hashes=[digest(json.dumps(x).encode()) for x in row.pop('output_ids')]
                        if i:measurements.append(dict(row,output_sha256=hashes))
                        e.release()
                    result=dict(status='ok',shape=shape,batch=b,runs=measurements,input_sha256=[digest(json.dumps(x).encode()) for x in rows[:b]],median_total_tps=statistics.median(x['aggregate_tokens_per_s'] for x in measurements))
                except torch.cuda.OutOfMemoryError as exc:result=dict(status='oom',shape=shape,batch=b,at=progress.copy(),message=str(exc),**oom_memory())
                except Exception as exc:
                    write(p,dict(status='error',shape=shape,batch=b,input_sha256=[digest(json.dumps(x).encode()) for x in rows[:b]],at=progress.copy(),message=repr(exc),traceback=traceback.format_exc()))
                    raise
                result['input_sha256']=[digest(json.dumps(x).encode()) for x in rows[:b]]
                if source is not None:result['source_rows']=source[:b]
                write(p,result);print(shape,b,result['status'],result.get('median_total_tps'),flush=True);e.release()
        return
    counts=collections.Counter()
    for file in ('legacy.jsonl.gz','quality32.jsonl.gz','longbenchv2_slice.jsonl.gz'):
        for s in load(file):
            suite='ruler32' if file=='quality32.jsonl.gz' else s['suite']
            if suite=='performance' or (args.suites and suite not in args.suites):continue
            group=suite,s['task']
            if args.limit and counts[group]>=args.limit:continue
            counts[group]+=1; ident=s['id'];p=dest/'quality'/(digest(ident.encode())+'.json')
            if p.exists() and json.loads(p.read_text())['status'] in ('ok','oom'):continue
            progress={}
            try:
                if e.name=='snapkv':
                    needle=s['suite']=='ruler' and s['task'].startswith('niah')
                    e.options.update(max_capacity_prompt=1024 if needle else 2048,window_size=16 if needle else 32,kernel_size=5 if needle else 7)
                r=run(e,[s['input_ids']],s['max_new_tokens'],set(s['stop_ids']),True,progress)
                ids=r.pop('output_ids')[0];text=tokenizer.decode(ids,skip_special_tokens=True)
                result=dict(r,status='ok',id=ident,suite=s['suite'],task=s['task'],input_sha256=s['input_sha256'],
                    output_ids=ids,prediction=text,**quality_score(s,text))
            except torch.cuda.OutOfMemoryError as exc:result=dict(status='oom',id=ident,suite=s['suite'],task=s['task'],at=progress.copy(),message=str(exc),**oom_memory())
            except Exception as exc:
                write(p,dict(status='error',id=ident,suite=s['suite'],task=s['task'],input_sha256=s['input_sha256'],at=progress.copy(),message=repr(exc),traceback=traceback.format_exc()))
                raise
            result['input_sha256']=s['input_sha256']
            result['input_tokens']=len(s['input_ids'])
            write(p,result);e.release()
            print(ident,result['status'],result.get('score'),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['check','speed','quality']);a.add_argument('method')
    a.add_argument('--rounds',type=int,choices=[1,3],default=1)
    a.add_argument('--batches',type=int,nargs='+',choices=list(range(1,9)))
    shapes=['9192+256','8192+2048','32768+256','32768+2048']
    from real_workloads import SUITES
    group=a.add_mutually_exclusive_group();group.add_argument('--shapes',nargs='+',choices=shapes,default=shapes)
    group.add_argument('--data-groups',nargs='+',choices=SUITES+['all']);a.add_argument('--limit',type=int)
    a.add_argument('--suites',nargs='+',choices=['longbench','ruler','ruler32','longgenbench','longbenchv2'])
    args=a.parse_args()
    try:
        with torch.inference_mode():work(args)
        (OUT/args.method/(args.action+'_last_error.json')).unlink(missing_ok=True)
    except Exception as exc:
        if args.action=='check':write(OUT/args.method/'gate.json',dict(status='failed',error=repr(exc)))
        write(OUT/args.method/(args.action+'_last_error.json'),dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise
