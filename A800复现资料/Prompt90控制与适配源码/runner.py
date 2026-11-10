#!/usr/bin/env python3
from pathlib import Path
from contextlib import ExitStack
import argparse,csv,io,os,signal,subprocess,sys,time,tarfile
import common as c
from report import report

def sources(cfg,rows,longgen,cases,old_protocol):
    pins=c.read(c.HERE/'source_hashes.json');protected={};reuse=[];refs={'quality':{},'speed':{}}
    refine=Path(cfg['refine_out'])
    for name,h in pins['refine_files'].items():
        path=refine/name
        if c.sha(path)!=h:raise ValueError('Refinement source changed: '+str(path))
        protected[str(path)]=h
    protocol=c.read(refine/'protocol.json')
    if (protocol['package_manifest']!=pins['refine_manifest'] or protocol['weights']!=cfg['weights'] or
        protocol['model']!=cfg['model'] or any(v!=old_protocol[k] for k,v in protocol['versions'].items())):raise ValueError('Refinement environment/model changed')
    for r in longgen:
        path=refine/'results'/c.profile()['id']/'quality'/c.key(r['id']);v=c.read(path)
        if (v['status']!='ok' or v['profile']!=c.profile() or v['effective_options']!=c.options(c.profile()) or
            v['experiment']!=protocol['experiment'] or any(v[k]!=x for k,x in c.meta(r).items())):raise ValueError('Invalid reusable LongGenBench record')
        reuse.append(dict(c.meta(r),source=str(path),sha256=c.sha(path),source_experiment=v['experiment'],source_gpu=v['gpu'],role='parameter_selection_reused'))
    for method in ('full','phasekv'):
        refs['quality'][method]={};refs['speed'][method]={}
        for r in rows+longgen:
            pin=pins['quality'][method][r['id']];path=Path(cfg[pin['root']])/pin['path']
            if c.sha(path)!=pin['sha256']:raise ValueError('Quality baseline changed: '+str(path))
            protected[str(path)]=pin['sha256'];v=c.read(path)
            if v['input_sha256']!=r['input_sha256'] or v['input_tokens']!=len(r['input_ids']) or v['status']!='ok':raise ValueError('Baseline input/status changed')
            fields=('id','suite','task','status','score','input_sha256','input_tokens','output_tokens_per_row','final_kv_bytes')
            refs['quality'][method][r['id']]=dict({k:v[k] for k in fields},source=str(path),sha256=pin['sha256'],
                estimate_s=v.get('wall_s') or v['ttft_s']+v['decode_s'])
        for case in cases:
            rel=f"results/{method}/speed/{case['shape']}_B{case['batch']}_r3.json";path=Path(cfg['old_out'])/rel
            h=pins['speed'][method][rel]
            if c.sha(path)!=h:raise ValueError('Speed baseline changed: '+str(path))
            protected[str(path)]=h;v=c.read(path)
            if v['input_sha256']!=case['input_sha256'] or v['batch']!=case['batch'] or v['status']!='ok' or len(v['runs'])!=3:raise ValueError('Invalid old speed baseline')
            refs['speed'][method][f"{case['shape']}_B{case['batch']}"]=dict(case,median_total_tps=v['median_total_tps'],
                estimate_s=4*sum(x['ttft_s']+x['decode_s'] for x in v['runs'])/3,source=str(path),sha256=h,
                gpu=c.read(Path(cfg['old_out'])/f'results/{method}/protocol.json')['gpu'])
    return reuse,refs,protected

def prepare(args):
    cfg,longgen,original=c.source_check(args)
    cfg=dict(cfg,recovery_kit=str(args.recovery_kit),refine_out=str(args.refine_out))
    for p in [c.HERE]+[Path(cfg[k]) for k in ('out','old_out','bench','kit','model','envs','recovery_kit','refine_out')]:
        p=p.resolve()
        if args.out==p or args.out.is_relative_to(p) or p.is_relative_to(args.out):raise ValueError('New results must use an isolated directory')
    rows=c.quality_rows(cfg);cases=c.speed_cases(cfg);reuse,refs,protected=sources(cfg,rows,longgen,cases,original)
    identity=dict(package_manifest=c.sha(c.HERE/'SHA256SUMS'),config_sha256=c.sha(args.config),profile=c.profile(),
        model=cfg['model'],weights=cfg['weights'],source_manifest=c.BASE,recovery_manifest=c.RECOVERY,
        quality=[dict(c.meta(r),pilot=r.get('pilot',False)) for r in rows],reused_longgen=reuse,
        speed=cases,speed_rounds=3,speed_warmup=1,quality_batch=1,reference_digest=c.digest(refs))
    plan=dict(identity,experiment=c.digest(identity),source_config=cfg,source_protocol=original)
    args.out.mkdir(parents=True,exist_ok=True)
    with c.lock(args.out/'runner.lock'):
        for name,obj in [('plan.json',plan),('references.json',refs),('preserved.json',protected)]:
            old=c.read(args.out/name)
            if old is not None and old!=obj:raise ValueError('Existing output identity differs: '+name)
            if old is None:c.write(args.out/name,obj,new=True)
        for item in reuse:
            path=args.out/'reused/longgenbench'/c.key(item['id']);path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                # Raw source remains byte-for-byte intact. Atomically publish the local reuse copy.
                tmp=path.with_name(path.name+f'.{time.time_ns()}.tmp');tmp.write_bytes(Path(item['source']).read_bytes())
                os.replace(tmp,path)
            if c.sha(path)!=item['sha256']:raise ValueError('Reused result changed')
        for row in rows:c.quality_record(args.out,row)
        for case in cases:
            for i in (1,2,3):c.speed_record(args.out,case,i)
            c.finish_speed(args.out,case)
    print(report(args.out),flush=True);return plan

def run(args):
    if sys.platform!='linux':raise RuntimeError('Run on the preserved A800 Linux server')
    plan=prepare(args);cfg=plan['source_config']
    if c.read(args.out/'summary.json')['complete']:print('All quality and speed results complete; no model loaded.');return
    with ExitStack() as stack:
        paths=[args.out/'runner.lock',Path(cfg['refine_out'])/'runner.lock',Path(cfg['out'])/'recovery.lock',
            Path(cfg['old_out'])/'queue.lock',Path(cfg['old_out'])/'long_append.lock',Path(cfg['old_out']).parent/'a800_phasekv_longgen_sweep_v1/runner.lock']
        handles=[stack.enter_context(c.lock(p)) for p in paths];gpu=select_gpu(args.gpu)
        for ident in (gpu['uuid'],str(args.gpu)):handles.append(stack.enter_context(c.lock(Path('/tmp')/('phasekv-a800-single-'+ident+'.lock'))))
        old=c.read(args.out/'gpu.json')
        if old and old['uuid']!=gpu['uuid']:raise ValueError('Resume GPU UUID differs; keep saved files and review before changing cards')
        if not old:c.write(args.out/'gpu.json',gpu,new=True)
        py,env=environment(cfg,gpu['uuid'],args.out);child=None
        def stop(signum,frame):raise KeyboardInterrupt('Interrupted; completed records retained')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            with (args.out/'run.log').open('a',encoding='utf-8') as log:
                child=subprocess.Popen([str(py),'-u',str(c.HERE/'gpu_worker.py'),'--out',str(args.out),'--uuid',gpu['uuid']],
                    cwd=cfg['bench'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(h.fileno() for h in handles))
                code=child.wait();child=None
            c.preserved(c.read(args.out/'preserved.json'));print(report(args.out),flush=True)
            if code:raise RuntimeError('Worker failed; see '+str(args.out/'run.log'))
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()

def export(args):
    plan=c.read(args.out/'plan.json')
    if not plan:raise ValueError('No prepared results')
    with c.lock(args.out/'runner.lock'):
        c.verify(c.HERE,plan['package_manifest']);c.preserved(c.read(args.out/'preserved.json'));report(args.out)
        archive=args.out.parent/f'a800_phasekv_p90_final_results_{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()%1000000:06}.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            tar.add(args.out,arcname=args.out.name)
            tar.add(c.HERE,arcname='test_package',filter=lambda x:None if '__pycache__' in x.name or x.name.endswith('.pyc') else x)
        archive.with_name(archive.name+'.sha256').write_bytes((c.sha(archive)+'  '+archive.name+'\n').encode('ascii'))
    print(archive)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','status','export'))
    p.add_argument('--config',type=Path,default=Path('/data/phasekv-results/a800_quality_fulltext_v1/config.json'))
    p.add_argument('--recovery-kit',type=Path,default=Path('/data/a800_quality_fulltext_v1'))
    p.add_argument('--refine-out',type=Path,default=Path('/data/phasekv-results/a800_phasekv_refine_v2'))
    p.add_argument('--out',type=Path,default=Path('/data/phasekv-results/a800_phasekv_p90_final_v1'))
    p.add_argument('--gpu',default='0');a=p.parse_args()
    for k in ('config','recovery_kit','refine_out','out'):setattr(a,k,getattr(a,k).resolve())
    if a.action=='status':print(report(a.out,save=False))
    else:{'prepare':prepare,'run':run,'export':export}[a.action](a)

def smi(query):
    text=subprocess.check_output(['nvidia-smi',query,'--format=csv,noheader,nounits'],text=True)
    return [[s.strip() for s in row] for row in csv.reader(io.StringIO(text)) if row]

def select_gpu(index):
    cards={i:dict(uuid=u,name=n,memory_mib=int(float(m)),used_mib=int(float(used))) for i,u,n,m,used in smi('--query-gpu=index,uuid,name,memory.total,memory.used')}
    card=cards.get(str(index))
    if not card or 'A800' not in card['name'] or card['memory_mib']<75000:raise ValueError('Requires one A800 80GB')
    if card['used_mib']>512 or any(u==card['uuid'] for u,p in smi('--query-compute-apps=gpu_uuid,pid')):raise RuntimeError('Selected GPU occupied; no process was stopped')
    return card

def environment(cfg,uuid,out):
    env=dict(os.environ)
    for k in ('PYTHONPATH','PYTHONHOME'):env.pop(k,None)
    venv=Path(cfg['envs'])/'main';py=venv/'bin/python'
    if not py.is_file():raise ValueError('Preserved main environment missing: '+str(py))
    env.update(CUDA_VISIBLE_DEVICES=uuid,PHASEKV_MODEL=cfg['model'],A800_OUT=str(out/'native_unused'),
        VIRTUAL_ENV=str(venv),PATH=str(venv/'bin')+os.pathsep+env.get('PATH',''),
        PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS=str(cfg['threads']),
        MKL_NUM_THREADS=str(cfg['threads']),OPENBLAS_NUM_THREADS='1',TORCH_CUDA_ARCH_LIST='8.0')
    return py,env

if __name__=='__main__':main()
