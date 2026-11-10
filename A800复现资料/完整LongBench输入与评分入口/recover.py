"""Take over at saved-result boundaries; reuse valid work; run full-text quality and B1/2/4/8 speed."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import contextlib
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

import brake
from common import HERE,METHODS,BATCHES,read,write,sha,verify,now,pending,terminal,result,rows,speed_cases,speed_result
from prepare import prepare,reuse
from report import summary

def acquire(path,blocking=False):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);handle=path.open('a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:handle.close();return None
    return handle

def alive(process):
    current=brake.live(process)
    return current is not None and current['state']!='Z'

def gpu_cards(config):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
    cards={}
    for row in csv.reader(io.StringIO(text)):
        i,u,n,m=[s.strip() for s in row]
        if i in config['gpus']:
            if u!=config['gpus'][i] or 'A800' not in n or int(m)<75000:raise ValueError('Original physical GPU mapping changed')
            cards[i]=dict(uuid=u,name=n)
    if len(cards)!=4:raise ValueError('Expected original four A800 GPUs')
    return cards

def busy(uuid):
    data=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return [line for line in data.splitlines() if line.split(',')[0].strip()==uuid]

def point_stamp(out,process):
    folder=Path(out)/'results'/process['method']/('speed' if process['action']=='speed' else 'quality')
    return {p.name:p.stat().st_mtime_ns for p in folder.glob('*.json')}

def stop_group(process):
    current=brake.live(process)
    if current is None or current['state']=='Z':return
    if current['group']!=process['pid']:raise ValueError('Refusing unexpected process group')
    try:
        os.killpg(process['pid'],signal.SIGTERM)
        os.killpg(process['pid'],signal.SIGCONT)
    except ProcessLookupError:return
    deadline=time.monotonic()+10
    while alive(process) and time.monotonic()<deadline:time.sleep(.1)
    if alive(process):os.killpg(process['pid'],signal.SIGKILL)

class Recovery:
    def __init__(self,config,mode):
        self.config=config;self.out=Path(config['out']);self.old=Path(config['old_out']);self.mode=mode
        self.cards=gpu_cards(config);self.lock=threading.RLock();self.stop=threading.Event();self.children={}
        previous=read(self.out/'recovery_queue.json',{})
        self.state=dict(identity=config['manifest'],mode=mode,phase='fulltext_pilot',jobs={},lanes={},active={},
                        speed_gpu=previous.get('speed_gpu',{}),started=now())
        source=read(self.old/'queue.json',{})
        for method in METHODS:
            if method not in self.state['speed_gpu']:
                pin=source.get('methods',{}).get(method,{}).get('gpu')
                protocol=read(self.old/'results'/method/'protocol.json',{})
                if protocol:
                    actual=protocol['gpu']['uuid'].removeprefix('GPU-')
                    physical=next((i for i,u in config['gpus'].items() if u.removeprefix('GPU-')==actual),None)
                    if physical is None or pin is not None and pin!=physical:raise ValueError('Old speed protocol/GPU mismatch')
                    pin=physical
                self.state['speed_gpu'][method]=pin
        self.matrix=read(Path(config['kit'])/'matrix.json')
        self.adopted=[]
        for key,saved in previous.get('active',{}).items():
            if alive(saved):
                index=brake.script_index(saved,HERE/'quality_worker.py')
                speedindex=brake.script_index(saved,HERE/'speed_worker.py')
                oldindex=brake.script_index(saved,Path(config['bench'])/'worker.py')
                if index is None and oldindex is None and speedindex is None:raise ValueError('Unexpected saved recovery worker')
                if index is not None or speedindex is not None:
                    if Path(brake.option(saved['argv'],'--config','')).resolve()!=(self.out/'config.json').resolve():raise ValueError('Saved worker belongs to another recovery')
                self.adopted.append(dict(saved,recovery_job=key))
        for method in METHODS:
            for stage in ('pilot','full','speed','other'):
                if mode=='pilot' and stage!='pilot':continue
                key=stage+':'+method
                self.state['jobs'][key]=dict(stage=stage,method=method,status='done' if self.complete(stage,method) else 'pending')
        for p in self.adopted:
            if p['recovery_job'] in self.state['jobs']:self.state['jobs'][p['recovery_job']]['status']='running'

    def speed_done(self,method):
        for case,b in speed_cases(self.config['bench'],method):
            r=speed_result(self.old,method,case,b)
            if r.get('status') not in ('ok','oom'):return False
        return True

    def complete(self,stage,method):
        if stage in ('pilot','full'):return not pending(self.out,method,pilot=stage=='pilot')
        if stage=='speed':return self.speed_done(method)
        for case in self.matrix['quality']:
            if case['group'].startswith('longbench/'):continue
            p=self.old/'results'/method/'quality'/(hashlib.sha256(case['id'].encode()).hexdigest()+'.json')
            r=read(p,{})
            if r.get('status') not in ('ok','oom'):return False
            if r.get('input_sha256')!=case['input_sha256']:raise ValueError('Other quality input differs')
        return True

    def persist(self):
        self.state['updated']=now();write(self.out/'recovery_queue.json',self.state)

    def claim(self,gpu):
        with self.lock:
            pilots=[j for j in self.state['jobs'].values() if j['stage']=='pilot']
            pilot_open=any(j['status'] not in ('done','failed') for j in pilots)
            if not pilot_open and any(j['status']=='failed' for j in pilots):
                self.state['phase']='pilot_needs_attention';self.persist();return None,'finished'
            if not pilot_open and self.mode=='pilot':
                self.state['phase']='pilot_complete';self.persist();return None,'finished'
            self.state['phase']='fulltext_pilot' if pilot_open else 'corrected_formal'
            priorities=('pilot',) if pilot_open else ('speed','full','other')
            for stage in priorities:
                for method in METHODS:
                    key=stage+':'+method;job=self.state['jobs'].get(key)
                    if not job or job['status']!='pending':continue
                    # Existing orphan workers may have completed records since the initial scan.
                    if self.complete(stage,method):job['status']='done';continue
                    if stage in ('speed','other'):
                        pin=self.state['speed_gpu'][method]
                        if pin is not None and pin!=gpu:continue
                        self.state['speed_gpu'][method]=gpu
                    job.update(status='running',gpu=gpu,started=now());self.persist()
                    return key,'run'
            unfinished=any(j['status'] in ('pending','running') for j in self.state['jobs'].values())
            self.persist();return None,'wait' if unfinished else 'finished'

    def env(self,gpu,method,quality):
        group=method if method in ('million','clusterkv') else 'main'
        envdir=Path(self.config['envs'])/group;env=dict(os.environ)
        for name in ('PYTHONPATH','PYTHONHOME'):env.pop(name,None)
        env.update(CUDA_VISIBLE_DEVICES=self.cards[gpu]['uuid'],PHASEKV_MODEL=self.config['model'],
            A800_OUT=str((self.out if quality else self.old)/'results'),A800_ENVS=self.config['envs'],
            PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS=str(self.config['threads']),MKL_NUM_THREADS=str(self.config['threads']),OPENBLAS_NUM_THREADS='1',
            TORCH_CUDA_ARCH_LIST='8.0',VIRTUAL_ENV=str(envdir),PATH=str(envdir/'bin')+os.pathsep+env.get('PATH',''))
        return str(envdir/'bin/python'),env

    def launch(self,key,gpu,handles,options=None):
        job=self.state['jobs'][key];stage,method=job['stage'],job['method'];quality=stage in ('pilot','full')
        py,env=self.env(gpu,method,quality)
        if quality:
            command=[py,'-u',str(HERE/'quality_worker.py'),'--config',str(self.out/'config.json'),'--method',method,'--stage','pilot' if stage=='pilot' else 'all']
        elif stage=='speed' and options==['speed']:
            command=[py,'-u',str(HERE/'speed_worker.py'),'--config',str(self.out/'config.json'),'--method',method]
        else:command=[py,'worker.py',options[0],method,*options[1:]]
        logfile=self.out/'logs'/f'{stage}-{method}.log';logfile.parent.mkdir(exist_ok=True)
        print(now(),'GPU',gpu,key,options or '',flush=True)
        with logfile.open('a') as log:
            log.write('\n'+now()+' '+repr(command)+'\n');log.flush()
            child=subprocess.Popen(command,cwd=self.config['bench'],env=env,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,pass_fds=tuple(h.fileno() for h in handles))
            saved=brake.proc(child.pid)
            if saved:saved.update(gpu=self.cards[gpu]['uuid'],action='quality' if quality else options[0],method=method)
            with self.lock:
                self.children[key]=child
                if saved:self.state['active'][key]=saved
                self.state['lanes'][gpu]=key;self.persist()
            code=child.wait()
            with self.lock:
                self.children.pop(key,None);self.state['active'].pop(key,None);self.persist()
        return code

    def run_job(self,key,gpu,handles):
        job=self.state['jobs'][key];stage,method=job['stage'],job['method']
        if stage in ('pilot','full'):return self.launch(key,gpu,handles)
        directory=self.old/'results'/method
        # Preserve previous failed diagnostics before the original resumable worker retries them.
        for path in directory.rglob('*.json'):
            record=read(path,{})
            if record.get('status') in ('error','failed') or path.name.endswith('_last_error.json'):
                target=self.out/'source_error_archive'/method/str(time.time_ns())/path.relative_to(directory)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        if read(directory/'gate.json',{}).get('status')!='ok':
            code=self.launch(key,gpu,handles,['check'])
            if code:return code
        if stage=='speed':options=['speed']
        else:options=['quality','--suites','longgenbench','ruler32']
        return self.launch(key,gpu,handles,options)

    def lane(self,gpu):
        adopted=[p for p in self.adopted if p.get('gpu') in (gpu,self.cards[gpu]['uuid'])]
        while any(alive(p) for p in adopted) and not self.stop.is_set():
            with self.lock:self.state['lanes'][gpu]='waiting for saved worker boundary';self.persist()
            for process in adopted:
                if not alive(process):continue
                if process.get('needs_batch_switch') and point_stamp(self.old,process)!=process['initial_stamp']:
                    stop_group(process)
                    with self.lock:self.state.setdefault('boundary_stops',[]).append(dict(pid=process['pid'],time=now(),method=process['method']))
            self.stop.wait(.2)
        with self.lock:
            for p in adopted:
                key=p.get('recovery_job')
                if key in self.state['jobs']:
                    j=self.state['jobs'][key];j['status']='done' if self.complete(j['stage'],j['method']) else 'failed'
                    self.state['active'].pop(key,None)
        if self.stop.is_set():return
        handles=[]
        try:
            for token in (gpu,self.cards[gpu]['uuid']):
                h=acquire(Path('/tmp')/f'phasekv-a800-single-{token}.lock')
                if not h:raise RuntimeError('GPU lock remains owned by another process: '+gpu)
                handles.append(h)
            if busy(self.cards[gpu]['uuid']):raise RuntimeError('Unexpected occupied GPU; no process was stopped: '+gpu)
            while not self.stop.is_set():
                key,action=self.claim(gpu)
                if action=='finished':break
                if action=='wait':self.stop.wait(1);continue
                try:code=self.run_job(key,gpu,handles)
                except Exception as exc:
                    code=1
                    with self.lock:self.state['jobs'][key]['error']=repr(exc)
                with self.lock:
                    job=self.state['jobs'][key]
                    job.update(status='done' if code==0 and self.complete(job['stage'],job['method']) else 'failed',returncode=code,ended=now())
                    self.state['lanes'][gpu]='idle';self.persist();print(summary(self.out),flush=True)
        finally:
            for h in handles:h.close()

    def interrupt(self):
        self.stop.set()
        with self.lock:children=list(self.children.values())
        for child in children:
            saved=brake.proc(child.pid)
            if saved:stop_group(saved)

def snapshot(old):
    records={}
    for path in (old/'results').rglob('*.json'):
        r=read(path,{})
        if r.get('status') in ('ok','oom') or path.name=='protocol.json':records[str(path)]=sha(path)
    return records

def run(args):
    config=read(args.out/'config.json')
    if not config:raise ValueError('Run prepare first')
    if verify(HERE)!=config['manifest']:raise ValueError('Package changed')
    if verify(Path(config['bench']))!=config['bench_manifest']:raise ValueError('Frozen benchmark changed')
    old=Path(config['old_out']);guard=acquire(args.out/'recovery.lock')
    if not guard:raise RuntimeError('Recovery is already running')
    runner=None;held=[]
    try:
        # Complete preparation first. Only now change live scheduling.
        record=brake.brake(old,Path(config['kit']),args.append_kit)
        runner=Recovery(config,args.mode)
        preserved=snapshot(old);write(args.out/'preserved'/f'{time.time_ns()}.json',preserved)
        # SIGKILL applies only to the identified, frozen Python schedulers. Their handlers would
        # kill valid children, so leave those independent worker sessions alive for adoption.
        for parent in record['controllers']:
            current=brake.live(parent)
            if current and current['state'] not in ('T','t','Z'):raise RuntimeError('Scheduler is not frozen')
            if current and current['state']!='Z':os.kill(parent['pid'],signal.SIGKILL)
        deadline=time.monotonic()+10
        while any(alive(p) for p in record['controllers']):
            if time.monotonic()>deadline:raise RuntimeError('Scheduler did not exit')
            time.sleep(.1)
        for path in (old/'queue.lock',old/'long_append.lock'):
            h=acquire(path)
            if not h:raise RuntimeError('Another scheduler owns '+str(path))
            held.append(h)
        _,oldworkers=brake.inventory(old,Path(config['kit']),args.append_kit,Path(config['bench']))
        known={p['pid'] for p in runner.adopted}
        for process in oldworkers:
            if process['pid'] in known or process['cropped_quality'] or not alive(process):continue
            # All legacy speed loops use the superseded matrix. Switch after one saved point.
            process['needs_batch_switch']=process['action']=='speed'
            process['initial_stamp']=point_stamp(old,process)
            runner.adopted.append(process)
        record.update(status='handed_to_recovery',recovery_out=str(args.out),time=now());write(old/'quality_recovery_brake.json',record)
        write(args.out/'adopted_workers.json',runner.adopted)
        reuse(config)
        def interrupted(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(runner.lane,gpu) for gpu in config['gpus']]
            try:
                for future in as_completed(futures):future.result()
            except BaseException:
                runner.interrupt();raise
        changed=[p for p,h in preserved.items() if not Path(p).is_file() or sha(p)!=h]
        if changed:raise RuntimeError('Prior completed result changed: '+repr(changed[:5]))
        write(args.out/'preservation_check.json',dict(status='ok',checked=len(preserved),time=now()))
        with runner.lock:
            failed=any(j['status']=='failed' for j in runner.state['jobs'].values())
            runner.state['phase']='needs_attention' if failed else ('pilot_complete' if args.mode=='pilot' else 'complete');runner.persist()
        print(summary(args.out),flush=True)
        return int(failed)
    finally:
        if runner:runner.stop.set()
        for handle in held:handle.close()
        guard.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','rescue','status'))
    p.add_argument('--old-out',type=Path,default=Path('/data/phasekv-results/a800_four_gpu_v1'))
    p.add_argument('--kit',type=Path,default=Path('/data/a800_four_gpu_v1'))
    p.add_argument('--append-kit',type=Path,default=Path('/data/a800_long_append_v1'))
    p.add_argument('--out',type=Path,default=Path('/data/phasekv-results/a800_quality_fulltext_v1'))
    p.add_argument('--mode',choices=('all','pilot'),default='all');a=p.parse_args();a.out=a.out.resolve()
    if a.action in ('prepare','rescue'):prepare(a.old_out,a.kit,a.out)
    if a.action in ('run','rescue'):return run(a)
    if a.action=='status':print(summary(a.out))
    return 0

if __name__=='__main__':sys.exit(main())
