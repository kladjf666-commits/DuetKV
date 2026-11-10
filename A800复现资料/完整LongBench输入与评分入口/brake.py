"""Freeze only the v1 schedulers; stop cropped LongBench workers; leave speed workers running."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import signal
import time

def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().split(') ',1)[1].split()
        argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        return dict(pid=int(pid), start=stat[19], state=stat[0], ppid=int(stat[1]),
                    group=int(stat[2]), argv=argv, cwd=str((p/'cwd').resolve()))
    except (OSError,ValueError,IndexError):return None

def live(saved):
    current=proc(saved['pid'])
    return current if current and current['start']==saved['start'] else None

def script_index(process, path):
    for i,arg in enumerate(process['argv'][1:],1):
        if arg.startswith('-'):continue
        candidate=Path(arg)
        if not candidate.is_absolute():candidate=Path(process['cwd'])/candidate
        if candidate.resolve()==path.resolve():return i
    return None

def option(argv, key, default):
    if key in argv:return argv[argv.index(key)+1]
    return next((x.split('=',1)[1] for x in argv if x.startswith(key+'=')),default)

def inventory(out,kit,append_kit,bench):
    controllers=[];workers=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        p=proc(path.name)
        if not p:continue
        for script in (kit/'campaign.py',append_kit/'append.py'):
            index=script_index(p,script)
            if index is not None and 'run' in p['argv'][index+1:]:
                selected=Path(option(p['argv'][index+1:],'--out','/data/phasekv-results/a800_four_gpu_v1')).resolve()
                if selected==out:controllers.append(p)
        index=script_index(p,bench/'worker.py')
        if index is None:continue
        try:
            env=dict(x.split(b'=',1) for x in (path/'environ').read_bytes().split(b'\0') if b'=' in x)
            if Path(env.get(b'A800_OUT',b'').decode()).resolve()!=out/'results':continue
            args=p['argv'][index+1:]
            if len(args)<2 or args[0] not in ('check','speed','quality'):continue
            suites=[]
            if '--suites' in args:
                for arg in args[args.index('--suites')+1:]:
                    if arg.startswith('--'):break
                    suites.append(arg)
            p.update(action=args[0],method=args[1],gpu=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode(),
                     cropped_quality=args[0]=='quality' and (not suites or 'longbench' in suites))
            workers.append(p)
        except (OSError,ValueError):continue
    return controllers,workers

def brake(out,kit,append_kit):
    out=out.resolve();layout=read(out/'layout.json');bench=Path(layout['bench']).resolve()
    if not (bench/'worker.py').is_file():raise ValueError('Missing original worker')
    controllers,_=inventory(out,kit,append_kit,bench)
    journal=out/'quality_recovery_brake.json'
    previous=read(journal)
    saved={p['pid']:p for p in previous.get('controllers',[]) if live(p)}
    for p in controllers:saved[p['pid']]=p
    record=dict(time=datetime.now(timezone.utc).isoformat(),out=str(out),bench=str(bench),
                controllers=list(saved.values()),status='freezing_schedulers')
    write(journal,record)
    for p in saved.values():
        if live(p):
            try:os.kill(p['pid'],signal.SIGSTOP)
            except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while any((live(p) or {}).get('state') not in (None,'T','t','Z') for p in saved.values()):
        if time.monotonic()>deadline:raise RuntimeError('Scheduler did not freeze; inspect journal')
        time.sleep(.1)
    _,workers=inventory(out,kit,append_kit,bench)
    record.update(workers=workers,status='stopping_cropped_quality');write(journal,record)
    stopped=[]
    for p in workers:
        if not p['cropped_quality']:continue
        current=live(p)
        if current and current['group']!=p['pid']:
            raise RuntimeError('Unexpected worker process group; refusing broad signal')
        if current:
            try:os.killpg(p['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
            stopped.append(p)
    deadline=time.monotonic()+10
    while any((live(p) or {}).get('state') not in (None,'Z') for p in stopped) and time.monotonic()<deadline:
        time.sleep(.2)
    for p in stopped:
        current=live(p)
        if current and current['state']!='Z':
            try:os.killpg(p['pid'],signal.SIGKILL)
            except ProcessLookupError:pass
    record.update(status='braked',stopped_cropped_quality=[p['pid'] for p in stopped],
                  preserved_workers=[dict(pid=p['pid'],method=p['method'],action=p['action'],gpu=p['gpu']) for p in workers if not p['cropped_quality']])
    write(journal,record)
    print(json.dumps(record,indent=2))
    print('BRAKED: old schedulers frozen; cropped LongBench workers stopped; speed and other quality workers left running.')
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('/data/phasekv-results/a800_four_gpu_v1'))
    parser.add_argument('--kit',type=Path,default=Path('/data/a800_four_gpu_v1'))
    parser.add_argument('--append-kit',type=Path,default=Path('/data/a800_long_append_v1'))
    args=parser.parse_args();brake(args.out,args.kit,args.append_kit)
