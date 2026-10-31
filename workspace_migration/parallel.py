"""Single host: per-device gates, matched timing trials and parallel quality shards.

Installed alongside attention_ablation.py as workspace_parallel.py. No tensor
parallelism or CPU offload. Each worker still loads a complete FP16 model.
"""
import argparse
import concurrent.futures as cf
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid

HERE = Path(__file__).resolve().parent


def inventory():
    import torch
    driver = ctypes.CDLL('libcuda.so.1')
    if driver.cuInit(0):
        raise RuntimeError('CUDA driver unavailable')
    result = []
    for i in range(torch.cuda.device_count()):
        dev = ctypes.c_int()
        raw = (ctypes.c_ubyte * 16)()
        if driver.cuDeviceGet(ctypes.byref(dev), i) or driver.cuDeviceGetUuid_v2(ctypes.byref(raw), dev):
            raise RuntimeError('Cannot identify visible GPU')
        prop = torch.cuda.get_device_properties(i)
        gpu = 'GPU-'+str(uuid.UUID(bytes=bytes(raw)))
        ver = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()
        result.append(dict(uuid=gpu, name=prop.name, total_memory=prop.total_memory,
                           sm=[prop.major, prop.minor], driver=ver, torch=torch.__version__, cuda=torch.version.cuda))
    return result


def plan(rows, count, parallel_performance=False):
    # Keep all four methods and repeats of a prompt together. Round-robin
    # within each task/budget to avoid concentrating all long prompts on one GPU.
    offsets = {}
    result = {}
    perf_index = 0
    for s in sorted(rows, key=lambda s: s['id']):
        if s['suite'] == 'performance':
            result[s['id']] = perf_index % count if parallel_performance else 0
            perf_index += 1
            continue
        key = (s['suite'], s['task'], s.get('length_budget'))
        pos = offsets.get(key, 0)
        result[s['id']] = pos % count
        offsets[key] = pos + 1
    return result


def worker(file):
    import attention_ablation as b
    job = json.loads(Path(file).read_text())
    fn = b.samples
    assignment = json.loads((Path(job['out'])/'assignment.json').read_text())
    manifest = json.loads((Path(job['out'])/'run_manifest.json').read_text())
    p = manifest['protocol']['parallel']
    if hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()!=p['assignment_sha256']:
        raise RuntimeError('Shard assignment changed')
    gpu = p['devices'][job['lane']]['uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise RuntimeError('Wrong GPU for this shard')
    def subset():
        rows, meta = fn()
        return [s for s in rows if assignment[s['id']]==job['lane']], meta
    b.samples = subset
    def snapshot():
        import time
        r = subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader'], capture_output=True, text=True)
        return {'time':time.time(),'csv':r.stdout.strip(),'error':r.stderr.strip()}
    b.gpu_snapshot = snapshot
    b.worker(file)


def capacity(model_path, folder):
    """One real frozen prompt per input length and arm; no speed claims."""
    import gc
    import torch
    import attention_ablation as b
    from ablation_runtime import load_model
    from ablation_generate import generate
    gate = json.loads((folder/'gpu_gate.json').read_text())
    manifest = json.loads((folder/'run_manifest.json').read_text())
    if gate.get('status')!='passed' or gate.get('protocol_hash')!=manifest['protocol_hash']:
        raise RuntimeError('Pilot requires a successful gate for this hardware and protocol')
    torch.manual_seed(20261029)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows,_ = b.samples()
    probes = [next(s for s in rows if s['suite']=='performance' and s['task']==shape)
              for shape in ('8192+256','16384+256','32000+256')]
    report = []
    for method in b.METHODS:
        model,opts,_ = load_model(model_path,method)
        for s in probes:
            print(f'Capacity pilot {method}/{s["task"]}',flush=True)
            row = dict(method=method,shape=s['task'],sample=s['id'],input_sha256=s['input_sha256'])
            try:
                result = generate(model,method,opts,s)
                if result['output_tokens']!=s['max_new_tokens']: raise ValueError('Pilot output length mismatch')
                row.update(status='ok',peak_allocated_bytes=result['peak_allocated_bytes'],
                           peak_reserved_bytes=result['peak_reserved_bytes'])
                del result
            except torch.OutOfMemoryError as exc:
                row.update(status='oom',reason=str(exc))
            report.append(row)
            b.atomic(folder.parent.parent/'capacity.json',dict(protocol_hash=manifest['protocol_hash'],
                     timing_excluded=True,expected=12,records=report))
            gc.collect(); torch.cuda.empty_cache()
        del model
        gc.collect(); torch.cuda.empty_cache()
    print('Capacity pilot: '+str(sum(r['status']=='ok' for r in report))+'/12 succeeded; see capacity.json')


def main():
    import fcntl
    import attention_ablation as b
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpus', type=int, default=2, help='First N devices visible in this allocation, 1 to 8')
    ap.add_argument('--model', default='/data/Llama-3.1-8B-Instruct/snapshots/master')
    ap.add_argument('--out', type=Path, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--gate-only', action='store_true')
    mode.add_argument('--capacity-only', action='store_true', help='1 GPU, gates + 12 real long-input probes; no formal timings')
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--parallel-performance', action='store_true',
                    help='Shard fixed-length prompts over identical GPUs; each GPU runs one worker at a time.')
    args = ap.parse_args()
    out = args.out.resolve()
    if not 1 <= args.gpus <= 8:
        ap.error('--gpus must be between 1 and 8')
    if args.capacity_only and args.gpus!=1: ap.error('Use --gpus 1 for a rental capacity pilot')
    children = set()
    mutex = threading.Lock()
    cancelled = threading.Event()
    def stop_all():
        cancelled.set()
        with mutex:
            procs = list(children)
        for p in procs:
            if p.poll() is None: p.terminate()
        for p in procs:
            try: p.wait(timeout=12)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
    def interrupt(*unused):
        stop_all()
        raise KeyboardInterrupt
    def launch(cmd, env, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a',encoding='utf-8') as f:
            with mutex:
                if cancelled.is_set(): raise RuntimeError('Coordinator is stopping')
                p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                children.add(p)
            try:
                if p.wait(): raise RuntimeError(f'Child failed; see {log}')
            finally:
                with mutex: children.discard(p)
    def concurrently(fn, indices):
        with cf.ThreadPoolExecutor(max_workers=args.gpus) as pool:
            futures = [pool.submit(fn, i) for i in indices]
            try:
                for f in cf.as_completed(futures): f.result()
            except BaseException:
                stop_all()
                raise
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with contextlib.ExitStack() as stack:
        for path in ('/data/phasekv-native-validation.lock','/data/phasekv-native-benchmark.lock'):
            lock = stack.enter_context(open(path,'a+'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        devices = json.loads(subprocess.check_output([sys.executable,__file__,'inventory'],text=True))[:args.gpus]
        if len(devices)!=args.gpus or len({d['uuid'] for d in devices})!=args.gpus:
            raise RuntimeError('Requested number of distinct visible GPUs is unavailable')
        if len({(d['name'], d['total_memory'], tuple(d['sm']), d['driver']) for d in devices})!=1:
            raise RuntimeError('Use identical GPU models, memory profiles and drivers for this experiment')
        for d in devices:
            busy = subprocess.check_output(['nvidia-smi','-i',d['uuid'],'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if busy: raise RuntimeError(f"GPU {d['uuid']} is busy: {busy}")
        out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()) and not (out/'run_manifest.json').is_file():
            raise RuntimeError('Use a new --out directory, or a previous parallel campaign')
        if (out/'run_manifest.json').exists() and 'parallel' not in json.loads((out/'run_manifest.json').read_text())['protocol']:
            raise RuntimeError('Cannot mix serial and parallel campaigns')
        subprocess.run([sys.executable,HERE/'verify_attention_ablation_bundle.py'],check=True)
        import tempfile
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            subprocess.run([sys.executable,HERE/'attention_ablation.py','--model',args.model,'--out',tmp,'--prepare-only'],check=True)
            manifest = json.loads((Path(tmp)/'run_manifest.json').read_text())
        rows, _ = b.samples()
        assignment = plan(rows, args.gpus, args.parallel_performance)
        protocol = manifest['protocol']
        protocol['code'][Path(__file__).name] = b.digest(Path(__file__).read_bytes())
        timing_policy = ('concurrent GPUs; all arms and repeats of each prompt on one device'
                         if args.parallel_performance else 'exclusive lane 0; three matched repeats')
        protocol['parallel'] = dict(devices=devices, performance=timing_policy,
                                    assignment_sha256=b.digest(json.dumps(assignment,sort_keys=True).encode()))
        h = b.digest(json.dumps(protocol,sort_keys=True).encode())
        manifest['protocol_hash'] = h
        mf = out/'run_manifest.json'
        if mf.exists() and json.loads(mf.read_text())['protocol_hash']!=h:
            raise RuntimeError('Hardware, code, samples or shard layout changed; use a fresh --out')
        b.atomic(mf,manifest)
        (out/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        def lane(i): return out/'lanes'/str(i)
        def env(i): return {**os.environ,'CUDA_VISIBLE_DEVICES':devices[i]['uuid'], 'PYTHONUNBUFFERED':'1',
                            'TRITON_CACHE_DIR':str(out/'jit'/str(i)), 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                            'TOKENIZERS_PARALLELISM':'false'}
        def gate(i):
            folder = lane(i)
            b.atomic(folder/'run_manifest.json',manifest)
            b.atomic(folder/'assignment.json',assignment)
            dest = folder/'gpu_gate.json'
            b.atomic(dest,{'status':'running','protocol_hash':h})
            launch([sys.executable,HERE/'ablation_validate.py','--model',args.model,'--out',dest,'--protocol-hash',h],env(i),folder/'gpu_gate.log')
            result = json.loads(dest.read_text())
            if result['status']!='passed' or result['protocol_hash']!=h: raise RuntimeError('GPU gate failed')
        print(f'Running {args.gpus} independent GPU gates. Logs: {out}/lanes/*/gpu_gate.log',flush=True)
        concurrently(gate,range(args.gpus))
        if args.gate_only: print('All GPU gates passed; no formal timings collected.'); return 0
        if args.capacity_only:
            launch([sys.executable,__file__,'capacity',args.model,str(lane(0))],env(0),lane(0)/'capacity.log')
            report=json.loads((out/'capacity.json').read_text())
            ok=sum(r['status']=='ok' for r in report['records'])
            print(f'Capacity pilot: {ok}/12 succeeded; {out}/capacity.json',flush=True)
            return 0 if ok==12 else 2
        def measure(i,suites):
            folder = lane(i)
            for suite in suites:
                for method in b.METHODS:
                    todo = any(assignment[s['id']]==i and b.pending(b.result_path(folder,m,s,r),args.retry_failed)
                               for m,s,r in b.expected(rows,[suite],[method]))
                    if not todo: continue
                    job = folder/'jobs'/f'{suite}__{method}.json'
                    b.atomic(job,dict(out=str(folder),model=args.model,suite=suite,method=method,lane=i,
                                      protocol_hash=h,retry_failed=args.retry_failed))
                    log = folder/'logs'/f'{suite}__{method}.log'
                    print(f'GPU {i}: {suite}/{method} -> {log}',flush=True)
                    launch([sys.executable,__file__,'worker',job],env(i),log)
        ordinary_path = b.result_path
        def summarize_shards(destination, samples, suites):
            b.result_path = lambda _,m,s,r: ordinary_path(lane(assignment[s['id']]),m,s,r)
            try:
                return b.summarize(destination,samples,suites,b.METHODS)
            finally:
                b.result_path = ordinary_path
        try:
            if args.parallel_performance:
                concurrently(lambda i: measure(i,['performance']),range(args.gpus))
            else:
                measure(0,['performance'])
            # Publish speed comparisons as soon as all timed workers finish.
            perf_rows = [s for s in rows if s['suite']=='performance']
            summarize_shards(out/'performance_first',perf_rows,['performance'])
            print(f"Performance results ready: {out/'performance_first/ablation_pairs.json'}",flush=True)
            concurrently(lambda i: measure(i,['longbench','ruler','longgenbench']),range(args.gpus))
        finally:
            stop_all()
            status = summarize_shards(out,rows,b.SUITES)
        print('Parallel campaign: '+json.dumps(dict(status)),flush=True)
        return 2 if status['pending'] or status['error'] or status['oom'] else 0


if __name__=='__main__':
    if sys.argv[1:2]==['inventory']: print(json.dumps(inventory()))
    elif sys.argv[1:2]==['worker']: worker(sys.argv[2])
    elif sys.argv[1:2]==['capacity']: capacity(sys.argv[2],Path(sys.argv[3]))
    else: raise SystemExit(main())
