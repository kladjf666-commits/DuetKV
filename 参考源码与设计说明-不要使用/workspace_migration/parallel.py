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

# Previous FP16 Llama-3.1-8B, batch=1 native 4090D benchmark: failures out of
# nine records per shape (three frozen inputs x three repeats).
PRIOR_OOM = {
    '16384+256': {'full': 9, 'phasekv': 0},
    '16384+4096': {'full': 9, 'phasekv': 0},
    '32000+256': {'full': 9, 'phasekv': 9},
}
PRIOR_PROTOCOL = '7a7191ec4bdbc3c05cd640be19fb04aa1e277ab06ba2a8a4a73c0d4567cb9bb0'


def prior_oom_shape(sample):
    return sample['suite']=='performance' and sample['task'] in PRIOR_OOM


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
        if 'prior_oom_filter' in manifest['protocol']:
            rows = [s for s in rows if not prior_oom_shape(s)]
        selected = job.get('sample_ids')
        return [s for s in rows if assignment[s['id']]==job['lane']
                and (selected is None or s['id'] in selected)], meta
    b.samples = subset
    def snapshot():
        import time
        r = subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader'], capture_output=True, text=True)
        return {'time':time.time(),'csv':r.stdout.strip(),'error':r.stderr.strip()}
    previous_snapshot = b.gpu_snapshot
    b.gpu_snapshot = snapshot
    try:
        b.worker(file)
    finally:
        b.samples, b.gpu_snapshot = fn, previous_snapshot


def profile_decode(model, method, options, sample, out):
    """Separate unmeasured pass; record steps 144..159 including cold-tier updates."""
    import torch
    import ablation_runtime as runtime
    import attention_ablation as b
    from ablation_generate import generate
    from torch.profiler import profile, ProfilerActivity
    from torch.autograd import DeviceType
    length = len(sample['input_ids'])
    original = runtime.forward
    trace = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    active = False
    finished = False
    def traced(model, ids, cache, start, method, **kwargs):
        nonlocal active, finished
        if start == length+143 and ids.shape[1]==1:
            torch.cuda.synchronize()
            trace.start()
            active = True
        elif active and start == length+159:
            torch.cuda.synchronize()
            trace.stop()
            active, finished = False, True
        return original(model, ids, cache, start, method, **kwargs)
    runtime.forward = traced
    try:
        generate(model, method, options, sample)
    finally:
        runtime.forward = original
        if active: trace.stop()
    if not finished: raise RuntimeError('Decode profiling window was not reached')
    trace.export_chrome_trace(str(out/f'{method}.trace.json'))
    kernels = [event for event in trace.events() if event.device_type == DeviceType.CUDA]
    totals = {'attention_core':0., 'unpack_segment':0., 'other_device_work':0.}
    for event in kernels:
        name = event.name.lower()
        kind = ('unpack_segment' if '_unpack_segment' in name else 'attention_core'
                if any(key in name for key in ('flash_fwd','_segmented_attention','_merge_segments'))
                else 'other_device_work')
        totals[kind] += event.device_time_total
    b.atomic(out/f'{method}.json', {'sample':sample['id'],'timing_excluded':True,
             'decode_steps':[144,159], 'cuda_event_count':len(kernels),
             'status':'ok' if kernels else 'cuda_events_unavailable',
             'kernel_time_us_by_name':totals,'dram_bandwidth_measured':False,
             'notes':'Summed device-event time, not end-to-end latency. Projections and copies are outside attention_core.'})


def first_workers(spec_path):
    """Run the unchanged formal worker for each arm with the same loaded weights."""
    import ablation_runtime as runtime
    import attention_ablation as b
    spec = json.loads(Path(spec_path).read_text())
    config = json.loads((HERE/'config.json').read_text())['methods']
    original_load = runtime.load_model
    model = None
    def shared_load(path, method):
        nonlocal model
        if model is None:
            model, _, _ = original_load(path, 'full_fa2')
        for layer in model.model.layers:
            layer.self_attn.method = method
        return model, config[runtime.cache_method(method)], ['ablation_attention.py','ablation_kernels.py']
    runtime.load_model = shared_load
    try:
        for filename in spec['jobs']:
            worker(filename)
        if spec['profile']:
            job = json.loads(Path(spec['jobs'][0]).read_text())
            folder = Path(job['out'])
            target = folder/'decode_profiles'
            target.mkdir(exist_ok=True)
            sample = next(s for s in b.samples()[0] if s['id']==job['sample_ids'][0])
            for method in b.METHODS:
                if not all(b.result_path(folder,method,sample,r).is_file() and
                           not b.pending(b.result_path(folder,method,sample,r),True) for r in range(3)):
                    continue
                shared, options, _ = shared_load(job['model'],method)
                try:
                    profile_decode(shared,method,options,sample,target)
                except RuntimeError as error:
                    b.atomic(target/f'{method}.json',dict(status='profile_failed',timing_excluded=True,error=str(error)))
                    print(f'Optional CUDA trace failed for {method}: {error}',flush=True)
    finally:
        runtime.load_model = original_load


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
    volume = HERE.parent.parent
    ap.add_argument('--model', default=str(volume/'Llama-3.1-8B-Instruct/snapshots/master'))
    ap.add_argument('--out', type=Path, required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--gate-only', action='store_true')
    mode.add_argument('--capacity-only', action='store_true', help='1 GPU, gates + 12 real long-input probes; no formal timings')
    mode.add_argument('--first-8k', action='store_true', help='Measure and summarize 8192+256 first, then continue the full suite')
    mode.add_argument('--only-8k', action='store_true', help='Stop after the first 36 records; resume later using --first-8k')
    ap.add_argument('--profile-8k', action='store_true', help='Collect separate decode traces after the first 8K measurements')
    ap.add_argument('--retry-failed', action='store_true')
    ap.add_argument('--skip-prior-oom', action='store_true',
                    help='Exclude three historical 4090D OOM performance shapes from all four arms')
    ap.add_argument('--parallel-performance', action='store_true',
                    help='Shard fixed-length prompts over identical GPUs; each GPU runs one worker at a time.')
    args = ap.parse_args()
    out = args.out.resolve()
    if not 1 <= args.gpus <= 8:
        ap.error('--gpus must be between 1 and 8')
    if args.capacity_only and args.gpus!=1: ap.error('Use --gpus 1 for a rental capacity pilot')
    if args.capacity_only and args.skip_prior_oom: ap.error('Capacity pilot includes the excluded long shapes')
    first_8k = args.first_8k or args.only_8k
    if first_8k and args.gpus != 1: ap.error('First-8K mode requires the same single GPU for all arms')
    if args.profile_8k and not first_8k: ap.error('--profile-8k requires --only-8k or --first-8k')
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
        for name in ('phasekv-native-validation.lock', 'phasekv-native-benchmark.lock'):
            path = volume/name
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
        skipped = []
        protocol = manifest['protocol']
        if args.skip_prior_oom:
            excluded = [s for s in rows if prior_oom_shape(s)]
            rows = [s for s in rows if not prior_oom_shape(s)]
            skipped = [dict(method=m,id=s['id'],suite=s['suite'],shape=s['task'],repeat=r,
                            input_sha256=s['input_sha256'],status='skipped_prior_oom_shape')
                       for m,s,r in b.expected(excluded,['performance'],b.METHODS)]
            if len(skipped)!=108:
                raise RuntimeError('Frozen inputs do not match the historical OOM exclusion plan')
            protocol['prior_oom_filter'] = {
                'historical_run':'native_llama_20261029','historical_protocol_hash':PRIOR_PROTOCOL,
                'historical_oom_counts_out_of_9':PRIOR_OOM,'excluded_for_all_arms':True,
                'excluded_performance_records':len(skipped),
                'planned_records':sum(1 for _ in b.expected(rows,b.SUITES,b.METHODS)),
            }
        assignment = plan(rows, args.gpus, args.parallel_performance)
        protocol['code'][Path(__file__).name] = b.digest(Path(__file__).read_bytes())
        timing_policy = ('concurrent GPUs; all arms and repeats of each prompt on one device'
                         if args.parallel_performance else 'exclusive lane 0; three matched repeats')
        protocol['parallel'] = dict(devices=devices, performance=timing_policy,
                                    assignment_sha256=b.digest(json.dumps(assignment,sort_keys=True).encode()))
        if first_8k:
            protocol['parallel']['first_phase'] = '8192+256; shared weights, separate sample warmups and caches'
        h = b.digest(json.dumps(protocol,sort_keys=True).encode())
        manifest['protocol_hash'] = h
        mf = out/'run_manifest.json'
        if mf.exists() and json.loads(mf.read_text())['protocol_hash']!=h:
            raise RuntimeError('Hardware, code, samples or shard layout changed; use a fresh --out')
        b.atomic(mf,manifest)
        if args.skip_prior_oom:
            b.atomic(out/'skipped_prior_oom.json',{'policy':protocol['prior_oom_filter'],'records':skipped})
            print(f"Restricted campaign: {protocol['prior_oom_filter']['planned_records']} planned; "
                  f"{len(skipped)} deliberately excluded (see skipped_prior_oom.json).",flush=True)
        (out/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        def lane(i): return out/'lanes'/str(i)
        def env(i): return {**os.environ,'CUDA_VISIBLE_DEVICES':devices[i]['uuid'], 'PYTHONUNBUFFERED':'1',
                            'TRITON_CACHE_DIR':os.environ.get('PHASEKV_TRITON_CACHE_DIR',str(out/'jit'/str(i))), 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
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
        def measure(i,suites,selected_rows=None):
            folder = lane(i)
            for suite in suites:
                for method in b.METHODS:
                    eligible = rows if selected_rows is None else selected_rows
                    todo = any(assignment[s['id']]==i and b.pending(b.result_path(folder,m,s,r),args.retry_failed)
                               for m,s,r in b.expected(eligible,[suite],[method]))
                    if not todo: continue
                    job = folder/'jobs'/f'{suite}__{method}.json'
                    b.atomic(job,dict(out=str(folder),model=args.model,suite=suite,method=method,lane=i,
                                      protocol_hash=h,retry_failed=args.retry_failed,
                                      sample_ids=None if selected_rows is None else [s['id'] for s in selected_rows]))
                    log = folder/'logs'/f'{suite}__{method}.log'
                    print(f'GPU {i}: {suite}/{method} -> {log}',flush=True)
                    launch([sys.executable,__file__,'worker',job],env(i),log)
        ordinary_path = b.result_path
        def summarize_shards(destination, samples, suites):
            b.result_path = lambda _,m,s,r: ordinary_path(lane(assignment[s['id']]),m,s,r)
            try:
                status = b.summarize(destination,samples,suites,b.METHODS)
                if args.skip_prior_oom and destination == out:
                    summary_path = out/'summary.json'
                    summary = json.loads(summary_path.read_text())
                    summary['prior_oom_filter'] = protocol['prior_oom_filter']
                    b.atomic(summary_path,summary)
                    with (out/'summary.md').open('a',encoding='utf-8') as f:
                        f.write(f'\nExcluded {len(skipped)} historical OOM-shape records from all four arms; '
                                'counts above refer only to the selected workload. See skipped_prior_oom.json.\n')
                return status
            finally:
                b.result_path = ordinary_path
        perf_rows = [s for s in rows if s['suite']=='performance']
        rest = perf_rows
        if first_8k:
            early = [s for s in perf_rows if s['task']=='8192+256']
            if len(early)!=3 or any(len(s['input_ids'])!=8192 or s['max_new_tokens']!=256 for s in early):
                raise RuntimeError('Expected three frozen 8192+256 prompts')
            rest = [s for s in perf_rows if s['task']!='8192+256']
            files = []
            for method in b.METHODS:
                file = lane(0)/'jobs'/f'first_8k__{method}.json'
                b.atomic(file,dict(out=str(lane(0)),model=args.model,suite='performance',method=method,
                                   lane=0,protocol_hash=h,retry_failed=args.retry_failed,sample_ids=[s['id'] for s in early]))
                files.append(str(file))
            spec = lane(0)/'jobs/first_8k.json'
            b.atomic(spec,dict(jobs=files,profile=args.profile_8k))
            print(f'First 8192+256 measurements: {lane(0)}/first_8k.log',flush=True)
            try:
                launch([sys.executable,__file__,'first-workers',spec],env(0),lane(0)/'first_8k.log')
            finally:
                early_status = summarize_shards(out/'first_8k',early,['performance'])
            print(f'First 8K results: {out}/first_8k/ablation_pairs.json {dict(early_status)}',flush=True)
            if args.only_8k: return 2 if early_status.get('ok',0)!=36 else 0
        try:
            if args.parallel_performance:
                concurrently(lambda i: measure(i,['performance'],rest),range(args.gpus))
            else:
                measure(0,['performance'],rest)
            # Publish speed comparisons as soon as all timed workers finish.
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
    elif sys.argv[1:2]==['first-workers']: first_workers(sys.argv[2])
    elif sys.argv[1:2]==['capacity']: capacity(sys.argv[2],Path(sys.argv[3]))
    else: raise SystemExit(main())
