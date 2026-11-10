"""One-device 8K decode diagnostics. Uses the installed, frozen ablation deployment."""
import argparse
import contextlib
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import torch

from trace_budget import account, LABEL, HOST

VOLUME = Path('/root/data')
METHODS = ('full_fa2', 'phasekv_triton')
HIST_WALL = 2.792663
HIST_KERNEL = 0.855373


def prepare(model, method, prompts, opts):
    import ablation_runtime as rt
    from batch_attention import merge_prefills
    from transformers.cache_utils import DynamicCache
    packed = method == 'phasekv_triton'
    singles, first = [], []
    for ids in prompts:
        if packed:
            logits, cache = rt.phase_prefill(model, ids, opts)
        else:
            logits, cache, _ = rt.forward(model, ids, DynamicCache(), 0, method)
        first.append(logits.argmax(-1))
        singles.append(cache)
    return torch.cat(first), merge_prefills(singles, packed, opts)


def step(model, method, token, cache, pos, labels=False):
    from ablation_runtime import forward
    if not labels:
        logits, _, _ = forward(model, token, cache, pos, method)
        if method == 'phasekv_triton':
            cache.finish_decode_step()
        return logits.argmax(-1), logits
    scope = torch.profiler.record_function
    with scope('probe/model forward'):
        logits, _, _ = forward(model, token, cache, pos, method)
    if method == 'phasekv_triton':
        with scope('probe/cache finish'):
            cache.finish_decode_step()
    with scope('probe/argmax'):
        return logits.argmax(-1), logits


def set_path(model, method, batched):
    from ablation_attention import AblationAttention
    from batch_attention import BatchAttention
    cls = BatchAttention if batched else AblationAttention
    for layer in model.model.layers:
        layer.self_attn = cls(layer.self_attn.original, method)


def gate(model_path, opts, batches):
    """Compare independent frozen B=1 reference logits against the joint batch, through cooling."""
    import ablation_runtime as rt
    model, _, _ = rt.load_model(model_path, 'full_fa2', tiny=True)
    gen = torch.Generator(device='cuda').manual_seed(8491)
    n = max(batches)
    prompts = [torch.randint(3, 255, (1, 69), device='cuda', generator=gen) for _ in range(n)]
    teachers = torch.randint(3, 255, (n, 148), device='cuda', generator=gen)
    result = {}
    for method in METHODS:
        def logits_for(indices, batched):
            set_path(model, method, batched)
            _, cache = prepare(model, method, [prompts[i] for i in indices], opts)
            tokens = teachers[indices]
            values = []
            for i in range(tokens.shape[1]):
                _, logits = step(model, method, tokens[:, i:i+1], cache, 69+i)
                values.append(logits)
            return torch.cat(values, dim=1)
        refs = torch.cat([logits_for([i], False) for i in range(n)])
        for b in batches:
            values = logits_for(list(range(b)), b > 1)
            torch.testing.assert_close(values, refs[:b], atol=.005, rtol=.01)
            result[f'{method}/B{b}'] = float((values-refs[:b]).abs().max().item())
    return result


@contextlib.contextmanager
def profile_labels():
    from PhaseKV.phasekv_cache import PhaseKVCache
    from transformers.cache_utils import DynamicCache
    import ablation_kernels
    with contextlib.ExitStack() as stack:
        for obj, attr, label in [(PhaseKVCache,'append_hot','cache append'),
                                 (DynamicCache,'update','cache append'),
                                 (ablation_kernels,'segmented_phasekv_decode','attention dispatch'),
                                 (ablation_kernels,'ensure_workspace','workspace')]:
            original = getattr(obj, attr)
            def wrapped(*a, _fn=original, _label=label, **kw):
                with torch.profiler.record_function('probe/'+_label):
                    return _fn(*a, **kw)
            stack.enter_context(patch.object(obj, attr, wrapped))
        yield


def run_one(model, method, prompts, opts, mode, trace_path=None, cache_audit=None):
    batch = len(prompts)
    set_path(model, method, batch > 1)
    torch.cuda.reset_peak_memory_stats()
    token, cache = prepare(model, method, prompts, opts)
    torch.cuda.synchronize()
    length = prompts[0].shape[1]
    tokens = [token]
    if mode == 'all':
        elapsed = 0.
        start = time.perf_counter()
        for s in range(1, 256):
            token, logits = step(model, method, token, cache, length+s-1)
            tokens.append(token)
            if s in (128, 255):
                torch.cuda.synchronize()
                elapsed += time.perf_counter()-start
                start = time.perf_counter()
        row = {'ms':1000*elapsed/255, 'tokens_per_s':255*batch/elapsed}
    else:
        for s in range(1,144):
            token, _ = step(model, method, token, cache, length+s-1)
            if s == 128:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        profiling = mode in ('trace', 'trace_plain')
        with contextlib.ExitStack() as stack:
            if profiling:
                if mode == 'trace':
                    stack.enter_context(profile_labels())
                prof = stack.enter_context(torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]))
                stack.enter_context(torch.profiler.record_function(LABEL))
            start = time.perf_counter()
            for s in range(144,160):
                token, logits = step(model, method, token, cache, length+s-1, mode == 'trace')
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
        row = {'ms':1000*elapsed/16}
        if profiling:
            raw = trace_path.with_suffix('.json')
            prof.export_chrome_trace(str(raw))
            with raw.open('rb') as src, gzip.open(trace_path, 'wb', compresslevel=1) as dst:
                shutil.copyfileobj(src,dst)
            events = json.loads(raw.read_bytes())['traceEvents']
            raw.unlink()
            row.update(account(events))
            if mode == 'trace_plain':
                marker = next(e for e in events if e.get('name') == LABEL and e.get('ph') == 'X')
                start, end = marker['ts'], marker['ts'] + marker['dur']
                kernels = [e for e in events if e.get('cat') == 'kernel' and start <= e['ts'] < end]
                row['attention_gpu_ms'] = sum(e['dur'] for e in kernels if any(
                    key in e['name'] for key in ('flash_fwd', '_segmented_attention', '_merge_segments'))) / 16000
    if not torch.isfinite(logits).all().item():
        raise RuntimeError('Nonfinite final logits')
    row.update(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
               peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
               last_ids=token.flatten().tolist())
    if mode == 'all':
        row['output_sha256'] = hashlib.sha256(torch.cat(tokens,dim=1).cpu().numpy().tobytes()).hexdigest()
    if cache_audit is not None:
        row['cache_audit'] = cache_audit(cache)
    return row


def summarize(rows, batches, repeats):
    def series(method, b, mode):
        return [rows[f'{method}/B{b}/{mode}/{i}'] for i in range(repeats)
                if f'{method}/B{b}/{mode}/{i}' in rows and rows[f'{method}/B{b}/{mode}/{i}']['status']=='ok']
    points, slopes = {}, {}
    for method in METHODS:
        previous = None
        for b in batches:
            values = series(method, b, 'all')
            cur = statistics.median(r['ms'] for r in values) if len(values)==repeats else None
            points[f'{method}/B{b}'] = {'successful_repeats':len(values), 'ms_per_batch_step':cur,
                                         'total_tokens_per_s':1000*b/cur if cur else None}
            if previous is not None and cur is not None:
                slopes[f'{method}/{b-1}->{b}'] = cur-previous
            previous = cur
    gaps = {}
    for b in batches:
        full, phase = [points[f'{m}/B{b}']['ms_per_batch_step'] for m in METHODS]
        if full is not None and phase is not None:
            gaps[b] = phase-full
    out = dict(points=points, marginal_ms_per_batch_step=slopes, phase_minus_full_batch_ms=gaps)
    traces = [rows.get(f'{m}/B1/trace/0',{}) for m in METHODS]
    a = [series(m,1,'all') for m in METHODS]
    w = [series(m,1,'window') for m in METHODS]
    if all(t.get('status')=='ok' for t in traces) and all(len(v)==repeats for v in a+w):
        def diff(v):
            return statistics.median(i['ms'] for i in v[1])-statistics.median(i['ms'] for i in v[0])
        def dt(field):
            return traces[1][field]-traces[0][field]
        terms = {'historical_minus_new_full_run':HIST_WALL-diff(a),
                 'full_255_minus_matched_16':diff(a)-diff(w),
                 'unprofiled_minus_profiled_window':diff(w)-dt('wall_ms'),
                 'device_union_minus_sum':dt('gpu_union_ms')-dt('gpu_sum_ms'),
                 'new_minus_historical_device_sum':dt('gpu_sum_ms')-HIST_KERNEL}
        for k in HOST:
            terms['device_idle_while_'+k] = traces[1]['idle_ms'][k]-traces[0]['idle_ms'][k]
        if abs(sum(terms.values())-(HIST_WALL-HIST_KERNEL)) > 1e-5:
            raise RuntimeError('Timeline budget does not close')
        out['historical_residual_ms'] = dict(terms=terms, sum=sum(terms.values()),
            fresh_unprofiled_full_gap=diff(a), fresh_unprofiled_window_gap=diff(w),
            fresh_profiled_device_sum_gap=dt('gpu_sum_ms'),
            notes='Signed differences. Idle/CPU labels describe coincidence. Instrumentation and run variability remain explicit.')
    return out


def main():
    import fcntl
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-batch', type=int, choices=range(1,9), default=4)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--project', type=Path, default=VOLUME/'phasekv-ws/official_llama')
    ap.add_argument('--model', type=Path, default=VOLUME/'Llama-3.1-8B-Instruct/snapshots/master')
    ap.add_argument('--out', type=Path, default=VOLUME/'phasekv-results/rtx4090_decode_probe_v1')
    args = ap.parse_args()
    if args.repeats < 3:
        ap.error('Use at least three measured repetitions')
    sys.path.insert(0, str(args.project.resolve()))
    import attention_ablation as bench
    import ablation_runtime as rt
    import flash_attn, transformers, triton
    from verify_attention_ablation_bundle import verify
    verify()
    if (torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 4090'
            or torch.__version__!='2.5.1+cu121' or triton.__version__!='3.1.0'
            or transformers.__version__!='4.46.3' or flash_attn.__version__.split('+')[0]!='2.7.4.post1'):
        raise RuntimeError('Use the existing single-4090 ablation environment')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    locks = []
    for path in (VOLUME/'phasekv-native-benchmark.lock', out/'probe.lock'):
        locks.append(path.open('a'))
        fcntl.flock(locks[-1], fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() not in ('',str(os.getpid())):
        raise RuntimeError('Another GPU compute process is active')
    samples, manifest = bench.samples()
    cases = sorted((r for r in samples if r['suite']=='performance' and r['task']=='8192+256'), key=lambda r:r['id'])
    assert len(cases)==3 and all(len(r['input_ids'])==8192 and r['max_new_tokens']==256 for r in cases)
    inputs = [torch.tensor([r['input_ids']], device='cuda') for r in cases]
    opts = json.loads((args.project/'config.json').read_text())['methods']['phasekv']
    device = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    meta = dict(device=device, model=str(args.model), repeats=args.repeats,
                frozen_data=manifest['samples_sha256'], prompts=[r['id'] for r in cases],
                sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                baseline='frozen phasekv_triton vs full_fa2; sequential prefill excluded; joint equal-length decode')
    mp = out/'protocol.json'
    if mp.exists() and json.loads(mp.read_text())!=meta:
        raise RuntimeError('Protocol or physical GPU changed; use a fresh --out')
    bench.atomic(mp,meta)
    rp = out/'measurements.json'
    rows = json.loads(rp.read_text()) if rp.exists() else {}
    batches = list(range(1,args.max_batch+1))
    torch.manual_seed(20261031)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.inference_mode():
        bench.atomic(out/'numerical_gate.json', gate(args.model,opts,batches))
        gc.collect()
        torch.cuda.empty_cache()
        model, _, _ = rt.load_model(args.model, 'full_fa2')
        blocked = set()
        for b in batches:
            prompts = [inputs[i%len(inputs)] for i in range(b)]
            modes = ['all', 'window', 'trace'] if b==1 else ['all']
            for mode in modes:
                for i in range(-1 if mode=='all' else 0, 1 if mode=='trace' else args.repeats):
                    for method in (METHODS if i%2==0 else METHODS[::-1]):
                        key = f'{method}/B{b}/{mode}/{i}'
                        if method in blocked:
                            continue
                        if key in rows and rows[key]['status']!='trace_failed' and (i>=0 or rows[key]['status']=='oom'):
                            if rows[key]['status']=='oom' and mode=='all':
                                blocked.add(method)
                            continue
                        if i<0 and all(f'{method}/B{b}/all/{j}' in rows for j in range(args.repeats)):
                            continue
                        gc.collect()
                        torch.cuda.empty_cache()
                        try:
                            row = dict(status='ok', **run_one(model,method,prompts,opts,mode,out/f'{method}.trace.gz'))
                        except torch.OutOfMemoryError as err:
                            row = dict(status='oom',error=str(err))
                            if mode=='all':
                                blocked.add(method)
                        except RuntimeError as err:
                            if mode!='trace':
                                raise
                            row = dict(status='trace_failed',error=str(err))
                        rows[key] = row
                        bench.atomic(rp,rows)
                        bench.atomic(out/'summary.json', summarize(rows,batches,args.repeats))
                        print(key, row['status'], round(row.get('ms',0),3),'ms/batch-step',flush=True)
    summary = summarize(rows,batches,args.repeats)
    print('Batch  method          ms/whole step    +1 marginal ms   total token/s')
    for b in batches:
        for method in METHODS:
            point = summary['points'][f'{method}/B{b}']
            ms = point['ms_per_batch_step']
            if ms is None:
                print(b,method,'incomplete/OOM')
            else:
                marginal = summary['marginal_ms_per_batch_step'].get(f'{method}/{b-1}->{b}')
                print(f'{b:5}  {method:14} {ms:12.3f}  {str(round(marginal,3)) if marginal is not None else "-":>15}  {point["total_tokens_per_s"]:14.2f}')
    print('Finished.',out/'summary.json',flush=True)


if __name__ == '__main__':
    main()
