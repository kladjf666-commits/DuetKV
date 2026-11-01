"""Measure only channel-contiguous PhaseKV; reuse the recorded baselines."""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from unittest.mock import patch

import torch
import probe

HERE = Path(__file__).resolve().parent
FULL, OLD, NEW = 'full_fa2', 'phasekv_triton', 'phasekv_channel_k'
REPEATS = 5


@contextlib.contextmanager
def optimized(options):
    import ablation_attention
    import ablation_kernels
    import batch_attention
    import PhaseKV.phasekv_cache as frozen
    from channel_impl.phasekv_cache import PhaseKVCache
    from channel_impl.kernels.segmented_attention import segmented_phasekv_decode
    with contextlib.ExitStack() as stack:
        for module, name, value in (
                (frozen, 'PhaseKVCache', PhaseKVCache),  # frozen prefill imports this at call time
                (ablation_attention, 'PhaseKVCache', PhaseKVCache),
                (ablation_kernels, 'segmented_phasekv_decode', segmented_phasekv_decode),
                (batch_attention, 'segmented_phasekv_decode', segmented_phasekv_decode)):
            stack.enter_context(patch.object(module, name, value))
        yield OLD, dict(options, optimizations=dict(options['optimizations'], channel_contiguous_k=True))


def audit(cache):
    from channel_impl.phasekv_cache import PhaseKVCache
    from channel_impl.kernels.segmented_plan import compatible_4bit
    if not isinstance(cache, PhaseKVCache) or not cache.optimizations.channel_contiguous_k:
        raise RuntimeError('Optimized cache not installed')
    for layer in cache.layers:
        if not (layer.optimizations.specialized_4bit and layer.optimizations.channel_contiguous_k
                and layer._decode_workspace and layer._decode_workspace.compatible_4bit
                and compatible_4bit(layer.prompt, prompt=True, channel_contiguous_k=True)
                and layer.cold and all(compatible_4bit(s, prompt=False, channel_contiguous_k=True)
                                       for s in layer.cold)):
            raise RuntimeError('Optimized prompt/cold layout or specialization was lost')
    return dict(channel_contiguous_k=True, checked_layers=len(cache.layers),
                prompt_K_stride=cache.layers[0].prompt.key.packed.stride(),
                cold_segments=len(cache.layers[0].cold))


def reference_info(directory, protocol, baseline):
    result = {'kind':'previous run', 'same_gpu_uuid':None, 'output_hashes':{}}
    if not all((directory/n).is_file() for n in ('summary.json','protocol.json')):
        return result
    previous = json.loads((directory/'protocol.json').read_text())
    if (json.loads((directory/'summary.json').read_text()) != baseline
            or previous['frozen_data'] != protocol['data'] or previous['prompts'] != protocol['prompts']
            or previous['model'] != protocol['model']):
        result['note'] = 'Volume reference differs from bundled summary or inputs; metadata not linked'
        return result
    result['same_gpu_uuid'] = previous['device'].split(',')[0].strip()==protocol['gpu'].split(',')[0].strip()
    path = directory/'measurements.json'
    if path.is_file():
        rows = json.loads(path.read_text())
        for b in range(1,5):
            saved = [rows.get(f'{OLD}/B{b}/all/{i}',{}) for i in range(REPEATS)]
            if all(r.get('status')=='ok' and r.get('output_sha256') for r in saved):
                result['output_hashes'][str(b)] = sorted({r['output_sha256'] for r in saved})
    return result


def summary(rows, sizes, baseline, reference):
    points = {}
    for b in sizes:
        records = [rows.get(f'{b}/all/{i}',{}) for i in range(REPEATS)]
        ok = [r for r in records if r.get('status')=='ok']
        point = dict(successful_repeats=len(ok), required_repeats=REPEATS)
        points[str(b)] = point
        if len(ok)!=REPEATS:
            continue
        ms = statistics.median(r['ms'] for r in ok)
        old = baseline['points'][f'{OLD}/B{b}']['ms_per_batch_step']
        full = baseline['points'][f'{FULL}/B{b}']['ms_per_batch_step']
        prev = points.get(str(b-1),{}).get('ms_per_batch_step')
        hashes = {r['output_sha256'] for r in ok}
        expected = reference['output_hashes'].get(str(b))
        point.update(ms_per_batch_step=ms, total_tokens_per_s=1000*b/ms,
                     marginal_ms=ms-prev if prev else None,
                     historical_full_ms=full,historical_phasekv_ms=old,
                     throughput_vs_historical_old_pct=100*(old/ms-1), gap_vs_historical_full_ms=ms-full,
                     peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in ok),
                     repeats_identical=len(hashes)==1,
                     matches_historical_old=(hashes==set(expected) and len(hashes)==1) if expected else None)
    return dict(measured_arm=NEW, points=points, reference=reference)


@torch.inference_mode()
def main():
    import fcntl
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path('/root/data/phasekv-ws/official_llama'))
    ap.add_argument('--model',type=Path,default=Path('/root/data/Llama-3.1-8B-Instruct/snapshots/master'))
    ap.add_argument('--out',type=Path,default=Path('/root/data/phasekv-results/rtx4090_channel_only_v1'))
    ap.add_argument('--max-batch',type=int,choices=range(1,5),default=4)
    args = ap.parse_args()
    sizes = list(range(1,args.max_batch+1))
    sys.path.insert(0,str(args.project.resolve()))
    import attention_ablation as bench
    import ablation_runtime as rt
    import flash_attn,transformers,triton
    from verify_attention_ablation_bundle import verify
    verify()
    files = json.loads((HERE/'files.json').read_text())
    for name,digest in files.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Package fingerprint mismatch: {name}')
    if (torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 4090'
            or torch.__version__!='2.5.1+cu121' or triton.__version__!='3.1.0'
            or transformers.__version__!='4.46.3' or flash_attn.__version__.split('+')[0]!='2.7.4.post1'):
        raise RuntimeError('Expected the single RTX 4090 and pinned ablation environment')
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    lock = Path('/root/data/phasekv-native-benchmark.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if pids not in ('',str(os.getpid())):
        raise RuntimeError('Another compute process is using the GPU')
    samples, frozen = bench.samples()
    cases = sorted((s for s in samples if s['suite']=='performance' and s['task']=='8192+256'),key=lambda s:s['id'])
    assert len(cases)==3 and all(len(s['input_ids'])==8192 and s['max_new_tokens']==256 for s in cases)
    opts = json.loads((args.project/'config.json').read_text())['methods']['phasekv']
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    meta = dict(gpu=gpu,model=str(args.model.resolve()),data=frozen['samples_sha256'],prompts=[s['id'] for s in cases],
                repeats=REPEATS,measured_arm=NEW,files=files,base_options=opts,candidate_override={'channel_contiguous_k':True})
    mp = out/'protocol.json'
    if mp.exists() and json.loads(mp.read_text())!=json.loads(json.dumps(meta)):
        raise RuntimeError('Hardware or protocol changed; select a new --out')
    bench.atomic(mp,meta)
    rp = out/'measurements.json'
    rows = json.loads(rp.read_text()) if rp.exists() else {}
    torch.manual_seed(20261031)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baseline = json.loads((HERE/'historical_summary.json').read_text())
    reference = reference_info(Path('/root/data/phasekv-results/rtx4090_decode_probe_v1'),meta,baseline)
    inputs = [torch.tensor([s['input_ids']],device='cuda') for s in cases]
    print('Measured arm:',NEW,'; historical baseline GPU UUID match:',reference['same_gpu_uuid'],flush=True)
    model = None
    with optimized(opts) as (method,config):
        for b in sizes:
            if any(rows.get(f'{b}/all/{i}',{}).get('status')=='oom' for i in range(-1,REPEATS)):
                break
            todo = [i for i in range(REPEATS) if rows.get(f'{b}/all/{i}',{}).get('status')!='ok']
            if not todo:
                continue
            if model is None:
                model,_,_ = rt.load_model(args.model,method)
            prompts = [inputs[i%len(inputs)] for i in range(b)]
            for i in [-1,*todo]:
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    row = dict(status='ok', **probe.run_one(model,method,prompts,config,'all',cache_audit=audit))
                except torch.OutOfMemoryError as err:
                    row = dict(status='oom',error=str(err))
                rows[f'{b}/all/{i}'] = row
                bench.atomic(rp,rows)
                bench.atomic(out/'summary.json',summary(rows,sizes,baseline,reference))
                print(f'{NEW} B={b} round={i}',row['status'],round(row.get('ms',0),3),'ms/batch step',flush=True)
                if row['status']=='oom':
                    break
            if row['status']=='oom':
                break
    report = summary(rows,sizes,baseline,reference)
    bench.atomic(out/'summary.json',report)
    print('Batch   optimized ms   optimized token/s   throughput vs historical old   historical Full ms')
    for b,p in report['points'].items():
        if 'ms_per_batch_step' in p:
            print(b,round(p['ms_per_batch_step'],3),round(p['total_tokens_per_s'],2),
                  f"{p['throughput_vs_historical_old_pct']:+.2f}%",round(p['historical_full_ms'],3))
    print('Finished.',out/'summary.json',flush=True)


if __name__=='__main__':
    main()
