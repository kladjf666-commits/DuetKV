"""Run only channel-contiguous PhaseKV D=0.80 with the frozen B=1 worker."""
import argparse
import collections
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ARM = 'phasekv_channel_k_d080'
EXCLUDED = {'16384+256', '16384+4096', '32000+256'}
COUNTS = {'performance': 18, 'longbench': 640, 'ruler': 384, 'longgenbench': 48}


def select(rows):
    selected = [s for s in rows if 'phasekv' in s['methods']
                and (s['suite'] != 'performance' or s['task'] not in EXCLUDED)]
    if collections.Counter(s['suite'] for s in selected) != COUNTS:
        raise ValueError('Frozen campaign sample counts differ')
    return selected


def gpu_gate(options):
    import torch
    from transformers.cache_utils import DynamicCache
    from channel_impl.phasekv_cache import PhaseKVCache
    from channel_impl.kernels.packing import unpack_dequantize
    from channel_impl.kernels.segmented_attention import segmented_phasekv_decode

    torch.manual_seed(20261029)
    k, v = torch.randn(2, 1, 2, 300, 128, device='cuda', dtype=torch.float16).unbind()
    dense = DynamicCache()
    dense.update(k[:, :, :65], v[:, :, :65], 0)
    cache = PhaseKVCache.from_dense_prefill(dense, prompt_len=65, **{
        key: options[key] for key in ('decode_keep_ratio','recent_window','lazy_g','cold_bits',
                                      'prompt_bits','group_size','optimizations')})
    q = k[0, :, 0].repeat_interleave(4, 0)
    layer = cache.layers[0]
    errors = {}
    for i in range(272):
        cache.append_hot(0, k[:, :, i:i+1], v[:, :, i:i+1])
        cache.finish_decode_step()
        if i+1 not in (143, 144, 272):
            continue
        nseg = max(0, (i+1-128)//16)
        if [s.length for s in layer.cold] != [13]*nseg:
            raise RuntimeError('Unexpected cold segment lengths during gate')
        for j, seg in enumerate(layer.cold):
            values = v[:, :, 16*j:16*(j+1)]
            retained = values.float().norm(dim=-1).argsort(descending=True)[..., :13].sort().values
            target = values.gather(2, retained.unsqueeze(-1).expand(-1,-1,-1,128))
            # Quantization error is bounded by half a step per channel group.
            err = (unpack_dequantize(seg.value).float()-target.float()).abs()
            bound = seg.value.scale.repeat_interleave(32, dim=-1).float() * .51 + .005
            if (err > bound).any().item():
                raise RuntimeError('Cold V does not match ordered per-head top-13 selection')
        out = segmented_phasekv_decode(q, layer, q.shape[0])
        frozen = [layer.prompt, *layer.cold]
        hotk, hotv = layer.hot_views()
        key, val = [torch.cat([unpack_dequantize(getattr(s, field)) for s in frozen]+[hot],
                              dim=2)[0].float().repeat_interleave(4, 0)
                    for field, hot in [('key', hotk), ('value', hotv)]]
        logits = torch.einsum('hd,hnd->hn', q.float(), key) / 128**.5
        ref = torch.einsum('hn,hnd->hd', logits.softmax(-1), val)
        torch.testing.assert_close(out.float(), ref, atol=.003, rtol=.01)
        if not layer._decode_workspace.compatible_4bit:
            raise RuntimeError('Specialized attention fallback during gate')
        errors[str(nseg)] = float((out.float()-ref).abs().max())
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path('/root/data/phasekv-ws/official_llama'))
    p.add_argument('--model', type=Path, default=Path('/root/data/Llama-3.1-8B-Instruct/snapshots/master'))
    p.add_argument('--out', type=Path, default=Path('/root/data/phasekv-results/rtx4090_channel_keep80_full_v1'))
    args = p.parse_args()
    sys.path.insert(0, str(args.project.resolve()))
    import attention_ablation as bench
    import ablation_runtime as rt
    from verify_attention_ablation_bundle import verify
    from check_model import check
    from benchmark_metrics import score
    from retest import optimized
    from compare import report
    import torch, triton, transformers, flash_attn
    import fcntl

    lock = Path('/root/data/phasekv-native-benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = json.loads((HERE/'files.json').read_text())
    for name, expected in files.items():
        if bench.digest((HERE/name).read_bytes()) != expected:
            raise RuntimeError(f'Upload fingerprint mismatch: {name}')
    verified = verify()
    rows, frozen = bench.samples()
    rows = select(rows)
    for suite in ('longbench','ruler','longgenbench'):
        score(next(s for s in rows if s['suite']==suite), '')
    check(args.model)
    for name, expected in frozen['tokenizer_files'].items():
        if name != 'configuration.json' and bench.digest((args.model/name).read_bytes()) != expected:
            raise RuntimeError(f'Model/tokenizer differs from frozen benchmark: {name}')
    env = [str(torch.__version__), triton.__version__, transformers.__version__, flash_attn.__version__]
    if (env != ['2.5.1+cu121','3.1.0','4.46.3','2.7.4.post1'] or torch.cuda.device_count()!=1
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 4090'):
        raise RuntimeError(f'Expected original RTX 4090 benchmark environment; got {env}')
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
    pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                    '--format=csv,noheader'],text=True).split()
    if set(pids)-{str(os.getpid())}:
        raise RuntimeError(f'GPU busy: {pids}')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.project/'config.json').read_text())['methods']['phasekv']
    base_loader, base_trace = rt.load_model, rt.cache_trace
    with optimized(dict(config, decode_keep_ratio=.8)) as (_, opts), patch.object(rt, 'METHODS', [*rt.METHODS, ARM]):
        protocol = dict(method=ARM, batch=1, options=opts, gpu=gpu, environment=env,
                        model=str(args.model.resolve()), frozen_data=frozen['samples_sha256'],
                        weight_sizes={p.name:p.stat().st_size for p in args.model.glob('*.safetensors')},
                        excluded_shapes=sorted(EXCLUDED), records=1126, overlay=files,
                        code={**verified['files'], **verified['base_files']})
        fingerprint = bench.digest(json.dumps(protocol, sort_keys=True).encode())
        manifest = out/'run_manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())['protocol_hash'] != fingerprint:
                raise RuntimeError('Different hardware or protocol; select a fresh --out')
        elif any(out.iterdir()):
            raise RuntimeError('Unidentified files in result directory; select a fresh --out')
        bench.atomic(manifest, dict(protocol_hash=fingerprint, protocol=protocol))
        print(f'{ARM}: 1072 quality samples + 54 performance records, output={out}', flush=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            gate = gpu_gate(opts)
        bench.atomic(out/'gpu_gate.json', dict(status='passed', protocol_hash=fingerprint, max_abs_by_cold_segments=gate))
        print('GPU gate passed at 0, 1 and 9 cold segments.', flush=True)

        def load_candidate(model, method):
            if method != ARM:
                raise ValueError(method)
            loaded, _, source = base_loader(model, method)
            return loaded, opts, source + ['channel_impl']

        def trace(cache, method):
            from channel_impl.phasekv_cache import PhaseKVCache
            if not isinstance(cache, PhaseKVCache) or cache.decode_keep_ratio != .8:
                raise RuntimeError('Incorrect measured cache')
            t = base_trace(cache, method)
            for layer in cache.layers:
                ws = layer._decode_workspace
                if (not layer.optimizations.channel_contiguous_k or not layer.optimizations.specialized_4bit
                        or (ws is not None and not ws.compatible_4bit)
                        or any(s.length != 13 for s in layer.cold)):
                    raise RuntimeError('Unexpected measured cache layout')
            return dict(t, cold_kept=13*t['cold_segments'], cold_discarded=3*t['cold_segments'],
                        channel_contiguous_k=True)

        def pending(path, retry_failed=False):
            return not path.exists() or json.loads(path.read_text('utf-8'))['status']=='error'

        with patch.object(bench,'samples',return_value=(rows,frozen)), patch.object(rt,'load_model',load_candidate), \
             patch.object(rt,'cache_trace',trace), patch.object(bench,'pending',pending):
            for suite in bench.SUITES:
                job = out/'jobs'/f'{suite}.json'
                bench.atomic(job, dict(out=str(out), protocol_hash=fingerprint, suite=suite,
                                       method=ARM, model=str(args.model), retry_failed=False))
                bench.worker(job)
                gc.collect()
                torch.cuda.empty_cache()
        status = bench.summarize(out, rows, bench.SUITES, [ARM])
        if sum(status.values()) != 1126 or status['pending'] or status['error']:
            raise RuntimeError(f'Incomplete run: {status}')
        report(out, HERE)
        print(f'FINISHED {dict(status)} | {out}/comparison.json', flush=True)


if __name__ == '__main__':
    main()
