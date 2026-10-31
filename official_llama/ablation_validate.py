"""Mandatory CUDA gates. A failed check stops formal ablation measurements."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import native_runtime  # Installs the isolated, frozen runtime import path.

ROOT = Path(__file__).resolve().parent


def check_close(name, actual, expected, results, *, atol=.005, rtol=.01):
    import torch
    a, b = actual.float(), expected.float()
    error = a-b
    item = {'name': name, 'max_abs': error.abs().max().item(),
            'rms_abs': error.square().mean().sqrt().item(), 'atol': atol, 'rtol': rtol}
    results.append(item)
    print(json.dumps(item), flush=True)
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=False)


def kernel_checks(results, options):
    import torch
    from types import SimpleNamespace
    from transformers.cache_utils import DynamicCache
    from flash_attn import flash_attn_func
    from PhaseKV.phasekv_cache import PhaseKVCache
    from PhaseKV.kernels.packing import unpack_dequantize
    from PhaseKV.kernels.attention_triton import fused_phasekv_decode
    from ablation_kernels import dense_decode, materialize, segmented_phasekv_decode

    def reference(tiers):
        parts = (tiers.prompt, *tiers.prompt_extensions, *tiers.cold)
        hot_k, hot_v = tiers.hot_views()
        return (torch.cat([*(unpack_dequantize(s.key) for s in parts), hot_k], dim=2),
                torch.cat([*(unpack_dequantize(s.value) for s in parts), hot_v], dim=2))

    def flash(q, k, v):
        return flash_attn_func(q[None, None], k.transpose(1, 2), v.transpose(1, 2),
                              dropout_p=0., causal=True)[0, 0]

    def math_reference(q, k, v):
        repeat = q.shape[0] // k.shape[1]
        key, value = [x[0].float().repeat_interleave(repeat, dim=0) for x in (k, v)]
        weights = torch.softmax(torch.sum(q.float()[:, None]*key, dim=-1)/q.shape[-1]**.5, dim=-1)
        return (weights[..., None]*value).sum(dim=1)

    # Non-aligned and long lengths, real Llama head shape, and MHA/GQA variants.
    for heads, kvheads in ((32, 8), (4, 2), (4, 4)):
        for length in (17, 129, 4097, 16384):
            q = torch.randn(heads, 128, device='cuda', dtype=torch.float16)
            k = torch.randn(1, kvheads, length, 128, device='cuda', dtype=torch.float16)
            v = torch.randn_like(k)
            got = dense_decode(q, k, v, SimpleNamespace())
            label = f'dense_h{heads}_kv{kvheads}_n{length}'
            check_close(label+'_math', got, math_reference(q, k, v), results)
            check_close(label+'_fa2', got, flash(q, k, v), results)
    packing = {k: options[k] for k in ('prompt_bits', 'cold_bits', 'group_size',
                                      'recent_window', 'lazy_g', 'decode_keep_ratio', 'optimizations')}
    for bits in (2, 4, 8):
        for length in (17, 257):
            k = torch.randn(1, 8, length, 128, device='cuda', dtype=torch.float16)
            v = torch.randn_like(k)
            dense = DynamicCache()
            dense.update(k, v, 0)
            config = {**packing, 'prompt_bits': bits, 'cold_bits': bits}
            cache = PhaseKVCache.from_dense_prefill(dense, prompt_len=length, **config)
            # 657 tokens crosses ring wrap, 8-segment grouping, 32-slot page
            # allocation and descriptor-capacity growth with production options.
            steps = 658 if bits == 4 and length == 257 else 161
            checkpoints = {0, 128, 143, 144, 145, 160, 256, 272, 641, 656, 657}
            for step in range(steps):
                if step:
                    cache.append_hot(0, torch.randn_like(k[:, :, :1]), torch.randn_like(v[:, :, :1]))
                if step in checkpoints:
                    tiers = cache.layers[0]
                    actual = materialize(tiers)
                    expected = reference(tiers)
                    label = f'packed_b{bits}_n{length}_step{step}'
                    for name, a, b in zip(('K', 'V'), actual, expected):
                        check_close(label+'_unpack_'+name, a, b, results, atol=0., rtol=0.)
                    q = torch.randn(32, 128, device='cuda', dtype=torch.float16)
                    got = segmented_phasekv_decode(q, tiers, 32)
                    check_close(label+'_original_fused', got, fused_phasekv_decode(q, tiers, 32),
                                results, atol=0., rtol=0.)
                    check_close(label+'_materialized_fa2', got, flash(q, *actual), results)
                    check_close(label+'_materialized_math', got, math_reference(q, *actual), results)
                if step:
                    cache.finish_decode_step()


def model_checks(results, options, model_path):
    import torch
    from transformers.cache_utils import DynamicCache
    from ablation_attention import AblationAttention
    from ablation_runtime import load_model, forward, phase_prefill, METHODS
    from ablation_generate import generate
    from benchmark import samples

    model, _, _ = load_model(model_path, 'full_fa2', tiny=True)
    originals = [layer.self_attn.original for layer in model.model.layers]
    prompt = torch.randint(3, 255, (1, 257), device='cuda')
    teacher = torch.randint(3, 255, (1, 273), device='cuda')

    def run(method):
        for layer, original in zip(model.model.layers, originals):
            layer.self_attn = original if method.startswith('native_') else AblationAttention(original, method)
        phase = 'phasekv' in method
        if method == 'native_phasekv':
            native_runtime.install(model, 'phasekv', options)
        if phase:
            logits, cache = phase_prefill(model, prompt, options)
        else:
            logits, cache, _ = native_runtime.forward(model, prompt, DynamicCache(), 0, 'full')
        outputs = [logits.cpu()]
        for step in range(273 if phase else 17):
            logits, cache, _ = native_runtime.forward(model, teacher[:, step:step+1], cache,
                                                       prompt.shape[1]+step, 'phasekv' if phase else 'full')
            if phase:
                cache.finish_decode_step()
            outputs.append(logits.cpu())
        return torch.cat(outputs, dim=1)

    full_ref = run('native_full')
    for method in ('full_fa2', 'full_triton'):
        check_close('tiny_teacher_'+method, run(method), full_ref, results)
    phase_ref = run('native_phasekv')
    check_close('tiny_teacher_phasekv_triton', run('phasekv_triton'), phase_ref, results,
                atol=0., rtol=0.)
    check_close('tiny_teacher_phasekv_fa2', run('phasekv_fa2'), phase_ref, results)
    # Remove every reference before loading the actual 8B model on the 24GB GPU.
    del model, originals, full_ref, phase_ref
    gc.collect()
    torch.cuda.empty_cache()
    model, _, _ = load_model(model_path, 'full_fa2')
    frozen, _ = samples()
    sample = dict(next(s for s in frozen if s['suite'] == 'performance'))
    sample.update(id='ablation_real_model_smoke', input_ids=sample['input_ids'][:257],
                  max_new_tokens=161, natural_eos=False, stop_ids=[])
    smoke = []
    for method in METHODS:
        for layer in model.model.layers:
            layer.self_attn.method = method
        result = generate(model, method, options if method.startswith('phasekv_') else {}, sample)
        if result['output_tokens'] != 161:
            raise AssertionError('Fixed output length violated')
        result.pop('output_ids')
        smoke.append({'method': method, 'timing_excluded': True, **result})
        gc.collect()
        torch.cuda.empty_cache()
    return smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--protocol-hash', required=True)
    args = parser.parse_args()
    from benchmark import atomic
    report = {'status': 'running', 'protocol_hash': args.protocol_hash, 'checks': [],
              'started_at': time.time(), 'gpu_tests_executed': False}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; these gates cannot pass on CPU')
        import flash_attn
        import transformers
        import triton
        from prepare import verify
        verify()
        config = json.loads((ROOT/'config.json').read_text('utf-8'))
        if (torch.__version__.split('+')[0] != config['torch']
                or transformers.__version__ != config['transformers']
                or flash_attn.__version__ != config['flash_attn'] or triton.__version__ != '3.1.0'):
            raise RuntimeError('Use the pinned torch/transformers/flash-attn/triton environment')
        torch.manual_seed(20261031)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      flash_attn=flash_attn.__version__, triton=triton.__version__, gpu_tests_executed=True)
        with torch.inference_mode():
            kernel_checks(report['checks'], config['methods']['phasekv'])
            report['real_model_smoke'] = model_checks(report['checks'], config['methods']['phasekv'], args.model)
        report['status'] = 'passed'
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        report['finished_at'] = time.time()
        atomic(args.out, report)


if __name__ == '__main__':
    main()
