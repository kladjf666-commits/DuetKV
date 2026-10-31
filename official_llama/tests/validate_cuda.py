"""Real CUDA kernel gates; compares to materialized cache, not unquantized KV."""
from pathlib import Path
import argparse
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare import verify
sys.path.insert(0, str(ROOT / 'runtime'))
sys.path.insert(0, str(ROOT / 'runtime/KIVI'))


def check_close(name, actual, expected, checks):
    import torch
    a, b = actual.float(), expected.float()
    metrics = {'check': name, 'max_abs': (a - b).abs().max().item(),
               'rms_abs': ((a - b).square().mean().sqrt()).item(), 'atol': .005, 'rtol': .01}
    checks.append(metrics)
    torch.testing.assert_close(a, b, atol=.005, rtol=.01, equal_nan=False)
    metrics['status'] = 'passed'
    print(name, metrics, flush=True)


def kivi(checks):
    import torch
    from quant.new_pack import triton_quantize_and_pack_along_last_dim, unpack_tensor
    from quant.matmul import cuda_bmm_fA_qB_outer
    # Real Llama-3.1 head dimension and GQA factor, both K and V GEMV layouts.
    for inner, outer in ((128, 256), (256, 128)):
        b = torch.randn(1, 8, inner, outer, device='cuda', dtype=torch.float16)
        code, scale, mins = triton_quantize_and_pack_along_last_dim(b, 32, 4)
        q = torch.softmax(torch.randn(1, 32, 1, inner, device='cuda', dtype=torch.float16), dim=-1)
        actual = cuda_bmm_fA_qB_outer(32, q, code, scale, mins, 4)
        values = unpack_tensor(code, 4, 3).float()
        dense = (values * scale.float().repeat_interleave(32, dim=-1)
                 + mins.float().repeat_interleave(32, dim=-1)).half()
        expected = q.float() @ dense.repeat_interleave(4, dim=1).float()
        check_close(f'KIVI native CUDA GQA4 G32 4bit {inner}x{outer}', actual, expected, checks)


def phasekv(checks):
    import torch
    from transformers.cache_utils import DynamicCache
    from PhaseKV.phasekv_cache import PhaseKVCache
    from PhaseKV.kernels.attention_triton import fused_phasekv_decode
    from PhaseKV.kernels.packing import unpack_dequantize
    options = json.loads((ROOT / 'config.json').read_text())['methods']['phasekv']
    dense = DynamicCache()
    k = torch.randn(1, 8, 256, 128, device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    dense.update(k, v, 0)
    keep = torch.arange(round(256 * .7), device='cuda')[None, None].expand(1, 8, -1)
    keys = ('prompt_bits', 'cold_bits', 'group_size', 'recent_window', 'lazy_g', 'decode_keep_ratio', 'optimizations')
    cache = PhaseKVCache.from_dense_prefill(dense, prompt_len=256, keep_indices=[keep],
                                           **{key: options[key] for key in keys})
    del dense, k, v
    for step in range(800):
        key = torch.randn(1, 8, 1, 128, device='cuda', dtype=torch.float16)
        cache.append_hot(0, key, torch.randn_like(key))
        if step in (0, 127, 143, 255, 639, 799):
            q = torch.randn(32, 128, device='cuda', dtype=torch.float16)
            layer = cache.layers[0]
            actual = fused_phasekv_decode(q, layer, 32)
            segments = [layer.prompt, *layer.prompt_extensions, *layer.cold]
            hot_k, hot_v = layer.hot_views()
            k = torch.cat([unpack_dequantize(s.key) for s in segments] + [hot_k], dim=2)
            v = torch.cat([unpack_dequantize(s.value) for s in segments] + [hot_v], dim=2)
            k = k.repeat_interleave(4, dim=1).float()
            v = v.repeat_interleave(4, dim=1).float()
            scores = q.float()[None, :, None] @ k.transpose(-1, -2) / (128 ** .5)
            expected = (scores.softmax(-1) @ v)[0, :, 0]
            check_close(f'PhaseKV P5 cold/page/ring GQA4 step {step}', actual, expected, checks)
        cache.finish_decode_step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'results/cuda_kernels.json'))
    args = parser.parse_args()
    checks = []
    result = {'status': 'running', 'checks': checks}
    try:
        verify()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; these tests cannot pass on CPU')
        torch.manual_seed(20261028)
        torch.backends.cuda.matmul.allow_tf32 = False
        result['gpu'] = torch.cuda.get_device_name()
        result['torch'] = torch.__version__
        kivi(checks)
        phasekv(checks)
        result['status'] = 'passed'
    except Exception:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        raise
    finally:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
