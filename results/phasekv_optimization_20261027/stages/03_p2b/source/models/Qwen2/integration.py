"""One Qwen2/Qwen2.5 model instance, six independently reset KV methods."""
from eval import runtime
from eval.compressors import build_compressor
import baselines  # Registers token-eviction implementations.
from .packed_runtime import generate_packed

SUPPORTED_METHODS = ('full', 'kivi', 'snapkv', 'scope', 'pyramidkv', 'phasekv')


def generate(model, tokenizer, ids, spec, *, max_new_tokens, stop_on_eos=True):
    if model.config.model_type != 'qwen2':
        raise TypeError('Qwen2 family integration requires model_type=qwen2')
    if max_new_tokens < 1 or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError('Requires batch=1, nonempty input and max_new_tokens>=1')
    if ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('Actual prompt plus generation limit exceeds model context length')
    method = spec['method']
    if method not in SUPPORTED_METHODS:
        raise ValueError(f'Unsupported KV method: {method}')
    params = {k:v for k,v in spec.items() if k != 'method'}
    if method in ('kivi','phasekv'):
        result = generate_packed(model, tokenizer, ids, method=method, params=params,
                                 max_new_tokens=max_new_tokens, stop_on_eos=stop_on_eos,
                                 trace_compression=True)
        evidence = result['storage_evidence']
        if evidence['dense_shadow'] or evidence['code_dtype'] != 'torch.uint8':
            raise AssertionError('Expected genuine packed cache with no dense shadow')
        return result
    compressor = build_compressor(method, **params)
    return runtime.generate_with_compression(model, tokenizer, ids, compressor,
                max_new_tokens=max_new_tokens, stop_on_eos=stop_on_eos,
                greedy=True, seed=1234, trace_compression=True)
