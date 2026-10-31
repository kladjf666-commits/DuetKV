"""One Llama-2/Llama-3.1 model instance, six independently reset KV methods."""
from eval import runtime
from eval.compressors import build_compressor
import baselines  # Registers token-eviction implementations.
from .packed_runtime import generate_packed
from models.generation import stop_token_ids

SUPPORTED_METHODS = ('full', 'kivi', 'snapkv', 'scope', 'pyramidkv', 'phasekv')


def generate(model, tokenizer, ids, spec, *, max_new_tokens, stop_on_eos=True):
    if model.config.model_type != 'llama':
        raise TypeError('Llama integration requires model_type=llama')
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
                greedy=True, seed=1234, trace_compression=True,
                eos_token_ids=stop_token_ids(model, tokenizer))


def build_input_ids(tokenizer, task, sample, max_length):
    """Truncate body tokens, then apply the model template exactly once (no duplicate BOS)."""
    from scripts.run_longbench import DATASET2PROMPT
    raw = DATASET2PROMPT[task].format(context=sample['context'], input=sample['input'])
    tokens = tokenizer(raw, add_special_tokens=False).input_ids
    if len(tokens) > max_length:
        left = max_length // 2
        right = max_length - left
        raw = tokenizer.decode(tokens[:left], skip_special_tokens=True) + tokenizer.decode(tokens[-right:], skip_special_tokens=True)
    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError('Llama chat/instruct comparison requires the original chat template')
    return tokenizer.apply_chat_template([{'role':'user','content':raw}], tokenize=True,
                                         add_generation_prompt=True, return_tensors='pt')
