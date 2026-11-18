"""Bounded-lifetime prefill for the frozen Llama A800 adapter.

Selection, quantization, absolute positions and decode kernels are unchanged.
The eager observation tail selects each layer immediately; the model never
collects a tuple of full attention tensors. Packing consumes the dense cache.
"""
from contextlib import contextmanager
from types import MethodType

import torch

from .phasekv_cache import PhaseKVCache
from .prompt_tier import select_prompt_indices


@contextmanager
def capture_indices(originals, model_config, length, options, *, eager_forward=None):
    if eager_forward is None:
        from transformers.models.llama.modeling_llama import LlamaAttention
        eager_forward = LlamaAttention.forward
    saved = [attention.forward for attention in originals]
    selected = [None] * len(originals)

    def wrapper(index):
        def forward(attention, *args, **kwargs):
            kwargs['output_attentions'] = True
            output, weights, cache = eager_forward(attention, *args, **kwargs)
            if selected[index] is not None:
                raise RuntimeError('Observation layer executed more than once')
            selected[index] = select_prompt_indices(
                (weights,), prompt_len=length, model_config=model_config,
                prefill_keep_ratio=options['prefill_keep_ratio'],
                obs_window=options['obs_window'], sink_tokens=options['sink_tokens'],
                pool_kernel=7, pooling='maxpool')[0]
            # Never return weights to LlamaModel's all_self_attns collection.
            del weights
            return output, None, cache
        return forward

    try:
        for index, attention in enumerate(originals):
            attention.forward = MethodType(wrapper(index), attention)
        yield selected
        if any(index is None for index in selected):
            raise RuntimeError('Missing observation layer')
    finally:
        for attention, previous in zip(originals, saved):
            attention.forward = previous


def flatten_owned_cache(dense, indices):
    """Reshape in-place: no second list of views may keep consumed storage alive."""
    for i in range(len(dense.key_cache)):
        key, value, keep = dense.key_cache[i], dense.value_cache[i], indices[i]
        b, h, n, d = key.shape
        if value.shape != key.shape or keep.shape[:2] != (b, h):
            raise ValueError('Invalid batch/head geometry')
        dense.key_cache[i] = key.reshape(1, b * h, n, d)
        dense.value_cache[i] = value.reshape(1, b * h, n, d)
        indices[i] = keep.reshape(1, b * h, keep.shape[-1])


@torch.no_grad()
def memory_prefill(model, ids, options, runtime, progress, *, prefix_chunk_size=512):
    """Use 512 for speed; None preserves the original B1 quality prefix call."""
    from transformers.cache_utils import DynamicCache
    if ids.ndim != 2 or ids.shape[1] < 1:
        raise ValueError('Expected a nonempty [batch, prompt] tensor')
    if prefix_chunk_size is not None and prefix_chunk_size < 1:
        raise ValueError('prefix_chunk_size must be positive or None')
    length = ids.shape[1]
    observation = min(options['obs_window'], length)
    start = length - observation
    cache = DynamicCache()
    chunk = prefix_chunk_size or max(1, start)
    for i in range(0, start, chunk):
        end = min(i + chunk, start)
        progress['prefill_tokens'] = end
        _, cache, _ = runtime.forward(model, ids[:, i:end], cache, i, 'phasekv')
    originals = [layer.self_attn.original for layer in model.model.layers]
    mask = torch.zeros((1, 1, observation, length), device=ids.device, dtype=torch.float16)
    future = (torch.arange(length, device=ids.device)[None]
              > torch.arange(start, length, device=ids.device)[:, None])
    mask.masked_fill_(future[None, None], torch.finfo(torch.float16).min)
    del future
    with capture_indices(originals, model.config, length, options) as selected:
        logits, dense, attentions = runtime.forward(
            model, ids[:, start:], cache, start, 'phasekv',
            attention_mask=mask, output_attentions=False)
    if attentions is not None:
        raise RuntimeError('Unexpected retained attention tensors')
    del mask, cache, attentions
    progress['stage'] = 'select_and_pack_prompt'
    flatten_owned_cache(dense, selected)
    packing = {k: options[k] for k in ('prompt_bits', 'cold_bits', 'group_size',
               'recent_window', 'lazy_g', 'decode_keep_ratio', 'optimizations')}
    packed = PhaseKVCache.from_dense_prefill(
        dense, prompt_len=length, keep_indices=selected,
        consume_dense_cache=True, **packing)
    return logits, packed
