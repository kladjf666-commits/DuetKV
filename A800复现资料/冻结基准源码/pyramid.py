def prefill(model, ids, options):
    """Native dense FA2 prefix + native eager last-observation chunk for scores.

    PhaseKV selects only the final 32 query rows, so no N x N tensor is required.
    Explicit causal mask preserves absolute-position semantics in the tail.
    """
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaAttention
    from PhaseKV.phasekv_cache import PhaseKVCache
    from pyramid_source import PyramidKVCompressor
    from types import SimpleNamespace, MethodType
    from native_runtime import forward
    length = ids.shape[1]
    observation = min(options['window_size'], length)
    start = length - observation
    cache = DynamicCache()
    if start:
        _, cache, _ = forward(model, ids[:, :start], cache, 0, 'phasekv')
    originals = [layer.self_attn.original for layer in model.model.layers]
    saved = [attention.forward for attention in originals]
    mask = torch.zeros((1, 1, observation, length), device=ids.device, dtype=torch.float16)
    future = torch.arange(length, device=ids.device)[None] > torch.arange(start, length, device=ids.device)[:, None]
    mask.masked_fill_(future[None, None], torch.finfo(torch.float16).min)
    try:
        for attention in originals:
            attention.forward = MethodType(LlamaAttention.forward, attention)
        logits, dense, attentions = forward(model, ids[:, start:], cache, start, 'phasekv',
                                             attention_mask=mask, output_attentions=True)
    finally:
        for attention, previous in zip(originals, saved):
            attention.forward = previous
    policy=PyramidKVCompressor(SimpleNamespace(max_cache_tokens=options['max_capacity_prompt'],prefill_keep_ratio=None,extra=dict(obs_window=observation,pool_kernel=options['kernel_size'],pooling=options['pooling'],beta=options['beta'])))
    return logits,policy.on_prefill(dense,attentions=attentions,prompt_len=length,model_config=model.config)
