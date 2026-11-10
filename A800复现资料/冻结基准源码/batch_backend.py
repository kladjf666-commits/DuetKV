def packed_prompt(dense, length, options):
    from PhaseKV.phasekv_cache import PhaseKVCache
    if options['prefill_keep_ratio'] != 1.:
        raise ValueError('This benchmark packs all prompt positions')
    for tensors in (dense.key_cache, dense.value_cache):
        for i, x in enumerate(tensors):
            tensors[i] = x.reshape(1, x.shape[0] * x.shape[1], length, x.shape[-1])
    packing = {k: options[k] for k in ('prompt_bits', 'cold_bits', 'group_size', 'recent_window',
                                       'lazy_g', 'decode_keep_ratio', 'optimizations')}
    return PhaseKVCache.from_dense_prefill(dense, prompt_len=length, **packing)
