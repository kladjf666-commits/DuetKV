"""Four attention/cache ablations, isolated from the frozen native benchmark.

Only single-sequence FP16 inference is supported. Prefill is unchanged; all
four decode paths share projections, absolute RoPE, GQA layout and output code.
"""
import native_runtime as native

METHODS = ['full_fa2', 'phasekv_fa2', 'full_triton', 'phasekv_triton']


def cache_method(method):
    if method not in METHODS:
        raise ValueError(f'Unknown ablation: {method}')
    return 'phasekv' if method.startswith('phasekv_') else 'full'


def load_model(model_path, method, *, tiny=False):
    model, options, origin = native.load_model(model_path, cache_method(method), tiny=tiny)
    from ablation_attention import AblationAttention
    for layer in model.model.layers:
        original = getattr(layer.self_attn, 'original', layer.self_attn)
        layer.self_attn = AblationAttention(original, method)
    return model, options, origin + ['ablation_attention.py', 'ablation_kernels.py']


def reset_sample(model, method, options, output_limit):
    return native.reset_sample(model, cache_method(method), options, output_limit)


def forward(model, ids, cache, start, method, **kwargs):
    return native.forward(model, ids, cache, start, cache_method(method), **kwargs)


phase_prefill = native.phase_prefill


def cache_trace(cache, method):
    return native.cache_trace(cache, cache_method(method))


def cache_report(cache):
    report = native.cache_report(cache)
    # Dense split attention scratch belongs to this cache, not to the model.
    scratch = getattr(cache, '_ablation_scratch', {})
    extra = sum(t.numel() * t.element_size() for values in scratch.values() for t in values)
    report['auxiliary_bytes'] += extra
    report['total_owned_bytes'] += extra
    # Materialization buffers are per-layer temporaries, already included in
    # CUDA peak_allocated_bytes; never add them to persistent KV ownership.
    report['materialized_kv_bytes_max_layer'] = getattr(cache, '_materialized_peak', 0)
    return report
