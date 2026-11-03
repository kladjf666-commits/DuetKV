"""Check request independence before accepting any batched throughput measurements."""
from types import SimpleNamespace


def check_cooling(pack, options):
    import torch
    from PhaseKV.kernels.segmented_attention import segmented_phasekv_decode
    batch, kv_heads, query_heads, dim = 5, 8, 32, 128
    rng = torch.Generator(device='cuda').manual_seed(31415)
    def rand(*shape):
        return torch.randn(*shape, device='cuda', dtype=torch.float16, generator=rng)
    keys = [rand(batch, kv_heads, 65, dim) for _ in range(2)]
    values = [rand(batch, kv_heads, 65, dim) for _ in range(2)]
    def make(start, end):
        dense = SimpleNamespace(key_cache=[k[start:end] for k in keys],
                                value_cache=[v[start:end] for v in values])
        return pack(dense, 65, options)
    caches = [(0, batch, make(0, batch))] + [(i, i + 1, make(i, i + 1)) for i in range(batch)]
    steps = options['recent_window'] + options['lazy_g']
    hot_k = rand(steps, 2, batch, kv_heads, 1, dim)
    hot_v = rand(steps, 2, batch, kv_heads, 1, dim)
    q = rand(batch, query_heads, dim)
    worst = 0.
    for step in range(steps + 1):
        if step in (1, steps):
            for layer_id in range(2):
                outputs = []
                for start, end, cache in caches:
                    heads = (end - start) * query_heads
                    layer = cache.layer_tiers(layer_id)
                    if step == steps and (len(layer.cold) != 1 or layer.cold[0].length != 13):
                        raise RuntimeError('Cooling validation did not exercise 16-to-13 packing')
                    context = segmented_phasekv_decode(q[start:end].reshape(heads, dim), layer, heads)
                    outputs.append(context.reshape(end - start, query_heads, dim))
                expected = torch.cat(outputs[1:])
                torch.testing.assert_close(outputs[0], expected, atol=.003, rtol=.01)
                worst = max(worst, (outputs[0] - expected).abs().max().item())
        if step < steps:
            for start, end, cache in caches:
                for layer_id in range(2):
                    k = hot_k[step, layer_id, start:end].reshape(1, -1, 1, dim)
                    v = hot_v[step, layer_id, start:end].reshape_as(k)
                    cache.append_hot(layer_id, k, v)
                cache.finish_decode_step()
    return worst


def validate(model, rows, options, prefill, pack):
    import torch
    import ablation_runtime as rt
    # Compare logits at the same position using a fixed next token for each row.
    ids = torch.tensor([r[:48] for r in rows], device='cuda', dtype=torch.long)
    def logits(x, forced, method):
        pre, cache = prefill(model, x, method, options, {})
        post, _, _ = rt.forward(model, forced, cache, x.shape[1], method)
        return torch.cat((pre, post), dim=1)
    worst = 0.
    forced = ids[:, -2:-1].contiguous()
    for method in ('full_fa2', 'phasekv_triton'):
        for layer in model.model.layers:
            layer.self_attn.method = method
        single = torch.cat([logits(x[None], y[None], method) for x, y in zip(ids, forced)])
        for batch in range(2, len(rows) + 1):
            result = logits(ids[:batch], forced[:batch], method)
            reference = single[:batch]
            errors = (result - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-6)
            if not torch.isfinite(errors).all().item() or errors.max().item() > .025:
                raise RuntimeError(f'{method} B{batch}: batched logits differ from independent requests')
            worst = max(worst, errors.max().item())
    cold_error = check_cooling(pack, options)
    print(f'Batch validation passed: logits relative L2 {worst:.6f}; '
          f'packed attention max absolute error {cold_error:.6f}', flush=True)
    return dict(status='passed', batch_sizes=list(range(2, len(rows) + 1)), logits_max_relative_l2=worst,
                packed_attention_before_after_cooling_max_abs=cold_error)
