"""Timing and storage instrumentation around the already validated native adapters.

No compression kernels or token-selection policies are modified. Detailed cache
traversal runs in a separate untimed diagnostic pass, not in timed decode loops.
"""
import time


def generate(model, method, options, sample, *, diagnostic=False):
    import torch
    from transformers.cache_utils import DynamicCache
    from native_runtime import reset_sample, forward, phase_prefill, cache_report, cache_trace
    effective, mode = reset_sample(model, method, options, sample['max_new_tokens'])
    ids = torch.tensor([sample['input_ids']], dtype=torch.long, device='cuda')
    length = ids.shape[1]
    limit = sample['max_new_tokens']
    if length + limit > model.config.max_position_embeddings:
        raise ValueError('Original model context exceeded')
    stops = set(sample['stop_ids']) if sample['natural_eos'] else set()
    checkpoints = []
    segments = []
    tokens = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        if method == 'phasekv':
            logits, cache = phase_prefill(model, ids, options)
        else:
            cache = None if method == 'kivi' else DynamicCache()
            logits, cache, _ = forward(model, ids, cache, 0, method)
        token = logits[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - started
        if not torch.isfinite(logits).all().item():
            raise RuntimeError('Non-finite prefill logits')
        prefill = cache_report(cache)
        prefill_trace = cache_trace(cache, method)
        tokens.append(token)
        last_id = int(token.item()) if stops else None
        checkpoint_peak = prefill['physical_bytes']
        if diagnostic:
            checkpoints.append({'generated_cached':0, **prefill})
        boundary = 0
        # Endpoints are output counts, not input context positions.
        boundaries = {512, 2048, 4096, limit}
        torch.cuda.synchronize()
        chunk_started = time.perf_counter()
        for step in range(1, limit):
            if stops and last_id in stops:
                break
            logits, cache, _ = forward(model, token, cache, length+step-1, method)
            if method == 'phasekv':
                cache.finish_decode_step()
            token = logits[:, -1].argmax(-1, keepdim=True)
            tokens.append(token)
            if stops:
                last_id = int(token.item())
            end = len(tokens) in boundaries or (stops and last_id in stops)
            # Every-token traversal gives a measured persistent KV storage peak.
            # Diagnostic timings are deliberately excluded from speed summaries.
            if diagnostic:
                current = cache_report(cache)
                checkpoint_peak = max(checkpoint_peak, current['physical_bytes'])
                if end:
                    checkpoints.append({'generated_cached':step, **current})
            if end or step % 128 == 0:
                torch.cuda.synchronize()
                now = time.perf_counter()
                segments.append({'from_decode_step':boundary, 'to_decode_step':step,
                                 'seconds':now-chunk_started, 'tokens':step-boundary})
                boundary = step
                print(f'[{method}/{sample["id"]}] {len(tokens)}/{limit} tokens', flush=True)
                chunk_started = time.perf_counter()
        torch.cuda.synchronize()
        remaining = len(tokens)-1-boundary
        if remaining:
            segments.append({'from_decode_step':boundary, 'to_decode_step':len(tokens)-1,
                             'seconds':time.perf_counter()-chunk_started, 'tokens':remaining})
        decode_s = sum(x['seconds'] for x in segments)
        generation_wall = time.perf_counter()-started
        if not torch.isfinite(logits).all().item():
            raise RuntimeError('Non-finite final logits')
        output_ids = torch.cat(tokens, dim=1)[0].cpu().tolist()
        final = cache_report(cache)
        result = {'input_tokens':length, 'output_tokens':len(output_ids),
                  'output_ids':output_ids, 'stopped_on_eos':bool(stops and output_ids[-1] in stops),
                  'hit_output_limit':len(output_ids)==limit and not bool(stops and output_ids[-1] in stops),
                  'ttft_s':ttft, 'prefill_and_compress_s':ttft, 'decode_s':decode_s,
                  'decode_tokens_per_s':(len(output_ids)-1)/decode_s if len(output_ids)>1 else None,
                  'generation_wall_s_including_instrumentation':generation_wall,
                  'prefill_storage':prefill, 'final_storage':final,
                  'prefill_trace':prefill_trace, 'final_trace':cache_trace(cache, method),
                  'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                  'decode_chunks':segments, 'effective_options':effective, 'scope_mode':mode,
                  'storage_checkpoints':checkpoints,
                  'persistent_kv_peak_bytes_diagnostic':max(checkpoint_peak,final['physical_bytes']) if diagnostic else None,
                  'diagnostic_pass':diagnostic}
    return result
