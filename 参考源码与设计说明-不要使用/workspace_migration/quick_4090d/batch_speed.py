"""True fixed-length B2--B6 Llama decode on one RTX 4090."""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'base'))
import quick

ARMS = ('full_fa2', 'phasekv_triton')
BATCHES = range(2, 7)
PREFILL_CHUNK = 512


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


def prefill(model, ids, method, options, progress):
    from transformers.cache_utils import DynamicCache
    import ablation_runtime as rt
    cache = DynamicCache()
    for start in range(0, ids.shape[1], PREFILL_CHUNK):
        progress.update(stage='prefill', input_offset=start)
        logits, cache, _ = rt.forward(model, ids[:, start:start + PREFILL_CHUNK], cache, start, method)
    if method == 'phasekv_triton':
        progress['stage'] = 'pack_prompt'
        cache = packed_prompt(cache, ids.shape[1], options)
    return logits, cache


def measure(model, sample, batch, method, options, progress):
    import torch
    import ablation_runtime as rt
    ids = torch.tensor(sample['rows'][:batch], device='cuda', dtype=torch.long)
    length, output = ids.shape[1], sample['max_new_tokens']
    if ids.shape != (batch, sample['length_budget']) or length + output > model.config.max_position_embeddings:
        raise ValueError('Batch input shape or context length mismatch')
    for layer in model.model.layers:
        layer.self_attn.method = method
    packed = method == 'phasekv_triton'
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logits, cache = prefill(model, ids, method, options, progress)
    token = logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - started
    prefill_peak = torch.cuda.max_memory_allocated()
    progress['prefill_peak_allocated_bytes'] = prefill_peak
    if not torch.isfinite(logits).all().item():
        raise RuntimeError('Non-finite prefill logits')
    tokens = [token]
    torch.cuda.reset_peak_memory_stats()
    progress.update(stage='decode', step=0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, output):
        progress['step'] = step
        logits, cache, _ = rt.forward(model, token, cache, length + step - 1, method)
        if packed:
            cache.finish_decode_step()
        token = logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(token)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    decode_peak = torch.cuda.max_memory_allocated()
    progress['stage'] = 'verify_output'
    steps = output - 1
    absolute_length = cache.true_seq_length if packed else cache.get_seq_length()
    if not torch.isfinite(logits).all().item() or absolute_length != length + steps:
        raise RuntimeError('Invalid logits or final sequence length')
    if packed:
        cooled = max(0, (steps - options['recent_window']) // options['lazy_g'])
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        prompt_shape = (1, batch * model.config.num_key_value_heads, length, head_dim)
        for layer in cache.layers:
            if (layer.prompt.key.orig_shape != prompt_shape
                    or (layer.prompt.key.bits, layer.prompt.value.bits) != (4, 4)
                    or len(layer.cold) != cooled or any(s.length != 13 for s in layer.cold)
                    or layer.hot_len != steps - cooled * options['lazy_g']
                    or layer._decode_workspace is None or not layer._decode_workspace.compatible_4bit):
                raise RuntimeError('Packed prompt, cold segment or kernel layout mismatch')
    elif any(k.shape[:3] != (batch, model.config.num_key_value_heads, length + steps) for k in cache.key_cache):
        raise RuntimeError('Dense KV shape mismatch')
    output_ids = torch.cat(tokens, dim=1)
    if output_ids.shape != (batch, output):
        raise RuntimeError('Incomplete batched generation')
    storage = rt.cache_report(cache)
    return dict(batch=batch, input_tokens_per_sequence=length, output_tokens_per_sequence=output,
                decode_steps=steps, decoded_tokens=batch * steps, decode_s=seconds,
                aggregate_tokens_per_s=batch * steps / seconds, per_sequence_tokens_per_s=steps / seconds,
                ms_per_step=1000 * seconds / steps, ttft_s=ttft,
                prefill_peak_allocated_bytes=prefill_peak, decode_peak_allocated_bytes=decode_peak,
                peak_allocated_bytes=max(prefill_peak, decode_peak), final_kv_bytes=storage['physical_bytes'],
                output_sha256_by_row=[quick.digest(json.dumps(row).encode()) for row in output_ids.tolist()])


def run_sweep(model, samples, options, report, save, rounds, *, run, clean, oom_type):
    for sample in samples:
        shape = sample['task']
        batches = report['cases'].setdefault(shape, {})
        for batch in BATCHES:
            cell = batches.setdefault(str(batch), {'arms': {}})
            for index in range(rounds + 1):
                order = ARMS if index % 2 == 0 else ARMS[::-1]
                for arm in order:
                    entry = cell['arms'].setdefault(arm, dict(status='running', runs=[]))
                    if entry['status'] in ('oom', 'complete') or any(r['round'] == index for r in entry['runs']):
                        continue
                    phase = 'warmup' if index == 0 else f'round {index}/{rounds}'
                    progress = {'stage': 'setup'}
                    clean()
                    print(f'{shape} B{batch} {arm} {phase}', flush=True)
                    try:
                        row = run(model, sample, batch, arm, options, progress)
                    except oom_type as exc:
                        entry.update(status='oom', failed_round=index, oom_at=progress.copy(), error=str(exc))
                        print(f'OOM {shape} B{batch} {arm} {progress}', flush=True)
                    else:
                        entry['runs'].append(dict(round=index, warmup=index == 0, **row))
                        if index == rounds:
                            formal = [r for r in entry['runs'] if not r['warmup']]
                            entry['median'] = {k: statistics.median(r[k] for r in formal) for k in
                                               ('aggregate_tokens_per_s', 'per_sequence_tokens_per_s', 'ms_per_step',
                                                'ttft_s', 'prefill_peak_allocated_bytes', 'decode_peak_allocated_bytes',
                                                'peak_allocated_bytes', 'final_kv_bytes')}
                            entry['status'] = 'complete'
                        print(f'  {row["aggregate_tokens_per_s"]:.2f} total token/s, '
                              f'{row["peak_allocated_bytes"]/2**30:.2f} GiB peak', flush=True)
                    save()
            if all(cell['arms'][arm]['status'] == 'complete' for arm in ARMS):
                a, b = ([r for r in cell['arms'][arm]['runs'] if not r['warmup']] for arm in ARMS)
                cell['paired_phasekv_vs_full_pct'] = statistics.median(
                    100 * (y['aggregate_tokens_per_s'] / x['aggregate_tokens_per_s'] - 1) for x, y in zip(a, b))
            rates = [f'{arm} {cell["arms"][arm]["median"]["aggregate_tokens_per_s"]:.2f} token/s'
                     if cell['arms'][arm]['status'] == 'complete' else f'{arm} OOM' for arm in ARMS]
            print(f'RESULT {shape} B{batch}: ' + ' | '.join(rates), flush=True)
            save()
    report['oom_points'] = sum(a['status'] == 'oom' for bs in report['cases'].values()
                               for cell in bs.values() for a in cell['arms'].values())
    report['status'] = 'complete_with_oom' if report['oom_points'] else 'complete'
    save()


def campaign(model, options, base_sample, out, rounds, protocol):
    import torch
    from batch_check import validate
    prior = json.loads((HERE / 'quality_reference.json').read_text(encoding='utf8'))
    options = dict(options, prefill_keep_ratio=1., decode_keep_ratio=.8,
                   optimizations=dict(options['optimizations'], gqa_reuse=True))
    if options != prior['options'] or protocol['environment'] != prior['environment']:
        raise RuntimeError('PhaseKV policy or environment differs from the B1 run')
    samples = json.loads((HERE / 'batch_inputs.json').read_text(encoding='utf8'))['cases']
    if [(s['length_budget'], s['max_new_tokens']) for s in samples] != [(9192, 256), (8192, 2048)]:
        raise ValueError('Invalid benchmark cases')
    for s in samples:
        if (len(s['rows']) != 6 or len(set(s['row_sha256'])) != 6
                or s['rows'][0][:8192] != base_sample['input_ids'] or s['natural_eos']
                or any(len(r) != s['length_budget'] for r in s['rows'])
                or [quick.digest(json.dumps(r).encode()) for r in s['rows']] != s['row_sha256']):
            raise ValueError('Frozen batch inputs differ from the manifest')
    settings = dict(gpu=protocol['gpu'], model=protocol['model'], dtype=protocol['dtype'],
                    environment=protocol['environment'], options=options, rounds=rounds, batches=list(BATCHES),
                    warmups_per_point=1, prefill_chunk=PREFILL_CHUNK, row_sha256={s['task']:s['row_sha256'] for s in samples},
                    package_sha256=quick.digest((HERE / 'SHA256SUMS').read_bytes()))
    path = out / 'summary.json'
    report = json.loads(path.read_text(encoding='utf8')) if path.exists() else dict(protocol=settings, cases={})
    if report['protocol'] != settings:
        raise RuntimeError('Output directory contains a different protocol')
    def save():
        quick.write_json(path, report)
    def clean():
        gc.collect()
        torch.cuda.empty_cache()
    def attempt(model, sample, batch, arm, options, progress):
        try:
            return measure(model, sample, batch, arm, options, progress)
        except RuntimeError as exc:
            # Native FA2/CUDA can surface an allocation failure as RuntimeError.
            message = str(exc).lower()
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and not any(
                    text in message for text in ('out of memory', 'cudaerrormemoryallocation', 'cublas_status_alloc_failed')):
                raise
            progress.update(allocated_bytes=torch.cuda.memory_allocated(),
                            reserved_bytes=torch.cuda.memory_reserved())
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            raise torch.cuda.OutOfMemoryError(str(exc)) from exc
    report.update(status='running')
    report.pop('error', None)
    save()
    try:
        report['batch_check'] = validate(model, samples[0]['rows'], options, prefill, packed_prompt)
        save()
        run_sweep(model, samples, options, report, save, rounds, run=attempt, clean=clean,
                  oom_type=torch.cuda.OutOfMemoryError)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
        save()
        raise
    print('FINISHED', path, report['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, choices=(1, 3), default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    quick.benchmark(args, args.out, expected_gpu='NVIDIA GeForce RTX 4090', run_campaign=campaign)
