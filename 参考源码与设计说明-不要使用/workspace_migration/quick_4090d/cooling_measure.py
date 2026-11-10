"""Compare matched hot lengths around the first 16-token cooling operation."""
import contextlib
import gzip
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from cooling_trace import ActivityTrace

WINDOWS = {'reference129': (129, 129), 'before': (130, 143), 'cooling144': (144, 144),
           'first_cold145': (145, 145), 'after': (146, 159)}
STAGES = {0: 'model_other', 1: 'cache_append', 2: 'cache_finish',
          3: 'workspace', 4: 'attention_dispatch'}


def summarize_trace(raw):
    by_step = {step: dict(gpu=defaultdict(float), host=defaultdict(float),
                         counts=defaultdict(int), spans=[]) for step in range(129, 160)}
    for event in raw['events']:
        row = by_step[event['step']]
        name = event['name']
        category = ('attention_main' if name == '_segmented_attention' else
                    'attention_merge' if name == '_merge_segments' else STAGES[event['stage']])
        row['gpu'][category] += (event['end'] - event['start']) / 1e6
        row['counts'][category] += 1
        row['spans'].append((event['start'], event['end']))
    for event in raw['host']:
        row = by_step[event['step']]
        stage = STAGES[event['stage']]
        row['host'][stage] += event['ns'] / 1e6
        row['counts']['host_' + stage] += 1
    for step, row in by_step.items():
        for category, count in {'attention_main': 32, 'attention_merge': 32,
                                'host_cache_append': 32, 'host_cache_finish': 1,
                                'host_workspace': 32, 'host_attention_dispatch': 32}.items():
            if row['counts'][category] != count:
                raise RuntimeError(f'Incomplete step {step}: {category} {row["counts"][category]} != {count}')
        covered, cursor = 0, 0
        for start, end in sorted(row.pop('spans')):
            covered += max(0, end - max(cursor, start))
            cursor = max(cursor, end)
        row['gpu_union_ms'] = covered / 1e6
        row['gpu']['attention_total'] = row['gpu']['attention_main'] + row['gpu']['attention_merge']
        row['gpu']['cache_total'] = sum(row['gpu'][k] for k in ('cache_append', 'cache_finish', 'workspace'))
        row['host']['cache_total'] = sum(row['host'][k] for k in ('cache_append', 'cache_finish', 'workspace'))
    windows = {}
    for label, (start, end) in WINDOWS.items():
        values = [by_step[s] for s in range(start, end + 1)]
        windows[label] = {domain: {key: statistics.mean(v[domain].get(key, 0) for v in values)
                         for key in set().union(*(v[domain] for v in values))}
                         for domain in ('gpu', 'host')}
    return dict(steps=by_step, windows=windows, dropped=raw['dropped'])


def labels(trace):
    import ablation_kernels
    import PhaseKV.kernels.segmented_attention as segmented
    from PhaseKV.phasekv_cache import PhaseKVCache

    def wrap(function, stage):
        def tagged(*args, **kwargs):
            with trace.scope(stage):
                return function(*args, **kwargs)
        return tagged

    stack = contextlib.ExitStack()
    for owner, name, stage in ((PhaseKVCache, 'append_hot', 1), (PhaseKVCache, 'finish_decode_step', 2),
                               (segmented, 'ensure_workspace', 3), (ablation_kernels, 'segmented_phasekv_decode', 4)):
        stack.enter_context(patch.object(owner, name, wrap(getattr(owner, name), stage)))
    return stack


def generation(model, sample, options, method, trace=None):
    import ablation_runtime as rt
    from transformers.cache_utils import DynamicCache

    for layer in model.model.layers:
        layer.self_attn.method = method
    rt.reset_sample(model, method, options, 256)
    ids = torch.tensor([sample['input_ids']], device='cuda')
    if method == 'phasekv_triton':
        logits, cache = rt.phase_prefill(model, ids, options)
    else:
        logits, cache, _ = rt.forward(model, ids, DynamicCache(), 0, method)
    token = logits[:, -1].argmax(-1, keepdim=True)
    generated = [token]

    def advance(step):
        nonlocal token
        logits, _, _ = rt.forward(model, token, cache, ids.shape[1] + step - 1, method)
        if method == 'phasekv_triton':
            cache.finish_decode_step()
        token = logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token)

    for step in range(1, 129):
        advance(step)
    torch.cuda.synchronize()
    if trace:
        trace.start()
    wall, states = {}, []
    with labels(trace) if trace else contextlib.nullcontext():
        for name, (start, end) in WINDOWS.items():
            torch.cuda.synchronize()
            started = time.perf_counter()
            for step in range(start, end + 1):
                if trace:
                    trace.step = step
                    with trace.scope(0):
                        advance(step)
                else:
                    advance(step)
            torch.cuda.synchronize()
            wall[name] = 1000 * (time.perf_counter() - started) / (end - start + 1)
            if method == 'phasekv_triton':
                cold = len(cache.layers[0].cold)
                hot = cache.layers[0].hot_len
                if cold != max(0, (end - 128) // 16) or hot != end - 16 * cold:
                    raise RuntimeError(f'Unexpected cold/hot state after step {end}: {cold}, {hot}')
                states.append(dict(end_step=end, cold_segments=cold, hot_tokens=hot))
    if trace:
        trace.stop()
    for step in range(160, 256):
        advance(step)
    torch.cuda.synchronize()
    for layer in getattr(cache, 'layers', []):
        if len(layer.cold) != 7 or any(seg.length != 13 for seg in layer.cold):
            raise RuntimeError('Final cache differs from keep80 reference')
    output = torch.cat(generated, dim=1)[0].tolist()
    return dict(wall_ms=wall, output_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest(),
                cache_states=states)


def run(model, sample, options, historical, output):
    from quick import write_json, measure
    protocol = dict(base=historical['protocol'], gqa_reuse=True, windows=WINDOWS,
                    definition='Decode step 1 consumes the first output token; cooling happens after attention in step 144. '
                    'Step 145 first reads the new cold block. Windows 130-143 and 146-159 have matched hot lengths.',
                    instrumentation='CUPTI external correlation and host scopes in separate trace runs; '
                    'CPU and GPU durations overlap and must not be added. Main/merge counts checked per step.')
    report = dict(status='running', protocol=protocol, plain=[], traced=[])
    write_json(output / 'cooling_summary.json', report)
    reference = {row['arm']: row['output_sha256'] for row in historical['runs'] if not row['warmup']}

    for arm in ('Full_FA2', 'PhaseKV_GQA_on'):
        row = measure(model, arm, options, sample)
        if row['output_sha256'] != reference[arm]:
            raise RuntimeError(f'Warmup output differs from original speed test: {arm}')
    print('Warmup and original-output parity passed.', flush=True)
    for repeat in range(3):
        arms = ('PhaseKV_GQA_on', 'Full_FA2') if repeat % 2 == 0 else ('Full_FA2', 'PhaseKV_GQA_on')
        for arm in arms:
            method = 'phasekv_triton' if arm.startswith('PhaseKV') else 'full_fa2'
            row = generation(model, sample, options if method.startswith('phasekv') else {}, method)
            if row['output_sha256'] != reference[arm]:
                raise RuntimeError('Unprofiled greedy output changed')
            report['plain'].append(dict(repeat=repeat, arm=arm, **row))
        write_json(output / 'cooling_summary.json', report)
        print('Unprofiled paired repeat', repeat + 1, 'complete.', flush=True)
    for repeat in range(3):
        trace = ActivityTrace()
        row = generation(model, sample, options, 'phasekv_triton', trace)
        if row['output_sha256'] != reference['PhaseKV_GQA_on']:
            raise RuntimeError('Instrumented output changed')
        raw = trace.finish()
        with gzip.open(output / f'activity_{repeat}.json.gz', 'wt', encoding='utf-8', compresslevel=1) as handle:
            json.dump(raw, handle, separators=(',', ':'))
        report['traced'].append(dict(repeat=repeat, **row, **summarize_trace(raw)))
        write_json(output / 'cooling_summary.json', report)
        print('CUPTI repeat', repeat + 1, 'complete, no dropped records.', flush=True)
    report['status'] = 'complete'
    for domain in ('gpu', 'host'):
        report[domain + '_median_ms'] = {label: {key: statistics.median(
            run['windows'][label][domain].get(key, 0) for run in report['traced'])
            for key in set().union(*(run['windows'][label][domain] for run in report['traced']))}
            for label in WINDOWS}
    report['contrasts'] = {}
    for before, after in (('before', 'after'), ('before', 'cooling144'), ('reference129', 'first_cold145')):
        report['contrasts'][after + '_minus_' + before] = {
            domain: {key: statistics.median(run['windows'][after][domain].get(key, 0) -
                      run['windows'][before][domain].get(key, 0) for run in report['traced'])
                     for key in ('attention_total', 'cache_append', 'cache_finish', 'workspace', 'cache_total')}
            for domain in ('gpu', 'host')}
    write_json(output / 'cooling_summary.json', report)
    print('FINISHED', output / 'cooling_summary.json', flush=True)
