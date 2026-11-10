"""Read immutable A/B pairs; formal scores never include diagnostic passes."""
from statistics import median
import common as c


def validate_pair(out, row, experiment, case, kind, round_index=None):
    c.validate_saved(row, experiment)
    if row['case'] != list(case) or row['kind'] != kind:
        raise ValueError('Pair case/kind changed')
    if kind == 'formal' and row['formal_round'] != round_index:
        raise ValueError('Pair round changed')
    if set(row['arms']) != set(c.ARMS) or set(row['order']) != set(c.ARMS) or len(row['order']) != 2:
        raise ValueError('Incomplete A/B pair')
    gate = c.validate_saved(c.read(out / f'gates/{row["session"]}.json'), experiment)
    if gate['gpu_uuid'] != row['gpu_uuid'] or gate['session'] != row['session']:
        raise ValueError('Pair gate/GPU identity changed')
    for arm, value in row['arms'].items():
        if value['status'] != 'ok' or value['arm'] != arm or value['gpu_uuid'] != row['gpu_uuid']:
            raise ValueError('Pair arm identity/status changed')
        if value['input_tokens'] != case[0] or value['batch'] != case[2]:
            raise ValueError('Pair input geometry changed')
        if value['output_tokens_per_row'] != (case[1] if kind == 'formal' else 1):
            raise ValueError('Pair output length changed')
        if value['effective_options'] != c.options(c.PROFILE):
            raise ValueError('Pair settings changed')
    left, right = (row['arms'][arm] for arm in c.ARMS)
    if left['input_sha256'] != right['input_sha256']:
        raise ValueError('A/B input hashes differ')
    for key in ('output_sha256', 'first_token_sha256') if kind == 'formal' else ('first_token_sha256','packed_prompt_sha256'):
        if left[key] != right[key]:
            raise ValueError('A/B output/cache mismatch: ' + key)
    return row


def summarize(rows):
    return dict(formal_runs=len(rows), ttft_s=median(r['ttft_s'] for r in rows),
        ttft_min_s=min(r['ttft_s'] for r in rows), ttft_max_s=max(r['ttft_s'] for r in rows),
        decode_tps=median(r['aggregate_tokens_per_s'] for r in rows),
        e2e_tps=median(r['batch']*r['output_tokens_per_row']/(r['ttft_s']+r['decode_s']) for r in rows),
        peak_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30)


def report(out):
    if not (out / 'plan.json').exists():
        return dict(complete=False, state='not_prepared')
    plan = c.read(out / 'plan.json')
    result = dict(experiment=plan['experiment'], complete=False, formal_pairs=0, diagnostic_pairs=0,
                  speed={}, diagnostics={}, output_matches=True)
    text = ['A800 TTFT paired retest: same GPU, same loaded model',
            'FORMAL: plain in-memory progress; no phase instrumentation or status/checkpoint writes in timed regions.',
            'Two cases; 3 A/B pairs each; 1 warmup per arm/case on each unfinished-session visit.',
            'Allocator release/empty_cache policy remains identical for both arms.', '']
    sessions = set(); all_gpus = set()
    for case in c.CASES:
        name = c.case_name(case); pairs = []
        for index in (1,2,3):
            path = out / f'pairs/{name}/r{index}.json'
            if not path.exists(): continue
            pair = validate_pair(out, c.read(path), plan['experiment'], case, 'formal', index)
            sessions.add(pair['session']); all_gpus.add(pair['gpu_uuid']); pairs.append(pair)
        if pairs:
            if len({p['arms']['baseline']['output_sha256'] for p in pairs}) != 1:
                raise ValueError('Repeated formal outputs diverged: ' + name)
            values = {arm:summarize([p['arms'][arm] for p in pairs]) for arm in c.ARMS}
            values['paired_ttft_change_pct'] = [
                100*(p['arms']['optimized']['ttft_s']/p['arms']['baseline']['ttft_s']-1) for p in pairs]
            values['median_paired_ttft_change_pct'] = median(values['paired_ttft_change_pct'])
            result['speed'][name] = values; result['formal_pairs'] += len(pairs)
            text.append(name + f' : {len(pairs)}/3 formal pairs')
            for arm in c.ARMS:
                r=values[arm]
                text.append(f'  {arm}: TTFT={r["ttft_s"]:.6f}s [{r["ttft_min_s"]:.6f}, {r["ttft_max_s"]:.6f}], '
                    f'decode={r["decode_tps"]:.4f} tok/s; E2E={r["e2e_tps"]:.4f}; peak={r["peak_gib"]:.4f} GiB')
            text.append(f'  median paired TTFT change={values["median_paired_ttft_change_pct"]:+.2f}% (negative is faster)')
        path = out / f'diagnostics/{name}.json'
        if path.exists():
            row = validate_pair(out,c.read(path),plan['experiment'],case,'diagnostic')
            sessions.add(row['session']); all_gpus.add(row['gpu_uuid'])
            if pairs and row['arms']['baseline']['first_token_sha256'] != pairs[0]['arms']['baseline']['first_token_sha256']:
                raise ValueError('Diagnostic first token differs from formal run')
            result['diagnostic_pairs'] += 1
            result['diagnostics'][name] = {arm:row['arms'][arm]['phase_timings'] for arm in c.ARMS}
            text.append('  SEPARATE DIAGNOSTIC CUDA-stream spans (ms), baseline -> optimized:')
            for stage in ('prefix_forward','observation_and_selection','selection','flatten','packing','first_token'):
                a,b=(row['arms'][arm]['phase_timings']['stages'][stage]['cuda_stream_ms'] for arm in c.ARMS)
                text.append(f'    {stage}: {a:.3f} -> {b:.3f}')
    if len(all_gpus)>1: raise ValueError('Results mix physical GPUs')
    result.update(sessions=sorted(sessions), gpu_uuids=sorted(all_gpus),
                  complete=result['formal_pairs']==6 and result['diagnostic_pairs']==2)
    text += ['', 'DIAGNOSTICS: one extra prefill-only pass per arm/case; excluded from formal medians and peaks.',
             'Selection is INCLUDED in observation_and_selection; do not add it twice.',
             'CUDA-stream spans can include CPU launch gaps; they are not pure summed kernel execution times.',
             'Checkpoint/status I/O durations and before/after GPU/host snapshots are collected outside timing.',
             f'Process sessions: {len(sessions)}. Each committed pair always uses one process; a resumed campaign may span sessions.',
             'COMPLETE' if result['complete'] else 'INCOMPLETE']
    c.write(out / 'summary.json',result)
    (out / 'summary.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result
