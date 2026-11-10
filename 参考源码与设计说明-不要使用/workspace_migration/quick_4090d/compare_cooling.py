"""Compare the frozen and updated cache paths on the same resident Llama."""
import argparse
import contextlib
import hashlib
import statistics
from pathlib import Path
from unittest.mock import patch

import cooling_cpu


def compare(model, sample, options, historical, out, pairs, clocks):
    reference, = {row['output_sha256'] for row in historical['runs']
                  if row['arm'] == 'PhaseKV_GQA_on' and not row['warmup']}
    run_comparison(model, sample, options, historical['protocol'], out, pairs, clocks,
                   expected_keep80=reference)


def compare_retention(model, options, sample, out, rounds, protocol):
    options = dict(options, optimizations=dict(options['optimizations'], gqa_reuse=True))
    clocks = cooling_cpu.clock_details()
    if not all(clock['suitable_for_short_calls'] for clock in clocks.values()):
        raise RuntimeError(f'Clock resolution is insufficient for short CPU scopes: {clocks}')
    run_comparison(model, sample, options, protocol, out, rounds, clocks, include_keep100=True)


def differences(rows, baseline, candidate):
    speed = rows['whole_decode']
    old, new = speed[baseline], speed[candidate]
    delta = {'decode_ms_per_token': new['ms_per_token'] - old['ms_per_token'],
             'throughput_pct': 100 * (new['tokens_per_s'] / old['tokens_per_s'] - 1),
             'peak_allocated_bytes': new['peak_allocated_bytes'] - old['peak_allocated_bytes']}
    if 'final_kv_bytes' in old:
        delta['final_kv_bytes'] = new['final_kv_bytes'] - old['final_kv_bytes']
    timed = rows['host_timed']
    for stage, target, reference in (('cache_finish', 'cooling144', 'before'),
                                     ('workspace', 'first_cold145', 'reference129')):
        for domain in cooling_cpu.DOMAINS:
            a, b = timed[baseline]['windows'], timed[candidate]['windows']
            delta[f'{stage}_{domain}'] = ((b[target][domain][stage] - b[reference][domain][stage])
                                          - (a[target][domain][stage] - a[reference][domain][stage]))
    return delta


def whole_decode_medians(runs):
    grouped = {}
    for row in runs:
        if row['mode'] == 'whole_decode' and not row['warmup']:
            grouped.setdefault(row['variant'], []).append(row)
    return {variant: {metric: statistics.median(row[metric] for row in rows)
                      for metric in ('ms_per_token', 'tokens_per_s', 'peak_allocated_bytes', 'final_kv_bytes')
                      if metric in rows[0]} for variant, rows in grouped.items()}


def run_comparison(model, sample, options, protocol, out, pairs, clocks, *,
                   expected_keep80=None, include_keep100=False):
    import ablation_attention
    import ablation_runtime as rt
    import native_runtime
    import quick
    import torch
    import PhaseKV.phasekv_cache as cache_module
    import PhaseKV.kernels.segmented_attention as attention

    here = Path(__file__).resolve().parent
    candidate = native_runtime.module_at('PhaseKV._updated_cache', here / 'phasekv_cache.py')
    plan = native_runtime.module_at('PhaseKV.kernels._updated_plan', here / 'segmented_plan.py')
    variants = {'old': (cache_module.PhaseKVCache, attention.ensure_workspace, .8),
                'new': (candidate.PhaseKVCache, plan.ensure_workspace, .8)}
    comparisons = {'new80_minus_old80': ('old', 'new')}
    if include_keep100:
        variants['new100'] = (candidate.PhaseKVCache, plan.ensure_workspace, 1.)
        comparisons['new80_minus_new100'] = ('new100', 'new')
    signatures = {} if expected_keep80 is None else {'0.8': expected_keep80}
    report = dict(status='running', base_protocol=protocol, options=options,
                  arms={name: {'version': 'old' if name == 'old' else 'new', 'decode_keep_ratio': ratio,
                               'retained_per_16': round(16 * ratio)} for name, (_, _, ratio) in variants.items()},
                  comparison_arms={k: dict(baseline=a, candidate=b) for k, (a, b) in comparisons.items()},
                  candidate_sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                                    for name in ('phasekv_cache.py', 'segmented_plan.py')},
                  clocks=clocks, pairs=pairs, reference_by_retention=signatures, runs=[])
    if expected_keep80:
        report['expected_output_sha256'] = expected_keep80
    if include_keep100:
        report['retention_comparison'] = ('Updated code in both arms. Keep80 selects 13/16 tokens per KV head '
                                          'and packs across layers. Keep100 retains 16/16 and packs per layer.')
    path = out / 'summary.json'
    cooling_cpu.write_json(path, report)

    def measure(variant, mode):
        cls, ensure, ratio = variants[variant]
        settings = dict(options, decode_keep_ratio=ratio)
        original_trace = rt.cache_trace

        def check_path(cache, method):
            if type(cache) is not cls or cache.decode_keep_ratio != ratio:
                raise RuntimeError('Unexpected cache class or retention ratio')
            if variant != 'old' and cache.layers[0].cold:
                if (cache._batched_cold_pool is not None) != (ratio == .8):
                    raise RuntimeError(f'Unexpected cooling path for {variant}')
            return original_trace(cache, method)

        with contextlib.ExitStack() as stack:
            for owner, name, value in ((cache_module, 'PhaseKVCache', cls),
                                       (ablation_attention, 'PhaseKVCache', cls),
                                       (attention, 'ensure_workspace', ensure),
                                       (rt, 'cache_trace', check_path)):
                stack.enter_context(patch.object(owner, name, value))
            if mode == 'whole_decode':
                row = quick.measure(model, 'PhaseKV_GQA_on', settings, sample)
                row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            else:
                row = cooling_cpu.generation(model, sample, settings, instrumented=True)
        if row['output_sha256'] != signatures.setdefault(str(ratio), row['output_sha256']):
            raise RuntimeError(f'{variant}/{mode} output differs from the {ratio:g} retention reference')
        return row

    try:
        paired = {label: [] for label in comparisons}
        for pair in range(pairs + 1):
            rows = {}
            names = tuple(variants)
            offset = max(0, pair - 1) // 2 % len(names)
            order = names[offset:] + names[:offset]
            if pair > 0 and pair % 2 == 0:
                order = order[::-1]
            for mode, sequence in (('whole_decode', order), ('host_timed', reversed(order))):
                rows[mode] = {}
                for variant in sequence:
                    row = measure(variant, mode)
                    rows[mode][variant] = row
                    report['runs'].append(dict(pair=pair, warmup=pair == 0, variant=variant, mode=mode,
                                               decode_keep_ratio=variants[variant][2], **row))
                    cooling_cpu.write_json(path, report)
            if pair == 0:
                print('WARMUP passed for all arms and measurement modes.', flush=True)
                continue
            report['paired_comparisons'] = paired
            report['median_comparisons'] = {}
            for label, (baseline, updated) in comparisons.items():
                delta = differences(rows, baseline, updated)
                paired[label].append(delta)
                report['median_comparisons'][label] = {
                    name: statistics.median(item[name] for item in paired[label]) for name in delta}
                print(f'ROUND {pair}/{pairs} {label} throughput {delta["throughput_pct"]:+.2f}% '
                      f'decode {delta["decode_ms_per_token"]:+.3f} ms/token', flush=True)
            report['paired_new_minus_old'] = paired['new80_minus_old80']
            report['median_new_minus_old'] = report['median_comparisons']['new80_minus_old80']
            report['whole_decode_medians'] = whole_decode_medians(report['runs'])
            cooling_cpu.write_json(path, report)
        report['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        cooling_cpu.write_json(path, report)
    for name, row in report['whole_decode_medians'].items():
        print(f'{name} keep={variants[name][2]:g} {row["tokens_per_s"]:.2f} token/s, '
              f'{row["ms_per_token"]:.3f} ms/token', flush=True)
    print('FINISHED', path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--pairs', '--rounds', type=int, choices=(2, 4, 6))
    parser.add_argument('--rtx4090-retention', action='store_true')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--project', type=Path, default=Path('/data/phasekv_4090d_speed'))
    parser.add_argument('--history', type=Path, default=Path(
        '/data/phasekv-results/4090d_speed_20261101_221816_361/summary.json'))
    args = parser.parse_args()
    if args.pairs is None:
        args.pairs = 2 if args.rtx4090_retention else 4
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'run.claim').open('x') as claim:
        claim.write('cooling comparison\n')
    if args.rtx4090_retention:
        import sys
        if args.model is None:
            parser.error('--model is required for the RTX 4090 comparison')
        sys.path.insert(0, str(args.project.resolve()))
        import quick
        args.rounds = args.pairs
        quick.benchmark(args, args.out, expected_gpu='NVIDIA GeForce RTX 4090', run_campaign=compare_retention)
    else:
        cooling_cpu.benchmark(args, run_campaign=compare)
