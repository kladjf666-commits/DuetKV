"""Same-card, uninstrumented Full FA2 vs the two optimized PhaseKV policies."""
import argparse
import json
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'base'))
import quick

RATIOS = {'Full_FA2': None, 'PhaseKV_80': .8, 'PhaseKV_100': 1.0}


def measure_arm(model, arm, options, sample, *, ratios=RATIOS):
    import torch
    import ablation_runtime as rt
    from PhaseKV.phasekv_cache import PhaseKVCache
    from transformers.cache_utils import DynamicCache

    ratio = ratios[arm]
    trace = rt.cache_trace

    def audit(cache, method):
        if ratio is None:
            if not isinstance(cache, DynamicCache) or method != 'full_fa2':
                raise RuntimeError('Expected the dense Full FA2 path')
            if len(cache.key_cache) != len(model.model.layers):
                raise RuntimeError('Incomplete dense KV cache')
        else:
            if type(cache) is not PhaseKVCache or cache.decode_keep_ratio != ratio:
                raise RuntimeError('Unexpected PhaseKV cache or retention ratio')
            if options['prefill_keep_ratio'] == 1.0 and any(
                    layer.prompt.length != len(sample['input_ids'])
                    or (layer.prompt.key.bits, layer.prompt.value.bits) != (options['prompt_bits'],) * 2
                    for layer in cache.layers):
                raise RuntimeError('Packed prompt differs from the full-prompt configuration')
            if cache.layers[0].cold and (cache._batched_cold_pool is not None) != (ratio == .8):
                raise RuntimeError('Unexpected PhaseKV cooling path')
        return trace(cache, method)

    with patch.object(rt, 'cache_trace', audit):
        settings = {} if ratio is None else dict(options, decode_keep_ratio=ratio)
        row = quick.measure(model, 'Full_FA2' if ratio is None else 'PhaseKV_GQA_on', settings, sample)
    final = row['final_trace']
    if ratio is None:
        if final['physical_tokens'] != len(sample['input_ids']) + row['decode_steps']:
            raise RuntimeError('Full FA2 did not retain all decoded positions')
    else:
        group_size = options['lazy_g']
        cooled = max(0, (row['decode_steps'] - options['recent_window']) // group_size)
        kept = max(1, round(group_size * ratio))
        expected = (cooled * kept, row['decode_steps'] - cooled * group_size)
        if (final['cold_tokens'], final['hot_tokens']) != expected:
            raise RuntimeError(f'Unexpected final PhaseKV token counts; expected {expected}')
    row.update(arm=arm, decode_keep_ratio=ratio, decode_s=sum(c['seconds'] for c in row['decode_chunks']),
               peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return row


def campaign(model, options, sample, out, rounds, protocol):
    prior = json.loads((HERE / 'prior_run.json').read_text(encoding='utf-8'))
    if sample['input_sha256'] != prior['input_sha256']:
        raise RuntimeError('Input differs from the preceding RTX 4090 comparison')
    path = out / 'summary.json'
    measure_case(model, options, sample, rounds, protocol, prior,
                 save=lambda report: quick.write_json(path, report),
                 expected_phase_outputs=prior['phase_output_sha256'])
    print('FINISHED', path, flush=True)


def chunk_summary(case):
    """Median latency in matching early and late windows across measured rounds."""
    expected = case['protocol']['output_tokens'] - 1
    by_arm = {arm: {'first_128': [], 'last_127': []} for arm in case['retention']}
    for row in case['runs']:
        if row['warmup']:
            continue
        cursor = 0
        chunks = row['decode_chunks']
        for chunk in chunks:
            if (chunk['from_decode_step'] != cursor or chunk['tokens'] <= 0 or chunk['seconds'] <= 0
                    or chunk['to_decode_step'] - cursor != chunk['tokens']):
                raise RuntimeError('Invalid per-window decode timing')
            cursor = chunk['to_decode_step']
        if cursor != expected or chunks[0]['tokens'] != 128 or chunks[-1]['tokens'] != 127:
            raise RuntimeError('Unexpected first or last decode timing boundary')
        for name, chunk in [('first_128', chunks[0]), ('last_127', chunks[-1])]:
            by_arm[row['arm']][name].append(1000 * chunk['seconds'] / chunk['tokens'])
    return {arm: {key: statistics.median(values) for key, values in windows.items()}
            for arm, windows in by_arm.items()}


def measure_case(model, options, sample, rounds, protocol, prior, *, save, expected_phase_outputs=None,
                 ratios=RATIOS):
    options = dict(options, optimizations=dict(options['optimizations'], gqa_reuse=True))
    same_card = protocol['gpu'].split(',')[0] == prior['gpu_uuid']
    signatures = {}
    paired = {arm: [] for arm in ratios if ratios[arm] is not None}
    report = dict(status='running', protocol=protocol, phase_options=options, retention=ratios,
                  rounds=rounds, previous_run=prior, same_gpu_as_previous_run=same_card,
                  output_hash_by_arm=signatures, runs=[], paired_vs_full=paired)
    save(report)
    names = list(ratios)
    for index in range(rounds + 1):
        # Rotate the order between formal rounds.
        offset = max(0, index - 1) % len(names)
        order = names[offset:] + names[:offset]
        group = {}
        for arm in order:
            row = measure_arm(model, arm, options, sample, ratios=ratios)
            signature = row['output_sha256']
            if signature != signatures.setdefault(arm, signature):
                raise RuntimeError(f'Output changed between repeats for {arm}')
            if expected_phase_outputs is not None and ratios[arm] is not None:
                expected = expected_phase_outputs[str(ratios[arm])]
                row['matches_previous_phase_output'] = signature == expected
                if same_card and signature != expected:
                    raise RuntimeError(f'{arm} output differs from the preceding run on this GPU')
            group[arm] = row
            report['runs'].append(dict(round=index, warmup=index == 0, **row))
            save(report)
            print(f'{"WARMUP" if not index else f"ROUND {index}/{rounds}"} {arm}: '
                  f'{row["tokens_per_s"]:.3f} token/s, {row["ms_per_token"]:.4f} ms/token', flush=True)
        if index:
            full = group['Full_FA2']
            for arm, values in paired.items():
                row = group[arm]
                gain = 100 * (row['tokens_per_s'] / full['tokens_per_s'] - 1)
                delta = row['ms_per_token'] - full['ms_per_token']
                values.append(dict(round=index, throughput_pct=gain, delta_ms_per_token=delta))
                print(f'PAIRED {index}/{rounds} {arm} vs Full_FA2: {gain:+.3f}% throughput, '
                      f'{delta:+.4f} ms/token', flush=True)
            save(report)
    formal = [r for r in report['runs'] if not r['warmup']]
    report['arm_medians'] = {arm: {key: statistics.median(r[key] for r in formal if r['arm'] == arm)
                                   for key in ('tokens_per_s', 'ms_per_token', 'ttft_s',
                                               'peak_allocated_bytes', 'final_kv_bytes')} for arm in names}
    report['median_vs_full'] = {}
    for arm, values in paired.items():
        result = {key: statistics.median(v[key] for v in values)
                  for key in ('throughput_pct', 'delta_ms_per_token')}
        result['faster_rounds'] = sum(v['throughput_pct'] > 0 for v in values)
        report['median_vs_full'][arm] = result
        print(f'RESULT {arm} vs Full_FA2: {result["throughput_pct"]:+.3f}% throughput, '
              f'{result["delta_ms_per_token"]:+.4f} ms/token, faster {result["faster_rounds"]}/{rounds}',
              flush=True)
    report['status'] = 'complete'
    save(report)
    return report


def main(run_campaign=campaign):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, choices=(3, 6), default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'run.claim').touch(exist_ok=False)
    try:
        quick.benchmark(args, args.out, expected_gpu='NVIDIA GeForce RTX 4090', run_campaign=run_campaign)
    except (Exception, KeyboardInterrupt) as exc:
        path = args.out / 'summary.json'
        state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        quick.write_json(path, state)
        raise


if __name__ == '__main__':
    main()
