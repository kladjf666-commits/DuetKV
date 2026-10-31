"""Paired contrasts: never replace OOMs with artificial throughput values."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

CONTRASTS = [
    ('phasekv_fa2', 'full_fa2', 'cache_effect_with_FA2_including_materialization'),
    ('phasekv_triton', 'full_triton', 'cache_effect_with_common_Triton'),
    ('full_triton', 'full_fa2', 'attention_effect_with_full_cache'),
    ('phasekv_triton', 'phasekv_fa2', 'fused_vs_materialized_with_PhaseKV'),
]


def paired_report(out, rows):
    perf = defaultdict(list)
    quality = {}
    for row in rows:
        if row['suite'] == 'performance':
            perf[(row['method'], row['id'])].append(row)
        elif row['status'] == 'ok' and 'score' in row:
            quality[(row['method'], row['id'])] = row
    contrasts = []
    for candidate, baseline, label in CONTRASTS:
        paired = []
        skipped = []
        ids = sorted({key[1] for key in perf if key[0] in (candidate, baseline)})
        for sample_id in ids:
            a, b = perf.get((candidate, sample_id), []), perf.get((baseline, sample_id), [])
            if any(len(x) != 3 or {r['repeat'] for r in x} != {0, 1, 2}
                   or any(r['status'] != 'ok' or not r.get('decode_tokens_per_s') for r in x)
                   for x in (a, b)):
                skipped.append(sample_id)
                continue
            if (len({r['input_sha256'] for r in a+b}) != 1
                    or len({r['output_tokens'] for r in a+b}) != 1):
                raise RuntimeError('Performance pairs must have identical inputs and output lengths')
            sa, sb = [statistics.median(r['decode_tokens_per_s'] for r in x) for x in (a, b)]
            paired.append({'id': sample_id, 'shape': a[0]['task'],
                           'candidate_tps': sa, 'baseline_tps': sb, 'speedup': sa/sb})
        shapes = defaultdict(list)
        for row in paired:
            shapes[row['shape']].append(row['speedup'])
        qgroups = defaultdict(list)
        for (method, sample_id), a in quality.items():
            b = quality.get((baseline, sample_id))
            if method != candidate or b is None:
                continue
            if a['input_sha256'] != b['input_sha256']:
                raise RuntimeError('Quality input pairing mismatch')
            qgroups[(a['suite'], a['task'], a.get('length_budget'))].append(a['score']-b['score'])
        contrasts.append({'candidate': candidate, 'baseline': baseline, 'contrast': label,
                          'performance_pairs': paired, 'skipped_incomplete_or_failed_pairs': skipped,
                          'shape_speedups': [{'shape': k, 'paired_prompts': len(v),
                                              'median_speedup': statistics.median(v)} for k,v in shapes.items()],
                          'quality_successful_pairs': [{'suite': k[0], 'task': k[1], 'budget': k[2],
                                                        'paired_samples': len(v), 'mean_delta_points': statistics.mean(v)}
                                                       for k,v in qgroups.items()]})
    (Path(out)/'ablation_pairs.json').write_text(json.dumps({
        'contrasts': contrasts,
        'notes': ['Speed uses three successful repeats per identical prompt and fixed output length.',
                  'PhaseKV+FA2 includes per-step dequantization and copying in timed decode.',
                  'PhaseKV+FA2 rounds unpacked K/V to FP16; fused attention dequantizes in FP32.',
                  'Full prefill is FA2; PhaseKV prefill adds the original eager observation/selection/packing.',
                  'The common Triton framework is an ablation control, not a replacement for optimized Full+FA2.',
                  'Failed pairs are excluded explicitly; quality metrics are task specific.']
    }, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
