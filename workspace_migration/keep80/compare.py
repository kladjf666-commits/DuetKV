"""Compare completed records with bundled historical runs, without loading a model."""
from collections import defaultdict, Counter
import gzip
import json
from pathlib import Path
from statistics import mean, median


def report(out, here):
    hist = json.loads(gzip.decompress((here/'history.json.gz').read_bytes()))
    protocol = json.loads((out/'run_manifest.json').read_text('utf-8'))['protocol']
    if hist['frozen'] != protocol['frozen_data']:
        raise ValueError('Historical frozen inputs differ')
    with (out/'records.jsonl').open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    comparisons = {}
    for name, group in hist['groups'].items():
        baseline = {(r['id'],r['repeat']):r for r in group['rows']}
        if len(baseline) != len(group['rows']):
            raise ValueError(f'Duplicate historical results for {name}')
        quality, perf, counts = defaultdict(list), defaultdict(list), Counter()
        for r in rows:
            key = r['id'],r['repeat']
            old = baseline.get(key)
            if old is None:
                counts['reference_missing'] += 1
                continue
            if old['input_sha256'] != r['input_sha256']:
                raise ValueError(f'Unmatched input: {key}')
            counts[f"{r['status']}/{old['status']}"] += 1
            if r['suite']=='performance':
                perf[r['task'],r['id']].append((r,old))
            else:
                quality[r['suite'],r['task'],r['length_budget'],r['evaluation_split']].append((r,old))
        scores = []
        for (suite,task,budget,split),pairs in quality.items():
            ok = [(a,b) for a,b in pairs if a['status']==b['status']=='ok']
            if any(a['metric']!=b['metric'] for a,b in ok):
                raise ValueError('Quality scorer mismatch')
            scores.append(dict(suite=suite,task=task,budget=budget,split=split,
                               matched=len(pairs),both_ok=len(ok),metric=ok[0][0]['metric'] if ok else None,
                               candidate=mean(a['score'] for a,b in ok) if ok else None,
                               reference=mean(b['score'] for a,b in ok) if ok else None,
                               delta_points=mean(a['score']-b['score'] for a,b in ok) if ok else None,
                               cooled_pairs=sum(a['final_trace']['cold_segments']>0 for a,b in ok),
                               indexed_delta_points=(mean(a['indexed_accuracy_diagnostic']-b['indexed_accuracy_diagnostic']
                                   for a,b in ok) if ok and suite=='longgenbench' else None)))
        shapes = defaultdict(list)
        for (shape, sid),pairs in perf.items():
            shapes[shape]  # Preserve even a completely failed shape.
            if {a['repeat'] for a,b in pairs}!={0,1,2} or any(a['status']!=b['status'] or a['status']!='ok' for a,b in pairs):
                continue
            if any(a['output_tokens']!=b['output_tokens'] for a,b in pairs):
                raise ValueError(f'Fixed output length differs for {sid}')
            a,b = [median(p[i]['decode_tokens_per_s'] for p in pairs) for i in (0,1)]
            shapes[shape].append(dict(id=sid,candidate_tps=a,reference_tps=b,ratio=a/b))
        speed = [dict(shape=shape,complete_prompts=len(pairs),expected_prompts=3,
                      median_paired_change_pct=100*(median(p['ratio'] for p in pairs)-1) if pairs else None,
                      prompts=pairs) for shape,pairs in shapes.items()]
        comparisons[name] = dict(reference_gpu=group['gpu'],same_uuid=(group['gpu'][0]==protocol['gpu'].split(',')[0]),
                                 candidate_reference_status=counts,quality=scores,performance=speed)
    probe = dict(hist['optimized100_probe'])
    matches = [r for r in rows if r['id']==probe['id'] and r['status']=='ok']
    if any(r['input_sha256']!=probe['input_sha256'] or r['output_tokens']!=probe['output_tokens'] for r in matches):
        raise ValueError('Optimized 100% probe does not match the candidate input/output length')
    probe['candidate_tps'] = median(r['decode_tokens_per_s'] for r in matches) if len(matches)==3 else None
    probe['note'] = 'Single prompt B=1, historical 5-repeat probe and current 3-repeat formal timing harness.'
    probe['same_uuid'] = probe['gpu'].split(',')[0]==protocol['gpu'].split(',')[0]
    result = dict(candidate=protocol['method'],gpu=protocol['gpu'],historical=comparisons,
                  optimized100_probe=probe,notes=['Historical comparisons are cross-run.',
                  'Native D100 vs optimized D080 includes kernel layout and retention changes.',
                  'Quality uses successful pairs; speed uses all three successful repeats for each prompt.'])
    path=out/'comparison.json'
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)
