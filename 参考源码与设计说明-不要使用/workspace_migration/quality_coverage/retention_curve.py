"""Paired, per-dataset summaries on the predeclared prompt-retention grid."""
from collections import Counter
from math import isclose
from statistics import mean

GRID = (70, 75, 80, 85, 90, 95, 100)
ORDER = (80, 90, 75, 85, 95)
TASKS = ('gsm8k', 'csqa', 'mmlu')


def long_repeat(ids):
    """At least 128 consecutive tokens with a period of 1 through 16."""
    for period in range(1, 17):
        matched = 0
        for i in range(period, len(ids)):
            matched = matched + 1 if ids[i] == ids[i-period] else 0
            if matched + period >= 128:
                return True
    return False


def flip_pvalue(differences):
    """Exact one-sided paired sign-flip test on integer correct-count differences."""
    counts = Counter({0: 1})
    for delta in (abs(d) for d in differences if d):
        next_counts = Counter()
        for total, weight in counts.items():
            next_counts[total + delta] += weight
            next_counts[total - delta] += weight
        counts = next_counts
    observed = sum(differences)
    return sum(n for total, n in counts.items() if total >= observed) / sum(counts.values())


def analyze(rows, samples):
    lookup = {(r['id'], r['ratio_pct']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate prompt and ratio pair')
    result = {}
    for task in TASKS:
        ids = [s['id'] for s in samples if s['task'] == task]
        n = len(next(s['answers'] for s in samples if s['task'] == task))
        if len(ids) != 16 or any(len(s['answers']) != n for s in samples if s['task'] == task):
            raise ValueError(f'Unexpected compound prompts in {task}')
        curve = {}
        for pct in GRID:
            group = [lookup[k, pct] for k in ids if (k, pct) in lookup]
            if len(group) != len(ids):
                continue
            answered = sum(r['answered_questions'] for r in group)
            point = dict(prompts=len(group), official_score=mean(r['score'] for r in group),
                         indexed_accuracy=mean(r['indexed_accuracy_diagnostic'] for r in group),
                         answered=answered, total=len(group)*n, completion_pct=100*answered/(len(group)*n),
                         severe_loops=sum(r['severe_loop'] for r in group),
                         long_repeats=sum(r['long_repeat'] for r in group))
            if all('decode_s' in r for r in group):
                seconds = sum(r['decode_s'] for r in group)
                point['decode_tokens_per_s'] = sum(r['decode_steps'] for r in group)/seconds if seconds else None
            curve[pct] = point
        task_result = result[task] = dict(curve=curve, grid_complete=len(curve)==len(GRID))
        if len(curve) != len(GRID):
            continue
        intervals = []
        for low, high in zip(GRID, GRID[1:]):
            deltas = [round((lookup[k,high]['score']-lookup[k,low]['score'])*n/100) for k in ids]
            intervals.append(dict(low_pct=low, high_pct=high, official_gain_pp=100*sum(deltas)/(len(ids)*n),
                                  improved=sum(d>0 for d in deltas), tied=deltas.count(0),
                                  declined=sum(d<0 for d in deltas), one_sided_p=flip_pvalue(deltas)))
        adjusted = 0.
        for rank, interval in enumerate(sorted(intervals, key=lambda x:x['one_sided_p'])):
            adjusted = min(1., max(adjusted, (len(intervals)-rank)*interval['one_sided_p']))
            interval['holm_p_within_task'] = adjusted
        top = max(it['official_gain_pp'] for it in intervals)
        task_result.update(
            intervals=intervals,
            largest_positive_intervals=[[it['low_pct'],it['high_pct']] for it in intervals
                                        if top > 0 and isclose(it['official_gain_pp'], top)],
            supported_gain_intervals=[[it['low_pct'],it['high_pct']] for it in intervals
                                      if it['official_gain_pp'] >= 5 and it['holm_p_within_task'] <= .05],
            lowest_sustained_near100_pct=next((pct for pct in GRID if all(
                curve[k]['official_score'] >= curve[100]['official_score']-3-1e-9 and curve[k]['severe_loops']==0
                for k in GRID if k >= pct)), None))
    return result
