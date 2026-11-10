"""Read immutable runs and summarize paired memory/speed. CPU only."""
from pathlib import Path
from statistics import median
import common as c


def summarize(rows):
    if any(r['status'] != 'ok' for r in rows):
        return dict(status='failed', statuses=[r['status'] for r in rows])
    fields = dict(tps='aggregate_tokens_per_s', sequence_tps='per_sequence_tokens_per_s',
                  ttft_s='ttft_s', final_kv_gib='final_kv_bytes')
    result = {name: median(r[field] for r in rows) / (2**30 if name.endswith('_gib') else 1)
              for name, field in fields.items()}
    result.update(status='ok', formal_runs=len(rows),
        e2e_tps=median(r['batch'] * r['output_tokens_per_row'] / (r['ttft_s'] + r['decode_s']) for r in rows),
        prefill_peak_gib=max(r['prefill_peak_bytes'] for r in rows) / 2**30,
        decode_peak_gib=max(r['decode_peak_bytes'] for r in rows) / 2**30,
        peak_gib=max(r['peak_allocated_bytes'] for r in rows) / 2**30,
        tps_min=min(r['aggregate_tokens_per_s'] for r in rows),
        tps_max=max(r['aggregate_tokens_per_s'] for r in rows))
    return result


def report(out):
    out = Path(out); plan = c.read(out / 'plan.json'); expected = c.expected_paths(plan)
    records = {name: c.read(out / name) for name in expected if (out / name).exists()}
    if any(r['experiment'] != plan['experiment'] for r in records.values()):
        raise ValueError('Result identity mismatch')
    complete = len(records) == len(expected) and all(r['status'] == 'ok' for r in records.values())
    checks = {}; speed = {}; quality = []
    for case in c.CASES:
        name = c.case_name(case); speed[name] = {}
        for arm in c.ARMS:
            rows = [records[p] for i in (1, 2, 3) if (p := f'speed/{name}/{arm}/r{i}.json') in records]
            if rows: speed[name][arm] = summarize(rows)
        for i in (1, 2, 3):
            paths = [f'speed/{name}/{arm}/r{i}.json' for arm in c.ARMS]
            if all(p in records and records[p]['status'] == 'ok' for p in paths):
                a, b = [records[p] for p in paths]
                checks[f'{name}/r{i}'] = a['output_sha256'] == b['output_sha256']
        if all(speed[name].get(arm, {}).get('formal_runs') == 3 for arm in c.ARMS):
            a, b = [speed[name][arm] for arm in c.ARMS]
            speed[name]['optimized_change_percent'] = {
                k: (b[k] / a[k] - 1) * 100 for k in ('tps', 'e2e_tps', 'ttft_s', 'peak_gib', 'final_kv_gib')}
    for sample in plan['quality']:
        pair = {arm: records.get(f'quality/{c.digest(sample["id"])}/{arm}.json') for arm in c.ARMS}
        if all(r and r['status'] == 'ok' for r in pair.values()):
            checks[sample['id']] = pair['baseline']['output_sha256'] == pair['optimized']['output_sha256']
        quality.append(dict(sample=sample, results=pair))
    complete = complete and all(checks.values())
    state = c.read(out / 'state.json') if (out / 'state.json').exists() else {}
    result = dict(experiment=plan['experiment'], complete=complete, completed_records=len(records),
                  expected_records=len(expected), speed=speed, quality=quality, output_matches=checks, state=state)
    lines = ['BetterMemory 同卡 A/B 快速结果', f'COMPLETE={complete}; records={len(records)}/{len(expected)}',
             f'当前状态: {state.get("phase", "prepared")} / {state.get("current")}',
             '参数：Prompt90%，冷块目标80%，INT4/INT4，recent128，obs32。',
             '速度为3正式轮中位数；峰值为3轮最大PyTorch allocated；预热和门禁不计入。',
             '质量仅2条回归样本，完整输入/原输出上限/自然EOS，不代表全量质量。']
    for name, arms in speed.items():
        lines.append(name)
        for arm in c.ARMS:
            value = arms.get(arm)
            if not value: lines.append(f'  {arm}: pending'); continue
            if value['status'] != 'ok': lines.append(f'  {arm}: FAILED {value}'); continue
            lines.append(f'  {arm}: n={value["formal_runs"]}/3 decode={value["tps"]:.3f} total tok/s; '
                         f'e2e={value["e2e_tps"]:.3f}; TTFT={value["ttft_s"]:.3f}s; '
                         f'KV={value["final_kv_gib"]:.3f} GiB; prefill/decode/total peak='
                         f'{value["prefill_peak_gib"]:.3f}/{value["decode_peak_gib"]:.3f}/{value["peak_gib"]:.3f} GiB')
        if 'optimized_change_percent' in arms: lines.append('  变化%: ' + str(arms['optimized_change_percent']))
    lines.append('输出逐token一致: ' + str(checks))
    c.write(out / 'summary.json', result)
    import os, time
    temp = out / f'summary.txt.{os.getpid()}.{time.time_ns()}.tmp'
    temp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temp.replace(out / 'summary.txt')
    return result
