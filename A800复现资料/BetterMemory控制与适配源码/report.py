"""CPU summary and aligned historical comparison, with partial/failure status."""
from pathlib import Path
from statistics import median
import common as c


def summarize(rows):
    if any(r['status'] != 'ok' for r in rows):
        return dict(status='failed', statuses=[r['status'] for r in rows])
    fields = dict(tps='aggregate_tokens_per_s', sequence_tps='per_sequence_tokens_per_s',
                  ttft='ttft_s', kv_gib='final_kv_bytes')
    result = {name: median(r[field] for r in rows) / (2**30 if name.endswith('_gib') else 1)
              for name, field in fields.items()}
    result.update(status='ok' if len(rows) == 3 else 'partial', formal_runs=len(rows),
        e2e_tps=median(r['batch'] * r['output_tokens_per_row'] / (r['ttft_s'] + r['decode_s']) for r in rows),
        prefill_peak_gib=max(r['prefill_peak_bytes'] for r in rows) / 2**30,
        decode_peak_gib=max(r['decode_peak_bytes'] for r in rows) / 2**30,
        peak_gib=max(r['peak_allocated_bytes'] for r in rows) / 2**30,
        tps_min=min(r['aggregate_tokens_per_s'] for r in rows),
        tps_max=max(r['aggregate_tokens_per_s'] for r in rows),
        gpu_uuid=rows[0].get('gpu_uuid'))
    return result


def report(out):
    out = Path(out); plan = c.read(out / 'plan.json'); expected = c.expected_paths(plan)
    records = {name: c.read(out / name) for name in expected if (out / name).exists()}
    if any(r['experiment'] != plan['experiment'] for r in records.values()):
        raise ValueError('Result identity mismatch')
    complete = len(records) == len(expected) and all(r['status'] == 'ok' for r in records.values())
    speed = {}; checks = {}
    for case in c.CASES:
        name = c.case_name(case)
        rows = [records[p] for i in (1, 2, 3) if (p := f'speed/{name}/optimized/r{i}.json') in records]
        speed[name] = summarize(rows) if rows else dict(status='pending')
        successful = [r for r in rows if r['status'] == 'ok']
        if successful:
            checks[name] = len({r['output_sha256'] for r in successful}) == 1
            for row in successful:
                if (row['batch'], row['input_tokens'], row['output_tokens_per_row']) != (case[2], case[0], case[1]):
                    raise ValueError('Result shape mismatch: ' + name)
                if len(row['output_ids']) != case[2] or any(len(x) != case[1] for x in row['output_ids']):
                    raise ValueError('Saved output shape mismatch: ' + name)
                if c.digest(row['output_ids']) != row['output_sha256']:
                    raise ValueError('Saved output hash mismatch: ' + name)
        # A failed repeatability check must not appear as a valid performance result.
        if checks.get(name) is False: speed[name]['status'] = 'output_mismatch'
    complete = complete and all(checks.values())
    state = c.read(out / 'state.json') if (out / 'state.json').exists() else {}
    result = dict(experiment=plan['experiment'], complete=complete, completed_records=len(records),
        expected_records=len(expected), speed=speed, output_matches=checks, state=state)
    history = c.read(c.ROOT / 'data/historical_speed.json')
    comparison = dict(experiment=plan['experiment'], complete=complete,
        caveat=history['caveat'], historical_source_sha256=history['source_sha256'],
        speed={**history['speed'], 'phasekv90_bettermemory': speed})
    c.write(out / 'comparison.json', comparison)
    c.write(out / 'summary.json', result)
    lines = ['BetterMemory 最终16点峰值显存与速度',
        f'COMPLETE={complete}; records={len(records)}/{len(expected)}（1门禁+48正式轮）',
        f'当前状态: {state.get("phase", "prepared")} / {state.get("current")}',
        '仅优化版：Prompt90%，冷块目标80%，INT4/INT4，recent128，obs32。',
        '速度/KV取3轮中位数，峰值取最大PyTorch allocated；partial不是最终成绩。',
        '测速强制输出，不提供任务质量分。文本/EOS/重复诊断均在计时结束后保存。']
    for name, value in speed.items():
        if value['status'] not in ('ok', 'partial'):
            lines.append(f'{name}: {value["status"]}'); continue
        lines.append(f'{name}: {value["status"]} n={value["formal_runs"]}/3; '
            f'decode={value["tps"]:.4f} total tok/s; per-request={value["sequence_tps"]:.4f}; '
            f'e2e={value["e2e_tps"]:.4f}; TTFT={value["ttft"]:.4f}s; KV={value["kv_gib"]:.4f} GiB; '
            f'prefill/decode/total peak={value["prefill_peak_gib"]:.4f}/{value["decode_peak_gib"]:.4f}/{value["peak_gib"]:.4f} GiB')
    lines += ['', '与历史算法逐点对齐（非同卡受控加速比；OOM/未测保留原状态）',
              'scene | method | status | decode total | per-request | e2e tok/s | TTFT s | final KV GiB | prefill/decode/total peak GiB']
    for case in c.CASES:
        name = c.case_name(case)
        for method, points in comparison['speed'].items():
            r = points[name]
            if r['status'] != 'ok': lines.append(f'{name} | {method} | {r["status"]}'); continue
            lines.append(f'{name} | {method} | ok | {r["tps"]:.4f} | {r["sequence_tps"]:.4f} | '
                f'{r["e2e_tps"]:.4f} | {r["ttft"]:.4f} | {r["kv_gib"]:.4f} | '
                f'{r["prefill_peak_gib"]:.4f}/{r["decode_peak_gib"]:.4f}/{r["peak_gib"]:.4f}')
    import os, time
    temp = out / f'summary.txt.{os.getpid()}.{time.time_ns()}.tmp'
    temp.write_text('\n'.join(lines) + '\n', encoding='utf-8'); temp.replace(out / 'summary.txt')
    return result
