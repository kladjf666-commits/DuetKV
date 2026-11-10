"""Publish validated SCOPE reruns into the previous report with provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def read_records(prefix):
    return [json.loads(l) for l in prefix.with_suffix('.predictions.jsonl').read_text().splitlines() if l]


def main():
    old_prefix = ROOT / 'results/qwen2_1p5b_longbench_real_kv_20261023'
    old = json.loads(old_prefix.with_suffix('.json').read_text())
    if 'scope_rerun_update' in old['meta']:
        raise RuntimeError('Report already updated; do not overwrite the provenance archive')
    previous_records = read_records(old_prefix)
    full = {r['sample_id']: r for r in previous_records if r['method'] == 'full'}
    old_scope = {r['sample_id']: r for r in previous_records if r['method'] == 'scope'}
    tasks = old['meta']['tasks']
    runs = {}
    for budget in (256, 32):
        prefix = ROOT / f'results/qwen2_1p5b_scope_b{budget}_20261024'
        report = json.loads(prefix.with_suffix('.json').read_text())
        records = read_records(prefix)
        assert report['meta']['complete'] and len(records) == 160
        assert len({r['sample_id'] for r in records}) == 160
        assert {r['sample_id'] for r in records} == set(full)
        assert report['meta']['data_hashes'] == old['meta']['data_hashes']
        assert report['meta']['model'] == old['meta']['model']
        for r in records:
            ref = full[r['sample_id']]
            assert r['input_hash'] == ref['input_hash'] and r['answers'] == ref['answers']
            assert r['scope_decode_budget'] == budget
            assert r['protocol_hash'] == report['meta']['protocol_hash']
            trace = r['compression_trace']
            assert trace['implementation']['implementation']['revision'] == 'scope_slide_current_query_v2'
            assert trace['summary']['decode']['checks'] == len(r['generated_ids']) - 1
            assert trace['final_state']['physical_bytes'] == r['physical_kv_bytes']
            assert all(l['tokens'] <= r['prompt_tokens'] + budget for l in trace['final_state']['layers'])
            if budget == 256:
                assert trace['status'] == 'not_triggered'
                assert r['generated_ids'] == ref['generated_ids'] == old_scope[r['sample_id']]['generated_ids']
        total_ms = sum(r['wall_ms'] for r in records)
        outputs = [len(r['generated_ids']) for r in records]
        triggered = [r for r in records if r['compression_trace']['status'] == 'triggered']
        deltas = [np.array([r['qa_f1'] - full[r['sample_id']]['qa_f1'] for r in records if r['task'] == t]) for t in tasks]
        rng = np.random.default_rng(20261024)
        boot = sum(a[rng.integers(0, len(a), size=(10000, len(a)))].mean(1) for a in deltas) / len(tasks) * 100
        runs[budget] = {'prefix': prefix, 'report': report, 'records': records,
                        'summary': {'decode_budget': budget, 'samples': len(records),
                            'triggered_samples': len(triggered),
                            'triggered_sample_ids': [r['sample_id'] for r in triggered],
                            'eviction_steps': sum(r['compression_trace']['summary']['decode']['triggered_steps'] for r in records),
                            'evicted_layer_tokens': sum(r['compression_trace']['summary']['decode']['layer_token_totals']['evicted_tokens'] for r in records),
                            'mean_output_tokens_including_eos': statistics.mean(outputs),
                            'max_output_tokens_including_eos': max(outputs),
                            'end_to_end_output_tokens_per_second': sum(outputs) / (total_ms/1000),
                            'exact_output_matches_vs_full': sum(r['generated_ids'] == full[r['sample_id']]['generated_ids'] for r in records),
                            'qa_f1_delta_vs_full': statistics.mean(statistics.mean(a) for a in deltas) * 100,
                            'paired_bootstrap_95pct_ci': np.quantile(boot, [0.025, 0.975]).tolist()}}
    archive = ROOT / 'results/archive/before_scope_rerun_20261024'
    archive.mkdir(parents=True, exist_ok=False)
    for suffix in ('.json', '.md', '.predictions.jsonl', '.statistics.json'):
        src = old_prefix.with_suffix(suffix)
        shutil.copy2(src, archive / src.name)
    original_hash = old['meta'].pop('protocol_hash')
    old['meta']['original_protocol_hash'] = original_hash
    old['meta']['report_mode'] = 'composite_runs_with_per_method_provenance'
    old['meta']['method_protocol_hashes'] = {r['method']: original_hash for r in old['rows']}
    old['meta']['method_protocol_hashes']['scope'] = runs[256]['report']['meta']['protocol_hash']
    update = {'date': '2026-10-24', 'implementation': 'scope_slide_current_query_v2',
              'replacement_run': str(runs[256]['prefix'].relative_to(ROOT)),
              'supplementary_run': str(runs[32]['prefix'].relative_to(ROOT)),
              'previous_report_archive': str(archive.relative_to(ROOT)),
              'other_methods_rerun': False,
              'identical_inputs_answers_and_dataset_hashes_verified': True,
              'timing_note': 'New SCOPE includes trace collection; other timings are retained from 2026-10-23. Not an isolated decode-speed comparison.',
              'runs': [runs[b]['summary'] for b in (256,32)]}
    old['meta']['scope_rerun_update'] = update
    replacement = runs[256]['report']['rows'][0]
    old['rows'] = [replacement if r['method'] == 'scope' else r for r in old['rows']]
    # Keep optional changed-budget results separate from the original six-method table.
    old['supplementary_scope_runs'] = [{'row': runs[32]['report']['rows'][0], **runs[32]['summary']}]
    records_by_id = {r['sample_id']: r for r in runs[256]['records']}
    merged = [records_by_id[r['sample_id']] if r['method'] == 'scope' else r for r in previous_records]
    assert len(merged) == 960
    for before, after in zip(previous_records, merged):
        if before['method'] != 'scope':
            assert before == after
    old_prefix.with_suffix('.predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in merged))
    old_prefix.with_suffix('.json').write_text(json.dumps(old, ensure_ascii=False, indent=2)+'\n')
    stat_path = old_prefix.with_suffix('.statistics.json')
    stats = json.loads(stat_path.read_text())
    stats['scope_rerun_update'] = {'date': '2026-10-24', 'budget': 256,
                                  'paired_scores_unchanged_verified': True,
                                  'protocol_hash': runs[256]['report']['meta']['protocol_hash']}
    stat_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2)+'\n')
    md_path = old_prefix.with_suffix('.md')
    md = md_path.read_text()
    r = replacement
    new_row = (f"| scope | 160 | {r['qa_f1']:.2f} | {r['logical_kv_saving_pct']:.2f} | "
               f"{r['physical_kv_saving_pct']:.2f} | {r['mean_physical_kv_mib']:.2f} | "
               f"{r['peak_allocated_mib']:.1f} | {r['mean_wall_ms']:.1f} | {r['token_match_vs_full']:.4f} |")
    md = '\n'.join(new_row if line.startswith('| scope | 160 |') else line for line in md.splitlines()) + '\n'
    first, _, rest = md.partition('\n')
    md = first + '\n\n> 2026-10-24 已更新：主表 SCOPE 替换为修正后的 ALLKV + Slide（GQA适配）同参复测，其他五种方法保留2026-10-23结果。另附32-token生成预算敏感性测试。\n' + rest
    md = md.replace('SCOPE 的重要度等辅助状态', '算法辅助状态')
    lines = ['','## SCOPE 修正后的复测（2026-10-24）','',
             '两组均使用相同的5任务×32条输入，逐条核验输入token、答案和数据文件哈希。正文截断1536，添加模板后1211–1555 tokens，平均1552.81；BF16、贪心、原任务输出上限、正常遇EOS停止。',
             '评分已由历史累计注意力改为当前query注意力；当前支持ALLKV + Slide的GQA组内聚合适配，未实现Adaptive、Discontinuous或压缩prompt插件。参考 [SCOPE核对报告](../SCOPE/AUDIT.md)。','',
             '| 生成段预算 / 近期窗口 | QA F1 | 实际KV节省% | 平均KV MiB | 触发样本 | 淘汰步数 | 输出与full完全相同 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for b in (256,32):
        r=runs[b]['report']['rows'][0]; s=runs[b]['summary']
        lines.append(f"| {b} / 16 | {r['qa_f1']:.4f} | {r['physical_kv_saving_pct']:.4f} | {r['mean_physical_kv_mib']:.4f} | {s['triggered_samples']}/160 | {s['eviction_steps']} | {s['exact_output_matches_vs_full']}/160 |")
    s=runs[32]['summary']
    lines.extend(['',
        '256预算组仍未触发淘汰，160条输出与旧full和旧SCOPE完全一致；这不是有效压缩效果的证据。',
        f"32预算组有{s['triggered_samples']}条真正发生淘汰；样本：{', '.join(s['triggered_sample_ids']) or '无'}。由于完整保留prompt且大部分回答很短，总KV节省很有限，不能据此评价长输出场景。该组参数不同，单独展示，不替换主表。",
        f"32预算组相对full的F1差值为{s['qa_f1_delta_vs_full']:+.4f}，按任务配对bootstrap的95%区间为[{s['paired_bootstrap_95pct_ci'][0]:+.4f}, {s['paired_bootstrap_95pct_ci'][1]:+.4f}]；仅用于当前样本的描述。",'',
        '### 速度口径','',
        '| 生成段预算 | 平均输出tokens（含EOS） | 端到端输出tokens/s | 平均每题ms |',
        '|---|---:|---:|---:|'])
    for b in (256,32):
        r=runs[b]['report']['rows'][0]; s=runs[b]['summary']
        lines.append(f"| {b} | {s['mean_output_tokens_including_eos']:.2f} | {s['end_to_end_output_tokens_per_second']:.2f} | {r['mean_wall_ms']:.2f} |")
    lines.extend(['','以上速度包含prefill、缓存维护和触发日志采集，不是纯decode速度；其他方法的旧耗时没有同样的触发日志开销，也不是本轮同时重测，不宜据此排名速度。','',
        '### 原始记录与追溯','',
        '- [同参256预算逐题记录](qwen2_1p5b_scope_b256_20261024.predictions.jsonl)',
        '- [补充32预算逐题记录](qwen2_1p5b_scope_b32_20261024.predictions.jsonl)',
        '- [复测验证和汇总](scope_rerun_validation_20261024.json)',
        '- [更新前报告备份](archive/before_scope_rerun_20261024/qwen2_1p5b_longbench_real_kv_20261023.md)',
        '', '主报告JSON及逐题文件同步替换SCOPE行，保留各条记录真实的protocol_hash，并在meta中列出各方法所属运行。旧bootstrap区间经逐题核验未改变；其他方法没有补造触发记录。',''])
    md_path.write_text(md + '\n'.join(lines))
    validation = {**update, 'main_report_records': len(merged), 'unchanged_other_method_records': 800,
                  'main_report_sha256': hashlib.sha256(old_prefix.with_suffix('.json').read_bytes()).hexdigest()}
    (ROOT / 'results/scope_rerun_validation_20261024.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(update['runs'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
