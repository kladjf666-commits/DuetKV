"""Audit and compare downloaded runs without modifying their original results.

Run from any directory with Python 3.10+ (standard library only).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re
import statistics as stats
import string

METHODS = ['full', 'kivi', 'snapkv', 'scope', 'phasekv']
LABELS = {'full': 'Full', 'kivi': 'KIVI', 'snapkv': 'SnapKV', 'scope': 'SCOPE',
          'phasekv': 'PhaseKV'}
TASKS = ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa',
         'gov_report', 'qmsum', 'multi_news']
MODEL = 'Qwen2.5-7B-Instruct'
GIB, MIB = 2**30, 2**20


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path, manifest):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len({r['case_id'] for r in rows}) == len(rows), f'Duplicate case IDs: {path}'
    assert all(r['protocol_hash'] == manifest['protocol_hash'] for r in rows), f'Mixed protocols: {path}'
    assert all(r['model'] == MODEL for r in rows)
    for r in rows:
        if r['status'] != 'ok':
            continue
        for field in ('decode_s', 'total_s', 'prefill_and_compress_s'):
            assert math.isfinite(r[field]) and r[field] >= 0
        assert math.isclose(r['total_s'], r['decode_s']+r['prefill_and_compress_s'], rel_tol=1e-9)
        if r['output_tokens'] > 1:
            assert math.isclose(r['decode_tokens_per_s'], (r['output_tokens']-1)/r['decode_s'], rel_tol=1e-9)
        if r['phase'] in ('quality','needle'):
            assert math.isfinite(r['score']) and 0 <= r['score'] <= 100
        if r['phase'] == 'quality' and r['metric'] == 'qa_f1':
            def words(text):
                text = text.lower().translate(str.maketrans('', '', string.punctuation))
                return re.sub(r'\b(a|an|the)\b', ' ', text).split()
            pred = words(r['prediction'])
            scores = []
            for answer in r['answers']:
                gold = words(answer)
                overlap = sum((Counter(pred) & Counter(gold)).values())
                scores.append(200*overlap/(len(pred)+len(gold)) if overlap else 0)
            assert math.isclose(r['score'], max(scores), abs_tol=1e-9), 'QA score mismatch'
        if r['phase'] == 'needle':
            match = re.search(r'(?<!\d)'+re.escape(r['answers'][0])+r'(?!\d)', r['prediction'])
            assert r['score'] == (100 if match else 0), 'Needle score mismatch'
    return {r['case_id']: r for r in rows}


def summarize_performance(rows):
    grouped = defaultdict(list)
    for r in rows.values():
        if r['phase'] == 'performance' and r['status'] == 'ok' and not r['is_warmup']:
            assert r['fixed_output'] and r['output_tokens'] == r['requested_output_tokens']
            grouped[(r['sweep'], r['input_tokens'], r['output_tokens'])].append(r)
    result = []
    for (sweep, inputs, outputs), records in sorted(grouped.items()):
        result.append(dict(sweep=sweep, input_tokens=inputs, output_tokens=outputs,
                           repeats=[r['repeat'] for r in records], n=len(records),
                           decode_tps=stats.median(r['decode_tokens_per_s'] for r in records),
                           decode_tps_min=min(r['decode_tokens_per_s'] for r in records),
                           decode_tps_max=max(r['decode_tokens_per_s'] for r in records),
                           total_s=stats.median(r['total_s'] for r in records),
                           prefill_s=stats.median(r['prefill_and_compress_s'] for r in records),
                           peak_gib=max(r['peak_allocated_bytes'] for r in records)/GIB,
                           final_kv_mib=stats.median(r['final_kv_bytes'] for r in records)/MIB))
    return result


def clustered_ci(rows_a, rows_b, tasks, draws=3000):
    """Paired, task-stratified bootstrap; resample documents, retain all 4 lengths."""
    grouped = defaultdict(lambda: defaultdict(list))
    for case, a in rows_a.items():
        if a.get('task') not in tasks or a['phase'] != 'quality':
            continue
        b = rows_b[case]
        assert a['dataset_id'] == b['dataset_id'] and a['input_hash'] == b['input_hash']
        grouped[a['task']][a['dataset_id']].append(a['score'] - b['score'])
    clusters = []
    for task in tasks:
        assert len(grouped[task]) == 32
        assert all(len(v) == 4 for v in grouped[task].values())
        clusters.append([stats.mean(v) for _, v in sorted(grouped[task].items())])
    rng = random.Random(20261025)
    boot = sorted(sum(sum(rng.choices(values, k=len(values))) / len(values) for values in clusters)
                  / len(clusters) for _ in range(draws))
    return {'delta_points': stats.mean(stats.mean(v) for v in clusters),
            'ci95': [boot[int(.025*draws)], boot[int(.975*draws)]], 'draws': draws,
            'documents': sum(map(len, clusters)), 'rows': sum(map(len, clusters))*4}


def cell(value):
    return html.escape(str(value))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+cell(h)+'</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>'+cell(v)+'</td>' for v in row) + '</tr>'
                                        for row in rows) + '</tbody></table></div>'


def fmt(v, digits=2):
    return f'{v:,.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]/'reports'/'4090d_comparison_20261025')
    args = parser.parse_args()
    old, new = args.results/'4090d_qwen_v3', args.results/'4090d_qwen_phasekv_segmented'
    ma, mb = read_json(old/'manifest.json'), read_json(new/'manifest.json')
    for manifest in (ma, mb):
        without_hash = {k: v for k, v in manifest.items() if k != 'protocol_hash'}
        encoded = json.dumps(without_hash, sort_keys=True, ensure_ascii=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == manifest['protocol_hash'], 'Manifest hash mismatch'
    assert ma['protocol'] == mb['protocol'] and ma['data'] == mb['data'] and ma['checkpoints'] == mb['checkpoints']
    setting_diffs = {k: [ma['settings'].get(k), mb['settings'].get(k)]
                     for k in set(ma['settings'])|set(mb['settings']) if ma['settings'].get(k) != mb['settings'].get(k)}
    assert set(setting_diffs) == {'methods', 'out'}
    pa, pb = ma['profiles'][0], mb['profiles'][0]
    assert {k:v for k,v in pa.items() if k != 'methods'} == {k:v for k,v in pb.items() if k != 'methods'}
    assert next(s for s in pa['methods'] if s['method']=='phasekv') == pb['methods'][0]
    changed_sources = [k for k in sorted(set(ma['source_hashes'])|set(mb['source_hashes']))
                       if ma['source_hashes'].get(k) != mb['source_hashes'].get(k)]
    assert set(changed_sources) == {'PhaseKV/phasekv_cache.py', 'PhaseKV/kernels/attention_triton.py',
                                    'PhaseKV/kernels/segmented_attention.py', 'PhaseKV/kernels/segmented_plan.py'}
    state_old, state_new = read_json(old/'run_state.json'), read_json(new/'run_state.json')
    assert not state_old['failed'] and not state_new['failed'] and state_new['status']=='complete'
    assert {MODEL+'__'+m for m in METHODS if m!='phasekv'} <= set(state_old['finished'])
    validations = []
    for run in (old, new):
        for path in sorted(run.glob('gpu_validation_*.json')):
            validation = read_json(path)
            assert validation['status'] == 'passed'
            validations.append({'file': str(path.resolve()), 'status': 'passed',
                                'checks': len(validation['checks']), 'sha256': sha(path)})

    data, environments, sources, coverage = {}, {}, [], []
    for method in METHODS:
        run = old if method != 'phasekv' else new
        raw_method = method
        path = run/'records'/f'{MODEL}__{raw_method}.jsonl'
        manifest = mb if run == new else ma
        rows = read_records(path, manifest)
        data[method] = rows
        assert all(r['method'] == raw_method for r in rows.values())
        spec = next(s for s in manifest['profiles'][0]['methods'] if s['method'] == raw_method)
        assert all(r['parameters'] == spec for r in rows.values())
        assert len(rows) == 1179 and Counter(r['phase'] for r in rows.values()) == {
            'capacity':6, 'needle':81, 'performance':68, 'quality':1024}
        assert all(r['status'] == 'ok' for r in rows.values())
        sources.append({'file': str(path.resolve()), 'sha256': sha(path), 'records': len(rows)})
        coverage.append({'method': method, 'records': len(rows),
                         'phases': dict(Counter(r['phase'] for r in rows.values())),
                         'statuses': dict(Counter(r['status'] for r in rows.values())),
                         'capacity': read_json(path.with_suffix('.capacity.json'))})
        env_paths = list((run/'records').glob(f'{MODEL}__{raw_method}.environment.*.json'))
        assert len(env_paths) == 1
        environments[method] = read_json(env_paths[0])
        sources.append({'file': str(env_paths[0].resolve()), 'sha256': sha(env_paths[0])})
    assert all(e == environments['full'] for e in environments.values()), 'Runtime environments differ'
    base = data['full']
    for method, rows in data.items():
        assert set(rows) == set(base)
        for key, r in rows.items():
            for field in ('input_hash','input_tokens','phase','requested_output_tokens','task','sample_index','dataset_id'):
                assert r.get(field) == base[key].get(field), f'{method} {key}: {field} mismatch'
    quality, task_scores, length_scores, needle = [], [], [], []
    for method in METHODS:
        rows = list(data[method].values())
        q = [r for r in rows if r['phase']=='quality']
        needle_rows = [r for r in rows if r['phase']=='needle']
        assert all(r['score'] in (0,100) for r in needle_rows)
        quality.append({'method': method, 'n':len(q),
                        'qa_f1': stats.mean(r['score'] for r in q if r['metric']=='qa_f1'),
                        'rouge_l': stats.mean(r['score'] for r in q if r['metric']=='rouge_l'),
                        'macro8': stats.mean(r['score'] for r in q),
                        'needle_correct':sum(r['score']==100 for r in needle_rows), 'needle_n':len(needle_rows),
                        'needle_pct':stats.mean(r['score'] for r in needle_rows)})
        for task in TASKS:
            selected = [r for r in q if r['task']==task]
            assert len(selected) == 128
            task_scores.append({'method':method, 'task':task,'n':len(selected), 'score':stats.mean(r['score'] for r in selected)})
        for length in ma['settings']['quality_lengths']:
            selected = [r for r in q if r['length_budget']==length]
            length_scores.append({'method':method,'budget':length,'n':len(selected),'macro8':stats.mean(r['score'] for r in selected)})
        for length in sorted({r['input_tokens'] for r in needle_rows}):
            selected = [r for r in needle_rows if r['input_tokens']==length]
            needle.append({'method':method,'input_tokens':length,'n':len(selected),
                           'correct':sum(r['score']==100 for r in selected)})
    pair_stats = {method: {label: clustered_ci(data['phasekv'], data[method], tasks)
                           for label,tasks in [('qa',TASKS[:5]),('summary',TASKS[5:]),('macro8',TASKS)]}
                  for method in METHODS if method != 'phasekv'}
    perf = {m:summarize_performance(rows) for m,rows in data.items()}
    for m in METHODS:
        assert len(perf[m]) == 17 and all(p['n']==3 and sorted(p['repeats'])==[1,2,3] for p in perf[m])
    def find(m, sweep, inputs, outputs):
        return next(p for p in perf[m] if (p['sweep'],p['input_tokens'],p['output_tokens'])==(sweep,inputs,outputs))
    audit = {'comparison_methods':METHODS, 'settings_match_except_selection_and_output':True,'source_differences_between_runs':changed_sources,
             'checkpoint_and_data_equal':True,'all_environments_equal':True,'input_hash_mismatches':0,
             'duplicate_case_ids':0, 'completed_methods':METHODS,
             'validations':validations, 'source_files':sources,
             'manifests':[{'file':str((p/'manifest.json').resolve()),'sha256':sha(p/'manifest.json')} for p in (old,new)],
             'phasekv_parameters':pb['methods'][0], 'environment':environments['phasekv'],
             'score_audit': 'QA F1 and needle scores recomputed from predictions/answers; ROUGE-L aggregated as recorded',
             'timing_audit': 'Finite nonnegative timings, total=prefill+decode, throughput=(outputs-1)/decode verified'}
    result = {'model':MODEL,'audit':audit,'coverage':coverage,'quality':quality,'quality_by_task':task_scores,
              'quality_by_budget':length_scores,'quality_paired_cluster_bootstrap':pair_stats,
              'needle_by_length':needle,'performance':perf}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

    sections = []
    sections.append('<h2>主要结论</h2><p>本报告比较 <b>PhaseKV、Full、KIVI、SnapKV 和 SCOPE</b> 的质量、解码速度与显存使用。'
                    'PhaseKV 的优势集中在长上下文解码和最终 KV 缓存大小：输入 32000、输出 256 时，'
                    '<b>解码速度为 41.54 token/s，最终 KV 为 104.97 MiB</b>，在五种方法中分别最快和最小。'
                    '短输入长输出时，PhaseKV 慢于 Full 和 SnapKV；质量均值低于 Full，摘要任务的差距较明确。'
                    '以下结果均来自 Qwen2.5-7B-Instruct、单张 RTX 4090 D、BF16 权重和 batch=1，适用于本次具体实现与参数配置。</p>')
    sections.append('<div class="note"><b>显存口径：</b>KV 物理字节只统计缓存内容；总 allocated 峰值还含模型权重、注意力分数和工作缓冲区。'
                    '“KV 更小”不能等同于“总显存峰值更低”。所有方法均达到输入 32640 + 输出 128 的模型/设置上限，没有测到 OOM 边界，'
                    '因此不能据此比较谁支持更大的最大上下文，也未实现 24GB 满显存测试。</div>')
    sections.append('<h2>测试设置与数据完整性</h2><p>五种方法各完成 1179 条测试，全部成功，无重复、无 OOM、无跳过。'
                    '本报告由原始 JSONL 重新统计，共形成 1179 组五方法完整配对。'
                    '模型权重、tokenizer、数据文件和 prompt 模板的哈希一致；输入长度、输入哈希及抽样案例逐项匹配。'
                    '各运行记录的测试设置除方法选择与输出目录外一致，计时环境相同，所采用的方法均已完成并通过对应的 GPU 数值校验。</p>')
    sections.append(table(['实现','容量','检索','性能（含预热）','质量','总记录'],
                          [[LABELS[c['method']]]+[c['phases'].get(p,0) for p in ('capacity','needle','performance','quality')]+[c['records']] for c in coverage]))
    sections.append('<p>每种完整实现：容量 6 条；检索 9 种长度 × 3 个深度 × 3 个样本 = 81 条；性能 17 种形状 ×（1 次预热 + 3 次测量）= 68 条；'
                    '质量 8 任务 × 32 文档 × 4 输入预算 = 1024 条（256 个不同文档）。性能表剔除预热，使用 3 次测量中位数；峰值显存取这些测量中的最大值。'
                    '性能测试强制输出长度，质量测试允许 EOS；decode token/s 不包含 prefill 和首个输出 token，整条耗时包含 prefill、压缩和解码。'
                    'QA F1 与检索分数已由预测和答案重新核算；ROUGE-L 按原始记录汇总。计时加和与 token/s 公式已核对。</p>'
                    '<p>实际执行脚本使用 SDPA 和 512-token 分块 prefill；需要选择分数的方法在末尾32-token观察窗口切换 eager。'
                    '各运行采用相同执行脚本，源码哈希一致。输入预算包含聊天模板，质量样本按预算截断。</p>')
    sections.append('<h2>质量与检索</h2><p>分数范围 0–100。QA 为五个问答任务的 QA F1 均值；摘要为三个摘要任务的 ROUGE-L 均值。'
                    '八任务综合均值是本报告的描述性指标，混合两种指标，不是官方 LongBench 总分。输入会截断并加聊天模板；每任务仅采样 32 个文档。'
                    'custom_passkey 为脚本自建八位数字检索，不等同于完整 RULER。</p>')
    sections.append(table(['方法','QA F1（640条）','摘要 ROUGE-L（384条）','八任务均值','检索成功/81','检索成功率'],
                          [[LABELS[q['method']],fmt(q['qa_f1']),fmt(q['rouge_l']),fmt(q['macro8']),q['needle_correct'],fmt(q['needle_pct'])+'%'] for q in quality]))
    sections.append(table(['任务 / 指标']+[LABELS[m] for m in METHODS],
                          [[task+(' / F1' if task in TASKS[:5] else ' / ROUGE-L')]+[fmt(next(r['score'] for r in task_scores if r['task']==task and r['method']==m)) for m in METHODS] for task in TASKS]))
    sections.append('<p>PhaseKV 在 narrativeqa 上均值最高，qmsum 与 Full 接近；在 gov_report、multi_news、qasper、multifieldqa_en 和 hotpotqa 上低于 Full。'
                    '综合而言，PhaseKV 的质量收益随任务变化，需要与速度及缓存大小一起评估。</p>')
    sections.append(table(['输入预算 / 八任务均值']+[LABELS[m] for m in METHODS],
                          [[n]+[fmt(next(r['macro8'] for r in length_scores if r['budget']==n and r['method']==m)) for m in METHODS] for n in ma['settings']['quality_lengths']]))
    sections.append('<h3>PhaseKV 相对各基线的配对质量差</h3><p>单位为分；负数表示 PhaseKV 更低。95% 区间使用固定任务分层、文档为簇的配对 bootstrap（3000 次）；'
                    '同一文档四个输入预算一起抽样，避免把 1024 次测量误当作 1024 个独立文档。区间包含 0 时，本抽样无法明确区分均值高低。'
                    '区间不覆盖任务选择、配置选择及多重比较的不确定性。</p>')
    sections.append(table(['对照方法','QA 差 [95%区间]','摘要差 [95%区间]','八任务差 [95%区间]'],
                          [[LABELS[m]]+[f"{v['delta_points']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for v in (pair_stats[m]['qa'],pair_stats[m]['summary'],pair_stats[m]['macro8'])] for m in METHODS if m!='phasekv']))
    sections.append('<h3>检索按输入长度统计</h3>')
    sections.append(table(['输入 token / 每格成功数（共9次）']+[LABELS[m] for m in METHODS],
                          [[n]+[str(next(r['correct'] for r in needle if r['input_tokens']==n and r['method']==m))+'/9' for m in METHODS] for n in sorted({r['input_tokens'] for r in needle})]))

    sections.append('<h2>长输出性能：固定输入 1024</h2>')
    sections.append(table(['输出 token / 解码 token/s']+[LABELS[m] for m in METHODS],
                          [[n]+[fmt(find(m,'output_sweep',1024,n)['decode_tps']) for m in METHODS] for n in ma['settings']['outputs']]))
    sections.append('<p>输出从 256 增至 4096 时，PhaseKV 解码速度由 50.09 降至 40.66 token/s。输出 4096 时，'
                    'Full 为 48.93、SnapKV 为 50.24、KIVI 为 33.66、SCOPE 为 39.06 token/s。'
                    'PhaseKV 在这一场景中处于中间水平；各方法在长上下文场景中的相对速度见下表。</p>')
    sections.append('<h2>长上下文：速度与显存</h2><h3>输入 32000 + 输出 256</h3>')
    full_long=find('full','context_sweep',32000,256)
    long_rows=[]
    for m in METHODS:
        r=find(m,'context_sweep',32000,256)
        long_rows.append([LABELS[m],fmt(r['decode_tps']),fmt(r['total_s']),fmt(r['final_kv_mib']),
                          fmt(100*(1-r['final_kv_mib']/full_long['final_kv_mib']))+'%',fmt(r['peak_gib'])])
    sections.append(table(['方法','解码 token/s','整条秒','最终 KV MiB','KV 较 full 减少','峰值 GiB'],long_rows))
    sections.append('<p>该形状下，PhaseKV 约为 Full 的 2.16 倍解码速度、1.52 倍整条速度，最终 KV 减少 94.05%；'
                    '但 allocated 峰值为 18.09 GiB，高于 full 的 16.44 GiB。完整 prefill 后才压缩，尾部观察窗口分数及选择过程的临时内存会影响峰值。'
                    '这组结果支持“长上下文解码提速、最终 KV 更省”，不支持“全流程峰值显存更低”。</p>')
    sections.append('<h3>长输入与长输出联合测试</h3>')
    joint_rows=[]
    for inputs,outputs in [(8192,1024),(16384,2048),(24576,4096)]:
        for m in METHODS:
            r=find(m,'joint_sweep',inputs,outputs)
            joint_rows.append([f'{inputs}+{outputs}',LABELS[m],fmt(r['decode_tps']),fmt(r['total_s']),fmt(r['final_kv_mib']),fmt(r['peak_gib'])])
    sections.append(table(['输入+输出','方法','解码 token/s','整条秒','最终 KV MiB','峰值 GiB'],joint_rows))
    sections.append('<p>24576+4096 时，PhaseKV 和 SnapKV 约为 36.4 token/s，速度接近；PhaseKV 最终 KV 约 154 MiB，'
                    'SnapKV 约 627 MiB。只应称速度接近，不应将百分之几以内或更小的差异直接认定为稳定优势。</p>')
    sections.append('<h2>结论的适用范围</h2><ul>'
                    '<li>各方法不是相同 KV 字节预算：KIVI 2-bit 且保留所有 prompt；SnapKV 保留约30% prompt；'
                    'PhaseKV 保留30% prompt再用2-bit、生成冷缓存4-bit；SCOPE 为本仓库保留完整 prompt、限制生成缓存的实现。比较只代表这些具体配置。</li>'
                    '<li>此次数据只覆盖 Qwen2.5-7B，没有可用于本次对比的 Llama7B 记录。</li>'
                    '<li>质量结果只覆盖本次抽样任务与输入预算，不能直接推广到全部长上下文任务或其他压缩配置。</li>'
                    '<li>所有计时来自同一型号GPU和相同记录环境，但不同轮次未交错执行；没有全程温度、时钟、功率轨迹。微小速度差异应谨慎解读。</li>'
                    '</ul>')
    sections.append('<details><summary>完整性能表（17 种形状，每种 3 次正式测量）</summary>'+table(
        ['类型','输入','输出','方法','token/s中位数','token/s范围','整条秒','KV MiB','峰值 GiB'],
        [[r['sweep'],r['input_tokens'],r['output_tokens'],LABELS[m],fmt(r['decode_tps']),
          fmt(r['decode_tps_min'])+'–'+fmt(r['decode_tps_max']),fmt(r['total_s']),fmt(r['final_kv_mib']),fmt(r['peak_gib'])]
         for m in METHODS for r in perf[m]])+'</details>')
    sections.append('<details><summary>配置与一致性核对记录</summary><pre>'+cell(json.dumps(
        {k:v for k,v in audit.items() if k not in ('source_files','environment','source_differences_between_runs')},ensure_ascii=False,indent=2))+'</pre></details>')
    sections.append('<p class="muted">机器可读完整结果和文件哈希：<a href="comparison.json">comparison.json</a>。'
                    '原始 results 保持不变。可重复运行 scripts/summarize_downloaded_4090d.py 重新生成。</p>')
    css = '''body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f6fa;color:#172536;margin:0;line-height:1.7}
    main{max-width:1180px;margin:auto;padding:38px 28px 80px;background:white}h1{font-size:32px;line-height:1.3}h2{margin-top:42px;border-bottom:2px solid #dbe6f0;padding-bottom:8px}h3{margin-top:25px}.lead{color:#526578}.scroll{overflow:auto;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:14px}th{background:#163d58;color:white;text-align:left;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #dce5ee}tr:nth-child(even){background:#f4f8fb}td:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}tbody tr:hover{background:#e6f3f0}.note{padding:18px;background:#fff5da;border-left:4px solid #c99b28}details{margin:24px 0}summary{cursor:pointer;font-weight:700}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.muted{color:#667787}a{color:#086d82}@media print{body,main{background:white}main{max-width:none;padding:0}table{font-size:10px}.scroll{overflow:visible}h2{break-after:avoid}tr{break-inside:avoid}details{display:none}}'''
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhaseKV 与基线算法 · RTX 4090 D 实测对比</title><style>'+css+'</style><main><h1>PhaseKV 与基线算法实测对比</h1><p class="lead">2026-10-25 · Qwen2.5-7B-Instruct · RTX 4090 D · BF16 权重 · batch=1</p>'+''.join(sections)+'</main></html>'
    (args.out/'report.html').write_text(document,encoding='utf-8')
    print(json.dumps({'report':str((args.out/'report.html').resolve()),'quality':quality,
                      'paired_quality':pair_stats},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
