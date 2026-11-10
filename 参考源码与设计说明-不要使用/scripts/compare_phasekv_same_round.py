"""Five-method, same-round Qwen benchmark with two Full controls and matched prefill."""
import argparse
import html
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import retest_phasekv_prompt as t
b,s=t.b,t.s
HERE=Path(__file__).resolve()
VERSION="same-round-v1-five-methods"


def comparison_metrics(*args, **kwargs):
    # The original sweep's formatter only knows Full and PhaseKV specs.
    # Keep its tested paired statistics unchanged and restore the local hook.
    original = s.label
    s.label = lambda spec: {'full':'Full', 'kivi':'KIVI', 'snapkv':'SnapKV',
                           'scope':'SCOPE', 'phasekv':'PhaseKV'}[spec['method']]
    try:
        return s.metrics(*args, **kwargs)
    finally:
        s.label = original


def configs():
    return [("Full-native",{"method":"full"},"native"),
            ("Full-matched",{"method":"full"},"matched"),
            ("KIVI",{"method":"kivi","bits":2,"group_size":32,"recent_window":64},"matched"),
            ("SnapKV",{"method":"snapkv","prefill_keep_ratio":.3,"obs_window":32},"matched"),
            ("SCOPE",{"method":"scope","recent_window":16,"max_cache_tokens":32,"decoding_metric":"slide"},"matched"),
            ("PhaseKV",s.spec(ratio=.7,bits=4,window=128,cold=4,keep=1.),"matched")]


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default=str(b.ROOT/'results/phasekv_same_round'))
    p.add_argument('--data-dir',default=str(b.ROOT/'.cache/longbench'))
    p.add_argument('--model-path')
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--sample-offset',type=int,default=72)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--max-quality-drop',type=float,default=1.)
    for flag in ('quick','resume','dry-run','report-only'):
        p.add_argument('--'+flag,action='store_true')
    a=p.parse_args(argv)
    if min(a.samples,a.repeats)<1 or a.sample_offset<0 or not math.isfinite(a.max_quality_drop) or a.max_quality_drop<0:
        p.error('Counts must be positive and offset/drop nonnegative')
    a.out,a.data_dir=str(Path(a.out).resolve()),str(Path(a.data_dir).resolve())
    return a


def settings(args):
    a,profile=t.settings(args)
    a['methods']=['full','kivi','snapkv','scope','phasekv']
    a['performance_shapes'] += [(1024,256,'short'),(4096,256,'context'),(8192,256,'context')]
    profile['methods']=[spec for name,spec,policy in configs() if name!='Full-native']
    return a,profile


def report(out,manifest):
    records,results={},{}
    expected=set(manifest['expected'])
    for name,spec,policy in manifest['plan']:
        folder=out/'runs'/name
        if not (folder/'complete.json').exists():
            continue
        job,done=t.read(folder/'job.json'),t.read(folder/'complete.json')
        if done!={'hash':b.file_hash(folder/'records.jsonl'),'protocol':job['protocol_hash']}:
            raise ValueError('Completed results modified: '+name)
        if job['protocol_hash']!=b.digest({'manifest':manifest['hash'],'name':name}) or job['spec']!=spec or job['prefill_policy']!=policy:
            raise ValueError('Job protocol mismatch: '+name)
        rows=b.read_records(folder/'records.jsonl')
        if any(r['protocol_hash']!=job['protocol_hash'] or r['parameters']!=spec or
               (r['status']=='ok' and r.get('prefill_policy')!=policy) for r in rows):
            raise ValueError('Record protocol mismatch: '+name)
        if len(rows)!=len({r['case_id'] for r in rows}):
            raise ValueError('Duplicate cases: '+name)
        records[name]=rows
    lookup={name:{r['case_id']:r for r in rows} for name,rows in records.items()}
    for name,spec,policy in manifest['plan']:
        if name not in records:
            continue
        results[name]={'spec':spec,'policy':policy,'comparisons':{}}
        if name not in ('Full-native','Full-matched'):
            if 'Full-matched' not in lookup:
                raise ValueError('Missing matched Full')
            for r in records[name]:
                t.verify_pair(r['case_id'],r,lookup['Full-matched'].get(r['case_id']))
        for baseline in ('Full-native','Full-matched'):
            if baseline in records:
                results[name]['comparisons'][baseline]=comparison_metrics(records[name],records[baseline],expected,spec,
                    manifest['settings']['max_quality_drop'],intervals=True)
        results[name]['compression_activity']={phase:dict(samples=len(selected),triggered=sum(r.get('decode_compression_steps',0)>0 for r in selected))
            for phase in ('quality','performance','needle')
            for selected in [[r for r in records[name] if r.get('phase')==phase and r['status']=='ok' and not r.get('is_warmup')]]}
    pairs={}
    if 'PhaseKV' in records:
        for other in ('KIVI','SnapKV','SCOPE'):
            if other in records:
                pairs[other]=comparison_metrics(records['PhaseKV'],records[other],expected,results['PhaseKV']['spec'],
                                      manifest['settings']['max_quality_drop'],intervals=True)
    prefill_audit={}
    for name,rows in records.items():
        q=[r for r in rows if r.get('phase')=='quality' and r['status']=='ok']
        prefill_audit[name]={'quality_samples':len(q),'against':{}}
        for baseline in ('Full-native','Full-matched'):
            if baseline in lookup:
                paired=[r for r in q if r['case_id'] in lookup[baseline] and lookup[baseline][r['case_id']]['status']=='ok']
                prefill_audit[name]['against'][baseline]=dict(paired=len(paired),
                    first_token_differences=sum(r['generated_ids'][0]!=lookup[baseline][r['case_id']]['generated_ids'][0] for r in paired),
                    logits_hash_differences=sum(r.get('prefill_logits_sha256')!=lookup[baseline][r['case_id']].get('prefill_logits_sha256') for r in paired))
    payload=dict(results=results,phasekv_vs_methods=pairs,prefill_audit=prefill_audit,completed=len(records),planned=len(manifest['plan']))
    b.atomic_json(out/'comparison.json',payload)
    blocks=['<h1>PhaseKV 与基线：同轮比较</h1>']
    def p(x):blocks.append('<p>'+html.escape(x)+'</p>')
    def table(headers,rows):
        blocks.append('<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>')
    p(f"已完成 {len(records)}/{len(manifest['plan'])} 组。Qwen2.5-7B-Instruct，单GPU、单请求、原始BF16权重、同文档/输入/生成长度/seed。四种压缩方法统一使用Full-matched的prefill路径，并逐样本检查压缩前logits哈希和首token。Full-native另列作原生系统参考。")
    p('这是明确配置下的同轮比较，不是同等KV预算或同等准确率实验。SnapKV保留30%；PhaseKV保留70%并量化为4-bit；KIVI为2-bit、残差64；SCOPE为ALLKV+Slide（GQA适配）、生成预算32、近期窗口16。SCOPE保留完整prompt，未实现其adaptive/discontinuous等其他变体。')
    p('统一的是数据、prefill起点和测量方式；decode保留各算法必需的计算路径。例如SCOPE需要逐步attention分数，KIVI/PhaseKV使用本项目packed Triton内核。结果描述这些具体实现，不是对所有实现或算法理论上限的排名。')
    for baseline in ('Full-matched','Full-native'):
        blocks.append('<h2>相对 '+baseline+'</h2>')
        tab=[]
        for name,value in results.items():
            r=value['comparisons'].get(baseline)
            if not r:continue
            tab.append([name,f"{r['qa']:.2f}",f"{r['summary']:.2f}",f"{r['speed']:.3f}×",f"{100*r['kv']:.2f}%",f"{r['peak_gib']:.2f}",'通过' if r['feasible'] else '未通过'] if r['valid'] else [name,'缺失/OOM：不参与排名'])
        table(['方法','QA F1','摘要ROUGE-L','解码速度比','实际KV比','性能峰值GiB','1分均值门槛'],tab)
    blocks.append('<h2>固定长度性能与实际KV</h2>')
    tab=[]
    for name,value in results.items():
        r=value['comparisons'].get('Full-matched')
        if not r or not r['valid']:continue
        for x in r['shapes']:
            tab.append([name,f"{x['input']}+{x['output']}",f"{x['tokens_s']:.2f}",f"{x['prefill_s']:.3f}",f"{x['total_s']:.2f}",f"{x['prefill_kv_mib']:.2f}",f"{x['kv_mib']:.2f}",f"{x['peak_gib']:.2f}"])
    table(['方法','输入+输出','decode tok/s','prefill+压缩s','总耗时s','prefill KV MiB','最终KV MiB','峰值GiB'],tab)
    blocks.append('<h2>PhaseKV 相对其他压缩方法</h2>')
    table(['对照','QA差值','摘要差值','解码速度比','KV比'],[[name,f"{-r['qa_loss']:+.3f}",f"{-r['summary_loss']:+.3f}",f"{r['speed']:.3f}×",f"{100*r['kv']:.2f}%"] if r['valid'] else [name,'存在失败记录，无法作完整比较'] for name,r in pairs.items()])
    p('速度为每种形状正式重复的中位数，汇总比值为形状等权几何平均；不代表每个场景都加速。QA和摘要按任务分别平均；配对95%区间按文档聚类，未校正多重比较，均值门槛不代表统计等效或所有任务达标。')
    p('实际KV来自张量底层存储：打包方法计入代码、scale/min和高精度残差；dense方法按存储去重。KV不含模型权重、attention工作区或一般辅助状态，整进程峰值另列。质量任务自然遇EOS停止，实际输出长度可能不同；固定长度测速才是相同轨迹长度下的KV比较。')
    p('4096输出只测速，不评分长文语义；摘要质量输出上限512。容量探测可碰到模型上下文上限而非显存上限。压缩触发次数、任务/长度明细、needle及区间见下方和comparison.json。')
    for name,value in results.items():
        blocks.append('<details><summary>'+html.escape(name)+'：配置、质量与测量明细</summary><pre>'+html.escape(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>')
    blocks.append('<details><summary>压缩前一致性与PhaseKV直接配对区间</summary><pre>'+html.escape(json.dumps({'prefill':prefill_audit,'pairs':pairs},ensure_ascii=False,indent=2))+'</pre></details>')
    (out/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhaseKV 同轮对比</title><style>body{font:16px system-ui;line-height:1.7;max-width:1400px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #ccd5df}th{background:#edf4f8}pre{white-space:pre-wrap}.scroll{overflow-x:auto}details{margin:20px 0}</style>'+''.join(blocks)+'</html>',encoding='utf-8')
    return results


def main(argv=None):
    args=arguments(argv);out=Path(args.out)
    if args.report_only:
        report(out,t.read(out/'manifest.json'));return
    a,profile=settings(args);plan=configs()
    for name,spec,policy in plan:b.validate_spec(spec,a['chunk_size'])
    if args.dry_run:
        print(json.dumps(dict(groups=plan,settings=a,quality_cases_per_group=len(a['tasks'])*len(a['quality_lengths'])*a['samples'],
            performance_cases_per_group=len(a['performance_shapes'])*(a['repeats']+1)),ensure_ascii=False,indent=2));return
    out.mkdir(parents=True,exist_ok=True);mf=out/'manifest.json';lock=out/'RUNNING.lock'
    if mf.exists() and not args.resume:raise FileExistsError('Use --resume with identical options or a new --out')
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        b.check_sources();runner=str(b.ROOT/'scripts/benchmark_4090d.py')
        s.execute([sys.executable,runner,'__check_environment__'],out/'environment.log')
        data=b.prepare_data(args.data_dir,a['tasks']);expected=s.expected_keys(a,args.data_dir)
        b.atomic_json(out/'identity_job.json',[profile])
        s.execute([sys.executable,runner,'__identity__',str(out/'identity_job.json'),str(out/'identity.json')],out/'identity.log')
        sources=b.source_hashes()
        sources.update({n:b.file_hash(b.ROOT/n) for n in ['scripts/sweep_phasekv.py','scripts/retest_phasekv_prompt.py','scripts/compare_phasekv_same_round.py','run_phasekv_same_round.sh']})
        manifest=dict(version=VERSION,plan=plan,expected=sorted(expected),profile=profile,worker_settings=a,
            settings={k:v for k,v in vars(args).items() if k not in ('resume','dry_run','report_only')},data=data,source_hashes=sources,checkpoint=t.read(out/'identity.json'))
        manifest['hash']=b.digest(manifest)
        if mf.exists() and t.read(mf)['hash']!=manifest['hash']:raise ValueError('Code/data/settings/checkpoint changed; use a new --out')
        b.atomic_json(mf,manifest)
        for i,(name,spec,policy) in enumerate(plan,1):
            print(f'\n=== comparison: {i}/{len(plan)} {name} ===',flush=True)
            folder=out/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
            vp,validation=folder/'validation_job.json',folder/'validation.json'
            b.atomic_json(vp,{'profiles':[dict(profile,methods=[spec])],'prefill_policy':policy})
            if not validation.exists() or t.read(validation).get('status')!='passed':
                s.execute([sys.executable,'-u',str(HERE),'__validate__',str(vp),str(validation)],folder/'validation.log')
            rp,jp,done=folder/'records.jsonl',folder/'job.json',folder/'complete.json'
            job=dict(name=profile['name'],architecture=profile['architecture'],spec=spec,args=a,prefill_policy=policy,
                model_path=manifest['checkpoint'][profile['name']]['resolved_path'],records_path=str(rp),
                protocol_hash=b.digest({'manifest':manifest['hash'],'name':name}),
                matched_baseline=str(out/'runs/Full-matched/records.jsonl') if spec['method']!='full' else None)
            if done.exists():
                if t.read(done)!={'hash':b.file_hash(rp),'protocol':job['protocol_hash']}:raise ValueError('Completed records changed: '+name)
            else:
                b.atomic_json(jp,job)
                s.execute([sys.executable,'-u',str(HERE),'__worker__',str(jp)],folder/'run.log')
                b.atomic_json(done,{'hash':b.file_hash(rp),'protocol':job['protocol_hash']})
            results=report(out,manifest)
            if spec['method']=='full' and not results[name]['comparisons'][name]['valid']:raise RuntimeError('Full baseline incomplete')
        b.atomic_json(out/'complete.json',{'groups':[x[0] for x in plan],'manifest_hash':manifest['hash']})
        print(f"\nComplete: {out/'report.html'}",flush=True)
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='__worker__':
            job=t.read(sys.argv[2])
            with t.worker_policy(job['prefill_policy'],job.get('matched_baseline')):b.worker(job)
        elif len(sys.argv)>1 and sys.argv[1]=='__validate__':t.validate_job(t.read(sys.argv[2]),sys.argv[3])
        else:main()
    except KeyboardInterrupt:
        print('Interrupted; rerun identical options plus --resume.',file=sys.stderr);sys.exit(130)
