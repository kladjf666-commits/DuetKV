"""Read-only audit of downloaded retest records, with paired factorial contrasts."""
import collections
import hashlib
import html
import json
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retest_phasekv_prompt as t
s, b = t.s, t.b
SOURCE = b.ROOT / "results/downloads/20261027_prompt_retest/results/phasekv_prompt_retest"
OUT = b.ROOT / "reports/phasekv_prompt_retest_20261027"


def label(ratio, bits):
    return f"P{ratio}-B{bits}-W128-C4-D1"


def main():
    manifest, saved = t.read(SOURCE / "manifest.json"), t.read(SOURCE / "comparison.json")
    assert b.digest({k:v for k,v in manifest.items() if k != "hash"}) == manifest["hash"]
    names = [x[0] for x in manifest["plan"]]
    assert t.read(SOURCE / "complete.json") == {"groups": names, "manifest_hash": manifest["hash"]}
    assert {p.name for p in (SOURCE / "runs").iterdir() if p.is_dir()} == set(names)
    expected, raw, rows, counts, envs, audit = set(manifest["expected"]), {}, {}, collections.Counter(), [], {}
    for name, spec, policy in manifest["plan"]:
        folder = SOURCE / "runs" / name
        job, done = t.read(folder / "job.json"), t.read(folder / "complete.json")
        assert b.file_hash(folder / "records.jsonl") == done["hash"]
        assert done["protocol"] == job["protocol_hash"] and job["spec"] == spec and job["prefill_policy"] == policy
        assert job["protocol_hash"] == b.digest({"manifest":manifest["hash"], "name":name})
        records = b.read_records(folder / "records.jsonl")
        assert len(records) == len({r["case_id"] for r in records})
        assert all(r["protocol_hash"] == done["protocol"] and r["parameters"] == spec and r["prefill_policy"] == policy for r in records)
        assert t.read(folder / "validation.json")["status"] == "passed"
        raw[name] = {r["case_id"]:r for r in records}
        counts.update(r["status"] for r in records)
        envs.extend(t.read(p) for p in folder.glob("records.environment.*.json"))
        audit[name] = dict(records=len(records), phases=dict(collections.Counter(r["phase"] for r in records)),
                          capacity=t.read(folder / "records.capacity.json"))
        if spec["method"] == "phasekv":
            for r in records:
                t.verify_pair(r["case_id"],r,raw["Full-matched"].get(r["case_id"]))
    for name,spec,policy in manifest["plan"]:
        comparisons = {}
        for baseline in ("Full-native","Full-matched"):
            result = s.metrics(list(raw[name].values()),list(raw[baseline].values()),expected,spec,1.,intervals=True)
            assert result == next(r for r in saved["rows"] if r["name"] == name)["comparisons"][baseline]
            comparisons[baseline] = result
        rows[name] = comparisons
    qkeys = {k for k,r in raw["Full-matched"].items() if r["phase"] == "quality"}
    differences, exact = {}, {}
    for name in names:
        differences[name] = {base:sum(raw[name][k]["generated_ids"][0] != raw[base][k]["generated_ids"][0] for k in qkeys) for base in ("Full-native","Full-matched")}
        exact[name] = dict(all=sum(raw[name][k]["generated_ids"] == raw["Full-matched"][k]["generated_ids"] for k in qkeys),
            qa=sum(raw[name][k]["generated_ids"] == raw["Full-matched"][k]["generated_ids"] for k in qkeys if raw[name][k]["metric"] == "qa_f1"))
    assert differences == saved["first_token_differences"]
    env_fields = ("gpu","torch","cuda","transformers","triton","chat_template_sha256")
    environment = {k:sorted({str(e[k]) for e in envs}) for k in env_fields}
    assert all(len(v) == 1 for v in environment.values())
    docs = {(r["task"],r["sample_index"]) for r in raw["Full-matched"].values() if r["phase"] == "quality"}
    previous_docs = set()
    for stage in ("prefill","final"):
        previous_docs |= {(r["task"],r["sample_index"]) for r in b.read_records(b.ROOT / f"results/downloads/20261026_parameter_sweep/results/phasekv_parameter_sweep/{stage}/Full/records.jsonl") if r["phase"] == "quality"}
    assert previous_docs and not docs & previous_docs
    contrasts = []
    pairs = [("位宽",label(ratio,hi),label(ratio,lo)) for ratio in ("0.7","1") for lo,hi in ((2,4),(4,8))]
    pairs += [("保留比例",label("1",bits),label("0.7",bits)) for bits in (2,4,8)]
    for axis,left,right in pairs:
        a,z = rows[left]["Full-matched"],rows[right]["Full-matched"]
        contrasts.append(dict(axis=axis,left=left,right=right,qa_delta=a["qa"]-z["qa"],summary_delta=a["summary"]-z["summary"],
                              **s.paired_intervals(raw[left],raw[right],qkeys)))
    summary = dict(source=str(SOURCE),statuses=dict(counts),jobs=audit,validation_passes=len(names),unique_documents=len(docs),
        previous_document_overlap=0,environment=environment,first_token_differences=differences,exact_sequences=exact,rows=rows,contrasts=contrasts)
    OUT.mkdir(parents=True,exist_ok=True)
    b.atomic_json(OUT / "analysis.json",summary)
    for mode,name in (("balanced",label("0.7",4)),("summary",label("1",4)),("fidelity",label("1",8))):
        b.atomic_json(OUT / f"phasekv_{mode}.json",{"methods":[rows[name]["Full-matched"]["spec"]]})
    blocks, text = ["<h1>PhaseKV 六组 prompt 复测分析</h1>"], []
    def p(value,warn=False):
        text.append(value);blocks.append(f'<p class="{"note" if warn else ""}">{html.escape(value)}</p>')
    def h(value):
        text.append("\n"+value);blocks.append("<h2>"+html.escape(value)+"</h2>")
    def table(headers,data):
        text.append(" | ".join(headers));text.extend(" | ".join(map(str,r)) for r in data)
        blocks.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in data)+'</tbody></table></div>')
    def f(x):return f"{x:.2f}"
    def ci(v):return f"[{v[0]:+.3f}, {v[1]:+.3f}]"
    p("2026-10-27｜Qwen2.5-7B-Instruct｜RTX 4090 D｜单请求、原始BF16权重。全部六组固定热区128、冷区4-bit、生成保留100%；只改变prompt保留比例和位宽。")
    h("结论与配置选择")
    p("默认折中建议采用70%保留＋4-bit：相对同路径Full，QA只低0.042分，摘要低0.463分，达到原定分组均值各下降不超过1分的门槛；六种形状KV比值的几何平均为26.13%，解码速度比约1.165×。2-bit两组均未通过同路径Full质量门槛。",True)
    p("如果更重视摘要完整性，100%保留＋4-bit更合适：QA低0.299分，摘要高0.012分，KV比值33.36%。如果希望生成结果更接近同路径Full，可选100%＋8-bit，但KV比值升到51.55%，高于4-bit的成本较明显。这里的选择是本轮取舍，不是对所有任务的最优保证。")
    table(["用途","prompt保留","prompt位宽","热区","冷区位宽","生成保留"],[['默认折中','70%','4-bit',128,'4-bit','100%'],['摘要优先','100%','4-bit',128,'4-bit','100%'],['更接近Full的输出','100%','8-bit',128,'4-bit','100%']])
    h("完整性与一致性")
    p("8组均完成，每组1108条记录：1024质量、54 needle、24性能（含预热）、6容量探测。总计8864条全部ok、8份GPU校验通过，无OOM。完成文件SHA256、协议与参数一致；原始逐条数据重算与报告完全一致。模型、库版本与tokenizer模板的运行记录一致。ok表示运行成功，不表示答案正确。")
    p("每任务32篇，共256篇独立文档，各测试4种输入预算；本轮seed1234、打乱顺序offset40，与上轮筛选及复核文档均不重叠。1024条质量记录不是1024篇独立文档，统计区间按任务分层、按文档聚类，保留同篇的全部长度一起重采样。")
    p("六组PhaseKV的6144条质量记录，以及对应的性能、needle和可配对容量记录，都通过与Full-matched的输入、完整末位prefill logits哈希、首token一致性核对。压缩前比较起点现已一致；这不等于证明压缩后的输出无损。")
    h("为何必须区分两个Full")
    table(["基线","QA F1","摘要ROUGE-L","性能峰值GiB"],[[n,f(rows[n]['Full-matched']['qa']),f(rows[n]['Full-matched']['summary']),f(rows[n]['Full-matched']['peak_gib'])] for n in names[:2]])
    p("Full-native使用原生SDPA prefill；Full-matched和PhaseKV一样，在prefill末块请求attention分数，保留dense KV。两种Full在172/1024条质量样本的首token上不同；同路径Full的QA均值高1.601分、摘要低0.036分。不能据此断言某后端普遍更准，这是本批文档上的路径差异。")
    p("相对Full-native，六组PhaseKV都通过1分均值门槛；相对Full-matched，只有4/8-bit四组通过。仅比较原生Full会掩盖2-bit的QA损失。下面质量与参数归因以Full-matched为主，实际系统速度另列Full-native。",True)
    h("六组最终结果")
    table(["配置：保留/位宽","QA F1↑","摘要ROUGE-L↑","QA下降↓","摘要下降↓","解码比↑","KV比↓","needle","门槛"],[[('Full-matched' if n=='Full-matched' else n.split('-W')[0]),f(rows[n]['Full-matched']['qa']),f(rows[n]['Full-matched']['summary']),f(rows[n]['Full-matched']['qa_loss']),f(rows[n]['Full-matched']['summary_loss']),f"{rows[n]['Full-matched']['speed']:.3f}×",f"{100*rows[n]['Full-matched']['kv']:.2f}%",f"{round(rows[n]['Full-matched']['needle']['score']*54/100)}/54",'通过' if rows[n]['Full-matched']['feasible'] else '未通过'] for n in names[1:]])
    p("QA和摘要是不同指标，均为0–100分，分别按任务等权汇总。通过门槛只针对两类均值，不代表每个任务都达标、统计等效已证明或输出逐token一致。解码与KV比为六种固定形状的等权几何平均；不是所有场景统一加速，也不是模型总显存节省率。")
    h("位宽与保留比例的独立影响")
    table(["改变（其余参数固定）","QA变化","QA配对95%区间","摘要变化","摘要配对95%区间"],[[c['left'].split('-W')[0]+' − '+c['right'].split('-W')[0],f"{c['qa_delta']:+.3f}",ci(c['qa_delta_ci95']),f"{c['summary_delta']:+.3f}",ci(c['summary_delta_ci95'])] for c in contrasts])
    p("提高位宽：70%保留下2→4-bit，QA均值提高1.328分；100%保留下提高1.179分。两次直接QA差值的区间仍略跨零，不能表述为都已统计显著；但两种2-bit配置相对同路径Full的QA差值区间都低于零，4-bit则接近Full并通过均值门槛。4→8-bit的分数增益不稳定，不能仅因位宽更高就认定更好。")
    p("提高保留比例：70%→100%在4-bit下摘要提高0.475分，8-bit下提高0.526分；后者配对95%区间约[0.032, 1.030]，前者仍跨零。QA没有同步提高，2-bit下提高保留比例也没有修复QA损失。你之前本地观察到的保留比例收益，在本轮主要体现在4/8-bit摘要，而不是所有任务普遍上升。")
    p("全量保留＋2-bit仍损失约1.477分QA，而70%＋4-bit只损失0.042分：在本轮设置下，优先把prompt从2-bit升到4-bit，比仅将2-bit的保留比例升到100%更值得考虑。")
    h("质量区间与输出一致率")
    table(["配置","QA相对matched 95%区间","摘要相对matched 95%区间","QA完整输出相同/640","全部完整输出相同/1024"],[[n.split('-W')[0],ci(rows[n]['Full-matched']['qa_delta_ci95']),ci(rows[n]['Full-matched']['summary_delta_ci95']),exact[n]['qa'],exact[n]['all']] for n in names[2:]])
    p("100%＋8-bit有619/640条QA完整token序列与同路径Full一致，比70%＋4-bit的536/640更接近原输出；但长输出中一次分叉就会使完整序列计数不同，所以该一致率不能替代任务分数。默认70%＋4-bit的摘要差值区间下端约−1.081分，因此均值达标尚不等于以95%置信度证明损失一定小于1分。")
    h("按任务：默认折中与摘要方案")
    choices=['Full-matched',label('0.7',4),label('1',4),label('1',8)]
    table(["任务","指标"]+[n.split('-W')[0] for n in choices],[[task,'ROUGE-L' if task in b.SUMMARY_TASKS else 'F1']+[f(rows[n]['Full-matched']['tasks'][task]['score']) for n in choices] for task in b.TASK_LIMITS])
    p("70%＋4-bit的multi_news相对同路径Full下降约1.048分，虽然摘要分组均值达标；100%＋4-bit将multi_news损失降至约0.150分，gov_report损失也从约0.731缩小到0.459分。这是选择摘要方案的实际依据。")
    h("输入预算分层")
    lr=[]
    for length in (1536,3072,8192,16384):
        for summ,metric in ((False,'QA'),(True,'摘要')):
            lr.append([length,metric]+[f(st.mean(x['score'] for x in rows[n]['Full-matched']['quality_by_length'] if x['length']==length and (x['task'] in b.SUMMARY_TASKS)==summ)) for n in choices])
    table(["输入预算","指标"]+[n.split('-W')[0] for n in choices],lr)
    p("70%＋4-bit在1536预算下摘要低约1.05分，在其余预算下降约0.20–0.39分。100%＋4-bit在各预算的摘要均值更接近Full。预算是截断上限，不能认为每条输入都恰好达到该长度。")
    h("速度、KV与端到端耗时")
    balanced=rows[label('0.7',4)]['Full-native'];native=rows['Full-native']['Full-native']
    perf=[]
    for a,z in zip(balanced['shapes'],native['shapes']):
        perf.append([f"{a['input']}+{a['output']}",f(z['tokens_s']),f(a['tokens_s']),f"{a['speed_vs_full']:.2f}×",f(a['kv_mib']),f"{100*(1-a['kv_ratio']):.1f}%",f"{z['total_s']:.2f} → {a['total_s']:.2f}"])
    table(['输入+输出','原生Full tok/s','70%/4-bit tok/s','解码速度比','PhaseKV MiB','KV节省','Full→PhaseKV总耗时s'],perf)
    p("32000+256：70%/4-bit为33.86 tok/s，原生Full为19.65，解码约1.72×；总耗时20.29→14.92秒，端到端约1.36×，KV节省77.73%。24576+4096：解码约1.39×，端到端约1.38×，KV节省76.01%。")
    p("1024+4096：70%/4-bit只有33.70 tok/s，Full约44.51，解码慢约24.3%，总耗时92.13→121.68秒。短输入时若显存允许，Full更适合追求吞吐；不能将平均1.17×宣传为普遍收益。")
    p("100%/4-bit在32000+256约29.22 tok/s、KV556.86MiB；70%/4-bit约33.86 tok/s、392.79MiB。100%/8-bit约28.56 tok/s、994.36MiB。提高保留和位宽需要付出明显KV成本；4/8-bit之间细小速度差不一定稳定，每种形状仅3次正式计时。")
    p("70%/4-bit的性能峰值约18.15GiB，原生Full16.44GiB，同路径Full18.09GiB。取attention分数的prefill路径本身已抬高峰值，因此KV节省73.87%的几何平均不意味着整卡显存也节省73.87%。全部容量探测到32640+128的上下文上限，不是填满24GB后的最大容量证明。")
    h("建议如何采用结果")
    p("以70%/4-bit作为长上下文默认候选；摘要完整性优先时选100%/4-bit；需要尽量贴近Full输出时选100%/8-bit，并接受更高KV成本。保留W128/C4/D1，本轮并没有重新证明其他生成参数更优。两种2-bit方案可保留为极致压缩对照，不再作为质量优先默认。")
    p("原始自动推荐quality为100%/8-bit，是按最大分组均值损失最小选择；它不是每个任务分数都最高。speed/compression均为70%/4-bit。报告另外提供100%/4-bit摘要方案，便于按用途选择，而不是强行合成一个统一最佳。")
    p("结果只适用于本次Qwen2.5-7B、单请求与硬件/软件设置，不可直接外推Llama或并发服务。长4096输出仅测吞吐，语义质量评估摘要上限512。区间为1000次文档聚类bootstrap，未做多重比较校正；选优后的正式泛化结论仍应使用独立测试集。本轮使用新文档，不能直接与上一轮绝对分数相减。")
    p("原始prompt_reset_result和既有报告未修改。analysis.json包含原始记录审计与重算结果；phasekv_balanced.json、phasekv_summary.json、phasekv_fidelity.json可用于复现三种配置。")
    (OUT/'analysis.txt').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    (OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhaseKV 六组复测分析</title><style>body{font:16px system-ui,"Microsoft YaHei",sans-serif;line-height:1.75;color:#172d3d;background:#f4f7fa;margin:0}main{max-width:1320px;margin:30px auto;padding:30px;background:white}h1,h2{line-height:1.4}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #d5dfe8;text-align:right}th{background:#e8f1f6}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto;margin:18px 0}.note{padding:16px;background:#e7f3ee;border-left:4px solid #23866f}</style></head><body><main>'+''.join(blocks)+'</main></body></html>',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'statuses':dict(counts),'groups':len(names),'quality_documents':len(docs),'validation_passes':len(names),'all_comparisons_recomputed':True,'prefill_logits_matched':True},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
