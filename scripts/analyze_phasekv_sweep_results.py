"""Audit downloaded sweep records and build a Chinese analysis; does not modify input."""
import collections
import hashlib
import html
import json
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_phasekv as s

ROOT = s.b.ROOT
SOURCE = ROOT / "results/downloads/20261026_parameter_sweep/results/phasekv_parameter_sweep"
OUT = ROOT / "reports/phasekv_sweep_20261026"
CURRENT = "P0.3-B2-W32-C4-D1"
CHOICE = "P0.7-B2-W128-C4-D1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x):
    return f"{x:.2f}"


def build():
    saved = read(SOURCE / "comparison.json")
    manifest = read(SOURCE / "manifest.json")
    stages, raw, audit, envs = {}, {}, {}, []
    for stage in ("prefill", "decode", "final"):
        plan = read(SOURCE / stage / "plan.json")
        expected = set(plan["expected"])
        specs = [s.FULL] + plan["specs"]
        names = [s.label(spec) for spec in specs]
        assert sorted(names) == read(SOURCE / stage / "stage_complete.json")
        actual_dirs = {p.name for p in (SOURCE / stage).iterdir() if p.is_dir()}
        assert actual_dirs == set(names)
        stages[stage], raw[stage] = [], {}
        statuses, phases = collections.Counter(), collections.Counter()
        for spec in specs:
            name = s.label(spec)
            folder = SOURCE / stage / name
            job, done = read(folder / "job.json"), read(folder / "complete.json")
            assert job["spec"] == spec
            assert hashlib.sha256((folder / "records.jsonl").read_bytes()).hexdigest() == done["hash"]
            records = s.b.read_records(folder / "records.jsonl")
            assert all(r["protocol_hash"] == done["protocol"] == job["protocol_hash"] and r["parameters"] == spec for r in records)
            raw[stage][name] = records
            statuses.update(r["status"] for r in records)
            phases.update(r["phase"] for r in records)
            result = s.metrics(records, raw[stage]["Full"], expected, spec, manifest["settings"]["max_quality_drop"], intervals=stage == "final")
            assert result == next(r for r in saved[stage] if r["name"] == name), (stage, name, "summary differs from raw records")
            stages[stage].append(result)
            envs.extend(read(p) for p in folder.glob("records.environment.*.json"))
        audit[stage] = dict(jobs=len(names), records=sum(statuses.values()), statuses=dict(statuses), phases=dict(phases))
    validations = [read(p) for p in (SOURCE / "validation").glob("*/report.json")]
    assert all(v["status"] == "passed" for v in validations)
    audit["validation_count"] = len(validations)
    audit["total_records"] = sum(audit[x]["records"] for x in stages)
    fields = ("gpu", "torch", "cuda", "transformers", "triton", "tokenizer_class", "chat_template_sha256")
    audit["environment_variants"] = {k: sorted({str(e[k]) for e in envs}) for k in fields}
    assert all(len(v) == 1 for v in audit["environment_variants"].values())
    docs = lambda rr: {(r["task"], r["sample_index"]) for r in rr if r["phase"] == "quality"}
    assert not docs(raw["prefill"]["Full"]) & docs(raw["final"]["Full"])
    assert docs(raw["prefill"]["Full"]) == docs(raw["decode"]["Full"])
    audit["screen_unique_documents"] = len(docs(raw["prefill"]["Full"]))
    audit["final_unique_documents"] = len(docs(raw["final"]["Full"]))
    baseline_predictions = lambda rr: {r["case_id"]: (r["input_hash"], r["generated_ids"], r["score"]) for r in rr if r["phase"] == "quality"}
    assert baseline_predictions(raw["prefill"]["Full"]) == baseline_predictions(raw["decode"]["Full"])
    runner = (ROOT / "scripts/benchmark_4090d.py").read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    audit["runner_matches_recorded_source"] = hashlib.sha256(runner).hexdigest() == manifest["source_hashes"]["scripts/benchmark_4090d.py"]
    assert audit["runner_matches_recorded_source"]
    first = {}
    for stage, name in (("prefill", "P1-B8-W64-C4-D1"), ("final", CURRENT), ("final", CHOICE)):
        full = {r["case_id"]: r for r in raw[stage]["Full"]}
        quality = [r for r in raw[stage][name] if r["phase"] == "quality"]
        different = [r for r in quality if r["generated_ids"][0] != full[r["case_id"]]["generated_ids"][0]]
        first[stage + "/" + name] = dict(total=len(quality), different=len(different),
            tasks=dict(collections.Counter(r["task"] for r in different)), examples=[dict(case_id=r["case_id"],
                full_first=full[r["case_id"]]["generated_ids"][0], phase_first=r["generated_ids"][0],
                full_prediction=full[r["case_id"]]["prediction"], phase_prediction=r["prediction"]) for r in different[:3]])
    phase_names = [r["name"] for r in stages["final"] if r["name"] != "Full"]
    first_maps = [{r["case_id"]: r["generated_ids"][0] for r in raw["final"][name] if r["phase"] == "quality"} for name in phase_names]
    assert all(m == first_maps[0] for m in first_maps)
    audit["all_phasekv_final_first_tokens_identical"] = True
    final = {r["name"]: r for r in stages["final"]}
    contrasts = {}
    for left, right in ((CHOICE, CURRENT), (CHOICE, "P0.7-B2-W128-C8-D1"), ("P0.7-B2-W128-C8-D1", "P0.7-B2-W32-C8-D1")):
        lr, rr = ({r["case_id"]: r for r in raw["final"][name]} for name in (left, right))
        keys = {k for k, r in lr.items() if r["phase"] == "quality"}
        contrasts[left + " minus " + right] = dict(qa_delta=final[left]["qa"] - final[right]["qa"],
            summary_delta=final[left]["summary"] - final[right]["summary"], **s.paired_intervals(lr, rr, keys))
    OUT.mkdir(parents=True, exist_ok=True)
    s.b.atomic_json(OUT / "analysis.json", dict(source=str(SOURCE), audit=audit, stages=stages, paired_contrasts=contrasts, first_token_audit=first))
    s.b.atomic_json(OUT / "candidate_balanced.json", {"methods": [final[CHOICE]["spec"]],
        "analysis_note": "Conditional candidate only: failed the original 1-point QA/summary gate; Full prefill backend differs. Not a validated universal optimum."})

    blocks, plain = [], []
    def paragraph(text, warning=False):
        plain.append(text)
        blocks.append(f'<p class="{"warning" if warning else ""}">{html.escape(text)}</p>')
    def heading(text):
        plain.append("\n" + text)
        blocks.append("<h2>" + html.escape(text) + "</h2>")
    def table(headers, rows):
        plain.append(" | ".join(headers))
        plain.extend(" | ".join(map(str, row)) for row in rows)
        blocks.append('<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>' +
                      ''.join('<tr>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>')
    blocks.append('<h1>PhaseKV 参数搜索：结果与限制</h1>')
    paragraph("分析日期：2026-10-26。Qwen2.5-7B-Instruct，RTX 4090 D，单请求、原始 BF16 权重。这里比较不同 PhaseKV 参数配置，不涉及实现版本对比。")
    heading("结论")
    paragraph("没有 PhaseKV 配置通过预设质量门槛：QA F1 和摘要 ROUGE-L 相对 Full 的分组均值各下降不超过 1 分。因此原始 recommendations.json 为空是正确结果，并非导出失败。", True)
    paragraph("若接受约 1.45 分 QA 均值下降，已复核方案中更值得保留的折中候选为：prefill 保留70%、prompt 2-bit、热区128、冷区4-bit、生成保留100%。它改善摘要，QA没有改善；不应称为全局最优或已达标配置。")
    paragraph("发现准确率归因限制：最终复核1024条质量记录中，198条（19.34%）在KV压缩之前的首个生成token就与Full不同。代码与运行时记录吻合：Full prefill使用SDPA，PhaseKV为取注意力分数在最后一块请求output_attentions，转到手动attention路径。当前结果可描述这两条完整推理流程的表现，但不能把全部差距归因于KV位宽或token丢弃。", True)
    heading("完整性与样本")
    table(["阶段", "任务组数（含Full）", "成功记录", "质量记录", "性能记录（含预热）"], [[x, audit[x]["jobs"], audit[x]["records"], audit[x]["phases"].get("quality",0), audit[x]["phases"].get("performance",0)] for x in stages])
    paragraph("46个阶段任务、8410条记录全部为ok；38份GPU数值校验全部通过。完成标记、原始文件SHA256、参数和协议哈希均一致，逐条重算结果与comparison.json一致。这里的ok表示完成运行，不表示回答正确。")
    paragraph("筛选用4任务×8篇=32篇文档，每篇2种输入预算；复核用8任务×32篇=256篇不同文档，每篇4种预算，合计每组1024条质量记录。筛选与复核文档不重叠，但同一篇的4个长度不是4份独立样本。全程无OOM；所有最终配置容量探测均到32640输入+128输出，先达到原生32768上限，并未证明显存容量极限。")
    heading("最终复核总表")
    paragraph("P=prompt保留比例，B=prompt位宽，W=热区窗口，C=冷区位宽，D=冷却批次保留比例。热区始终FP16。质量满分100；QA与摘要分别按任务等权平均。速度/KV为六种固定形状相对Full比值的等权几何平均，不能当作所有实际业务的统一收益。")
    table(["配置", "QA F1", "摘要ROUGE-L", "速度/Full", "KV/Full", "needle成功", "性能峰值GiB", "门槛"], [[r["name"], fmt(r["qa"]), fmt(r["summary"]), f'{r["speed"]:.3f}×', f'{r["kv"]*100:.2f}%', f'{round(r["needle"]["score"]*54/100)}/54', fmt(r["peak_gib"]), "通过" if r["feasible"] else "未通过"] for r in stages["final"]])
    c, old = final[CHOICE], final[CURRENT]
    contrast = contrasts[CHOICE + " minus " + CURRENT]
    paragraph(f"70%/2-bit/W128/C4相对30%/2-bit/W32/C4：QA {contrast['qa_delta']:+.3f}分，摘要 {contrast['summary_delta']:+.3f}分；配对文档bootstrap 95%区间分别为[{contrast['qa_delta_ci95'][0]:.3f}, {contrast['qa_delta_ci95'][1]:.3f}]与[{contrast['summary_delta_ci95'][0]:.3f}, {contrast['summary_delta_ci95'][1]:.3f}]。摘要改善较稳定，QA改善没有证据。这是组合参数变化的效果，不能单独归因于保留比例。")
    paragraph(f"候选相对Full：QA下降{c['qa_loss']:.3f}分（差值95%区间[{c['qa_delta_ci95'][0]:.3f}, {c['qa_delta_ci95'][1]:.3f}]）；摘要下降{c['summary_loss']:.3f}分（差值95%区间[{c['summary_delta_ci95'][0]:.3f}, {c['summary_delta_ci95'][1]:.3f}]）。QA区间跨零不等于等效性已获证明；原门槛按均值判断，仍未通过。")
    heading("按任务看：不是所有准确率都提高")
    table(["任务", "指标", "Full", "P30/B2/W32/C4", "P70/B2/W128/C4", "候选−Full", "候选−P30"], [[t, "ROUGE-L" if t in s.b.SUMMARY_TASKS else "F1", fmt(final["Full"]["tasks"][t]["score"]), fmt(old["tasks"][t]["score"]), fmt(c["tasks"][t]["score"]), f'{c["tasks"][t]["score"]-final["Full"]["tasks"][t]["score"]:+.2f}', f'{c["tasks"][t]["score"]-old["tasks"][t]["score"]:+.2f}'] for t in s.b.TASK_LIMITS])
    paragraph("主要收益来自gov_report与multi_news摘要；hotpotqa、narrativeqa和qmsum相对30%参数反而下降。needle由52/54降至50/54，不能声称提高保留比例后检索必然改善。短needle输出没有充分检验生成冷区。")
    heading("按输入长度看")
    length_rows = []
    for n in (1536,3072,8192,16384):
        for summary, metric in ((False,"QA F1"),(True,"摘要ROUGE-L")):
            values = [st.mean(x["score"] for x in final[name]["quality_by_length"] if x["length"]==n and (x["task"] in s.b.SUMMARY_TASKS)==summary) for name in ("Full",CURRENT,CHOICE)]
            length_rows.append([n,metric]+[fmt(v) for v in values])
    table(["输入预算", "指标", "Full", "P30/B2/W32/C4", "P70/B2/W128/C4"],length_rows)
    paragraph("8192和16384预算下，候选的QA仍明显低于Full，且比30%参数更低；摘要则在全部预算上改善。输入预算是截断上限，不保证每篇文档实际达到该长度。")
    heading("速度与压缩：长输入收益大，短输入长输出反而更慢")
    perf=[]
    for f, o, a in zip(final["Full"]["shapes"],old["shapes"],c["shapes"]):
        perf.append([f'{a["input"]}+{a["output"]}',fmt(f["tokens_s"]),fmt(o["tokens_s"]),fmt(a["tokens_s"]),f'{a["speed_vs_full"]:.2f}×',fmt(a["kv_mib"]),f'{100*(1-a["kv_ratio"]):.1f}%',f'{f["total_s"]:.2f}→{a["total_s"]:.2f}'])
    table(["输入+输出", "Full tok/s", "P30 tok/s", "候选 tok/s", "候选/Full", "候选KV MiB", "KV节省", "Full→候选总耗时s"],perf)
    paragraph("候选在32000+256下解码约1.73×，端到端约1.36×；在24576+4096下解码约1.40×，端到端约1.39×。在1024+4096下解码仅Full的0.76×，总耗时由91.56秒增至120.76秒。不能用几何平均1.17×宣传为所有场景加速17%。")
    paragraph("性能测试峰值约18.09GiB，Full约16.44GiB。KV大幅缩小不保证峰值显存更低：prefill注意力分数及临时缓冲仍会抬高峰值。prefill+压缩耗时与Full接近、略高，主要收益来自长上下文decode。计时包含Python控制与缓存维护；每形状只有3次正式重复，小于约1%的差异不宜排出确定名次。")
    heading("各参数能够得出什么结论")
    paragraph("prefill保留比例：摘要通常随保留更多token而改善。当前只复核了30%与70%两类prompt配置，70%是可取的折中候选，尚不能断定50%不合适或100%没有价值。")
    paragraph("prompt位宽：这轮不能证明2-bit优于4/8-bit。小样本筛选所有候选都未达标，脚本按最大分组损失回退，QA成为决定项，导致只有P70/B2和P100/B2进入decode；最后3名又全部落在P70/B2。P100/B8筛选QA仍比Full低7.26分，但14/64条记录压缩前首token已不同，这不是纯量化损失测量。")
    paragraph("热区窗口：固定P70/B2/C8/D1，W128比W32摘要均值高约0.184分，但配对95%区间[-0.087, 0.439]跨零，尚不能断言稳定改善。W128候选640条QA中冷却触发数全部为0，全部输出不超过64token；因此QA根本没有检验其冷区位宽或保留比例。")
    paragraph("冷区位宽：固定P70/B2/W128/D1，4-bit与8-bit摘要分别25.215、25.206，QA相同。4-bit的KV比值18.91%，8-bit为22.91%；没有观察到8-bit质量收益，4-bit更适合作为折中默认值。它并非每个形状都更快。")
    paragraph("生成保留比例：0.5/0.75只做了筛选，没有进入最终复核。QA大多太短，不足以验证冷却丢弃的影响；现有证据不足以安全降低该比例。继续保持1.0。4096-token输出是强制长度测速，未评分其语义连贯性，不能据此宣称长输出质量保持。")
    heading("筛选矩阵（只作探索，不与复核绝对分数混比）")
    table(["配置", "QA F1", "摘要ROUGE-L", "速度/Full", "KV/Full"], [[r["name"],fmt(r["qa"]),fmt(r["summary"]),f'{r["speed"]:.3f}',f'{100*r["kv"]:.2f}%'] for r in stages["prefill"]])
    blocks.append('<details><summary>展开26组decode筛选结果</summary>')
    table(["配置", "QA F1", "摘要ROUGE-L", "速度/Full", "KV/Full"], [[r["name"],fmt(r["qa"]),fmt(r["summary"]),f'{r["speed"]:.3f}',f'{100*r["kv"]:.2f}%'] for r in stages["decode"]])
    blocks.append('</details>')
    heading("首token差异与下一步优先级")
    table(["记录范围", "压缩前首token不同", "总质量记录"], [[k,v["different"],v["total"]] for k,v in first.items()])
    paragraph("generate()先从prefill logits选出generated_ids[0]，之后才调用build_cache压缩KV。因此这198条首token差异在逻辑上不能由随后发生的KV量化/筛选造成。四组PhaseKV的首token逐条完全一致，说明PhaseKV参数内部比较有共同起点；Full与PhaseKV的绝对差距则包含prefill路径因素。当前证据不区分浮点数值差异放大与具体实现问题，需要受控复核。GPU校验通过不等于7B完整工作负载与Full的输出必须一致。")
    paragraph("优先增加一个共享prefill路径的Full控制组：与PhaseKV一样取末块attention分数，但保留dense KV不压缩，沿用同一输入、tokenizer和分块。先比较压缩前logits、首token及缓存，再比较prompt保留100%/8-bit的小样本。原生SDPA Full仍保留为实际系统性能基线，不能混淆两种基线用途。无需先重跑整轮46组。")
    paragraph("解决或量化prefill差异后，强制加入P70/B4、P100/B4、P100/B8等质量控制候选，W128/C4/D1先固定；增大筛选样本并按质量/速度/压缩多样性保留方案，避免最大QA损失一项筛掉全部高位宽配置。若重点是长输出质量，再补充真正评分的长输出任务。")
    paragraph("本轮候选在独立于筛选的文档上复核，但读完复核结果再次选优仍会产生选择偏差。这里的bootstrap区间按任务分层、按文档聚类，保留同一文档全部长度一起重采样；未做多重比较校正。最终结论只适用于这次Qwen与硬件设置，不能直接推到Llama或所有场景。")
    paragraph("附带candidate_balanced.json仅方便复现实验，不改变原始自动推荐为空的结论。原始sweep_result及此前算法对比报告均未修改。analysis.json包含重算指标、审计记录、配对区间及首token例子。")
    (OUT / "analysis.txt").write_text("\n\n".join(plain)+"\n",encoding="utf-8")
    (OUT / "report.html").write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhaseKV 参数搜索分析</title><style>body{font:16px system-ui,"Microsoft YaHei",sans-serif;line-height:1.75;color:#162d3b;background:#f7f9fc;margin:0}main{max-width:1280px;margin:30px auto;padding:30px;background:white}h1,h2{line-height:1.35}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}td,th{padding:9px;border:1px solid #d8e1e8;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf4f8}.scroll{overflow-x:auto}.warning{background:#fff1d6;padding:16px;border-left:4px solid #be7414}details{margin:20px 0}summary{cursor:pointer}</style></head><body><main>'+"\n".join(blocks)+'</main></body></html>',encoding="utf-8")
    print(json.dumps({"output":str(OUT),"audit":audit,"paired_contrasts":contrasts},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    build()
