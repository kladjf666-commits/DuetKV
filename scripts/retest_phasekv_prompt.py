"""Six forced prompt configurations, native and matched-prefill Full controls."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import html
import json
import math
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_phasekv as s
b = s.b
HERE = Path(__file__).resolve()
VERSION = "prompt-retest-v1-matched-logits"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configs():
    return [("Full-native", s.FULL, "native"), ("Full-matched", s.FULL, "matched")] + [
        (s.label(spec), spec, "matched") for spec in
        [s.spec(ratio=r, bits=bits, window=128, cold=4, keep=1.) for r in (.7, 1.) for bits in (2, 4, 8)]]


def fingerprint(logits):
    # Done after generate() stops its timers. BF16 is hashed as raw bytes.
    values = logits.detach().contiguous().cpu()
    import torch
    return b.digest({"dtype": str(values.dtype), "shape": list(values.shape),
                     "bytes": hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest()})


def verify_pair(key, record, reference):
    if record.get("status") != "ok":
        return
    if record.get("phase") == "capacity" and (reference is None or reference.get("status") != "ok"):
        return
    if reference is None:
        if record.get("phase") in ("quality", "performance", "needle"):
            raise ValueError(f"Missing matched baseline: {key}")
        return  # Capacity probes can differ when a memory boundary is reached.
    if reference["status"] != "ok" or record.get("input_hash") != reference.get("input_hash"):
        raise ValueError(f"Matched input/status differs: {key}")
    if not record.get("prefill_logits_sha256") or record["prefill_logits_sha256"] != reference.get("prefill_logits_sha256"):
        raise ValueError(f"Matched prefill logits differ BEFORE compression: {key}. Stop and investigate; do not relax this check.")
    if record["generated_ids"][0] != reference["generated_ids"][0]:
        raise ValueError(f"First token differs BEFORE compression: {key}")


@contextmanager
def worker_policy(policy, baseline_path=None):
    if policy not in ("native", "matched"):
        raise ValueError("Unknown prefill policy")
    original_prefill, original_generate, original_recorder = b.prefill_model, b.generate, b.Recorder
    captured = {}
    references = {r["case_id"]: r for r in b.read_records(baseline_path)} if baseline_path else None

    def prefill(model, ids, chunk_size, need_scores, observation=32):
        output = original_prefill(model, ids, chunk_size, policy == "matched", observation)
        # Keep a view of only the final logits, not the attention tensors or full output.
        captured["logits"] = output.logits[:, -1].detach()
        return output

    def generate(*args, **kwargs):
        captured.clear()
        try:
            result = original_generate(*args, **kwargs)
            result["prefill_policy"] = policy
            result["prefill_logits_sha256"] = fingerprint(captured["logits"])
            return result
        finally:
            captured.clear()

    class AuditedRecorder(original_recorder):
        def add(self, key, record):
            if references is not None:
                verify_pair(key, record, references.get(key))
            return super().add(key, record)

    b.prefill_model, b.generate, b.Recorder = prefill, generate, AuditedRecorder
    try:
        yield
    finally:
        captured.clear()
        b.prefill_model, b.generate, b.Recorder = original_prefill, original_generate, original_recorder


def settings(args):
    argv = ["--models", "qwen", "--methods", "full,phasekv", "--out", args.out, "--data-dir", args.data_dir,
            "--samples", str(args.samples), "--repeats", str(args.repeats), "--seed", str(args.seed)]
    if args.model_path:
        argv += ["--qwen", args.model_path]
    base = b.parse_args(argv)
    a = dict(vars(base), sample_offset=args.sample_offset,
             performance_shapes=[(1024,1024,"short"),(1024,4096,"long_output"),
                (16384,256,"long_input"),(16384,2048,"joint"),(24576,4096,"joint"),(32000,256,"long_input")],
             lengths=[1024,8192,16384,24576,32000])
    if args.quick:
        a.update(tasks=["qasper", "multifieldqa_en", "gov_report", "multi_news"],
                 quality_lengths=[3072,16384], phases=["performance", "quality"])
    return a, b.profiles_for(base)[0]


def report(out, manifest):
    plan = manifest["plan"]
    expected = set(manifest["expected"])
    records = {}
    completed = {}
    for name, spec, policy in plan:
        folder = out / "runs" / name
        if not (folder / "complete.json").exists():
            continue
        done, job = read(folder / "complete.json"), read(folder / "job.json")
        if done != {"hash": b.file_hash(folder / "records.jsonl"), "protocol": job["protocol_hash"]}:
            raise ValueError(f"Completed records changed: {name}")
        rr = b.read_records(folder / "records.jsonl")
        if any(r["protocol_hash"] != job["protocol_hash"] or r["parameters"] != spec for r in rr):
            raise ValueError(f"Record protocol mismatch: {name}")
        records[name], completed[name] = rr, {r["case_id"]: r for r in rr}
    rows = []
    for name, spec, policy in plan:
        if name not in records:
            continue
        row = {"name": name, "spec": spec, "policy": policy, "comparisons": {}}
        for baseline in ("Full-native", "Full-matched"):
            if baseline not in records:
                continue
            if name not in ("Full-native", "Full-matched") and baseline == "Full-matched":
                for r in records[name]:
                    verify_pair(r["case_id"], r, completed[baseline].get(r["case_id"]))
            result = s.metrics(records[name], records[baseline], expected, spec, manifest["settings"]["max_quality_drop"], intervals=True)
            row["comparisons"][baseline] = result
        rows.append(row)
    audit = {}
    if "Full-native" in records and "Full-matched" in records:
        for name in records:
            quality = [r for r in records[name] if r.get("phase") == "quality" and r["status"] == "ok"]
            audit[name] = {baseline: sum(r["generated_ids"][0] != completed[baseline][r["case_id"]]["generated_ids"][0] for r in quality)
                           for baseline in ("Full-native", "Full-matched")}
    b.atomic_json(out / "comparison.json", {"rows": rows, "first_token_differences": audit})
    blocks = ["<h1>PhaseKV prompt 六组复测</h1>",
        "<p>固定热区128、冷区4-bit、生成保留100%。六组全部测试，不筛掉高位宽候选。Full-native为实际SDPA流程；Full-matched与PhaseKV一样在prefill末块取attention分数，但保留dense KV。主要压缩质量比较使用Full-matched；实际系统速度同时参考Full-native。</p>",
        "<p>每条PhaseKV记录必须与Full-matched的输入、压缩前完整末位logits哈希、首token一致。不一致立即停止。哈希在计时结束后计算；审计保留的末位logits张量仍计入显存峰值，各组采用同样方式。</p>"]
    for baseline in ("Full-matched", "Full-native"):
        blocks.append(f"<h2>相对 {baseline}</h2><table><tr><th>配置</th><th>QA F1</th><th>摘要ROUGE-L</th><th>QA下降</th><th>摘要下降</th><th>速度比</th><th>KV比</th><th>门槛</th></tr>")
        for row in rows:
            r = row["comparisons"].get(baseline)
            if not r:
                continue
            cells = ([row["name"], f'{r["qa"]:.2f}', f'{r["summary"]:.2f}', f'{r["qa_loss"]:.2f}',
                      f'{r["summary_loss"]:.2f}', f'{r["speed"]:.3f}×', f'{100*r["kv"]:.2f}%', "通过" if r["feasible"] else "未通过"]
                     if r["valid"] else [row["name"], "缺失或失败记录，不参与推荐"])
            blocks.append("<tr>" + "".join("<td>"+html.escape(str(c))+"</td>" for c in cells) + "</tr>")
        blocks.append("</table>")
    blocks.append("<h2>首token差异数（质量样本）</h2><pre>" + html.escape(json.dumps(audit,ensure_ascii=False,indent=2)) + "</pre>")
    recommendations = {}
    if len(records) == len(plan):
        qualified = [r for r in rows if r["spec"]["method"] == "phasekv" and
                     r["comparisons"]["Full-matched"]["valid"] and r["comparisons"]["Full-matched"]["feasible"]]
        for mode, key in (("quality",lambda x:x["loss"]),("speed",lambda x:-x["speed"]),("compression",lambda x:x["kv"])):
            if qualified:
                selected = min(qualified,key=lambda r:key(r["comparisons"]["Full-matched"]))
                recommendations[mode] = selected["name"]
                b.atomic_json(out / ("qualified_"+mode+".json"), {"methods":[selected["spec"]]})
        blocks.append("<p>全部完成。相对Full-matched的达标候选："+html.escape(str(recommendations or "无"))+"。实际业务仍需检查相对Full-native的结果。</p>")
    else:
        blocks.append(f"<p>已完成 {len(records)}/{len(plan)} 组；全部结束前不导出推荐。</p>")
    b.atomic_json(out / "recommendations.json", recommendations)
    blocks.append("<p>均值门槛不代表每个任务达标或统计等效。QA和摘要分开按任务等权；速度及KV比值是固定形状的几何平均。详细区间按文档聚类bootstrap，未校正多重选择；本轮比较后仍应另留独立测试集。4096-token输出只测速，摘要质量生成上限512。各组原始记录、逐任务/长度/形状明细如下。</p>")
    for row in rows:
        blocks.append("<details><summary>"+html.escape(row["name"])+"</summary><pre>"+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+"</pre></details>")
    (out / "report.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PhaseKV六组复测</title><style>body{font:16px system-ui;line-height:1.65;max-width:1300px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #ccd5df}th{background:#edf4f8}pre{white-space:pre-wrap}details{margin:15px 0}</style>'+"\n".join(blocks)+"</html>",encoding="utf-8")
    return rows


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default=str(b.ROOT / "results/phasekv_prompt_retest"))
    p.add_argument("--data-dir",default=str(b.ROOT / ".cache/longbench"))
    p.add_argument("--model-path")
    p.add_argument("--samples",type=int,default=32)
    p.add_argument("--sample-offset",type=int,default=40,help="Default excludes previous screening 0..7 and verification 8..39")
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--max-quality-drop",type=float,default=1.)
    for flag in ("resume","dry-run","report-only","quick"):
        p.add_argument("--"+flag,action="store_true")
    a = p.parse_args(argv)
    if min(a.samples,a.repeats)<1 or a.sample_offset<0 or not math.isfinite(a.max_quality_drop) or a.max_quality_drop<0:
        p.error("Invalid sample/repeat/offset/quality settings")
    a.out,a.data_dir = str(Path(a.out).resolve()),str(Path(a.data_dir).resolve())
    return a


def validate_job(job, report_path):
    # The existing oracle explicitly requests scores for its semantic checks.
    # Native validation must preserve those requests; Full decode uses native prefill.
    with (worker_policy("matched") if job["prefill_policy"] == "matched" else nullcontext()):
        checks=b.gpu_validation(job["profiles"],report_path=report_path)
    b.atomic_json(report_path,{"status":"passed","prefill_policy":job["prefill_policy"],"checks":checks})


def main(argv=None):
    args = arguments(argv)
    out = Path(args.out)
    if args.report_only:
        report(out, read(out / "manifest.json"))
        return
    a, profile = settings(args)
    plan = configs()
    for name,spec,policy in plan:
        b.validate_spec(spec,a["chunk_size"])
    if args.dry_run:
        print(json.dumps(dict(groups=plan,quality_cases_per_group=len(a["tasks"])*len(a["quality_lengths"])*a["samples"],
                              performance_cases_per_group=len(a["performance_shapes"])*(a["repeats"]+1),settings=a),ensure_ascii=False,indent=2))
        return
    out.mkdir(parents=True,exist_ok=True)
    mf = out / "manifest.json"
    if mf.exists() and not args.resume:
        raise FileExistsError("Output exists; use identical settings with --resume, or choose a new --out")
    lock = out / "RUNNING.lock"
    with lock.open("x",encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        b.check_sources()
        runner = str(b.ROOT / "scripts/benchmark_4090d.py")
        s.execute([sys.executable,runner,"__check_environment__"],out / "environment.log")
        data = b.prepare_data(args.data_dir,a["tasks"])
        expected = s.expected_keys(a,args.data_dir)
        b.atomic_json(out / "identity_job.json",[profile])
        s.execute([sys.executable,runner,"__identity__",str(out / "identity_job.json"),str(out / "identity.json")],out / "identity.log")
        sources = b.source_hashes()
        sources.update({name:b.file_hash(b.ROOT / name) for name in ("scripts/sweep_phasekv.py","scripts/retest_phasekv_prompt.py","run_phasekv_prompt_retest.sh")})
        manifest = dict(version=VERSION,plan=plan,expected=sorted(expected),profile=profile,worker_settings=a,
            settings={k:v for k,v in vars(args).items() if k not in ("resume","dry_run","report_only")},
            data=data,source_hashes=sources,checkpoint=read(out / "identity.json"))
        manifest["hash"] = b.digest(manifest)
        if mf.exists() and read(mf)["hash"] != manifest["hash"]:
            raise ValueError("Code/data/model/settings changed; choose a new --out")
        b.atomic_json(mf,manifest)
        for i,(name,spec,policy) in enumerate(plan,1):
            print(f"\n=== retest: {i}/{len(plan)} {name} ===",flush=True)
            folder = out / "runs" / name
            folder.mkdir(parents=True,exist_ok=True)
            validation = folder / "validation.json"
            vp = folder / "validation_job.json"
            b.atomic_json(vp,{"profiles":[dict(profile,methods=[spec])],"prefill_policy":policy})
            if not validation.exists() or read(validation).get("status") != "passed":
                s.execute([sys.executable,"-u",str(HERE),"__validate__",str(vp),str(validation)],folder / "validation.log")
            rp, done, jp = folder / "records.jsonl",folder / "complete.json",folder / "job.json"
            job = dict(name=profile["name"],architecture=profile["architecture"],spec=spec,args=a,
                       model_path=manifest["checkpoint"][profile["name"]]["resolved_path"],records_path=str(rp),
                       protocol_hash=b.digest({"manifest":manifest["hash"],"name":name}),prefill_policy=policy,
                       matched_baseline=str(out / "runs/Full-matched/records.jsonl") if spec["method"]=="phasekv" else None)
            if done.exists():
                if read(done) != {"hash":b.file_hash(rp),"protocol":job["protocol_hash"]}:
                    raise ValueError("Completed records changed: "+name)
            else:
                b.atomic_json(jp,job)
                s.execute([sys.executable,"-u",str(HERE),"__worker__",str(jp)],folder / "run.log")
                b.atomic_json(done,{"hash":b.file_hash(rp),"protocol":job["protocol_hash"]})
            rows = report(out,manifest)
            own = next(r for r in rows if r["name"]==name)
            if spec["method"]=="full" and not own["comparisons"][name]["valid"]:
                raise RuntimeError("Baseline incomplete; cannot compare remaining candidates")
        b.atomic_json(out / "complete.json",{"groups":[x[0] for x in plan],"manifest_hash":manifest["hash"]})
        print(f"\nComplete: {out / 'report.html'}",flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        if len(sys.argv)>1 and sys.argv[1]=="__worker__":
            job=read(sys.argv[2])
            with worker_policy(job["prefill_policy"],job.get("matched_baseline")):
                b.worker(job)
        elif len(sys.argv)>1 and sys.argv[1]=="__validate__":
            validate_job(read(sys.argv[2]),sys.argv[3])
        else:
            main()
    except KeyboardInterrupt:
        print("Interrupted. Resume with the same options plus --resume.",file=sys.stderr)
        sys.exit(130)
