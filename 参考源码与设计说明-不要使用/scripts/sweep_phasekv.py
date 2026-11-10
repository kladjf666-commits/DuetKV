"""Sequential, resumable PhaseKV parameter search. Controller uses only stdlib."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
import math
import os
from pathlib import Path
import random
import statistics as st
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_4090d as b

VERSION = "phasekv-search-1"
FULL = {"method": "full"}
SCREEN_TASKS = ["qasper", "multifieldqa_en", "gov_report", "multi_news"]


def spec(ratio=.5, bits=4, window=64, cold=4, keep=1.):
    return dict(method="phasekv", prefill_keep_ratio=ratio, prompt_bits=bits,
                recent_window=window, cold_bits=cold, decode_keep_ratio=keep,
                obs_window=32, sink_tokens=0, lazy_g=16, group_size=32)


def label(s):
    if s["method"] == "full":
        return "Full"
    return f"P{s['prefill_keep_ratio']:g}-B{s['prompt_bits']}-W{s['recent_window']}-C{s['cold_bits']}-D{s['decode_keep_ratio']:g}"


def unique(specs):
    return list({label(s): s for s in specs}.values())


def prefill_grid():
    return unique([spec(.3, 2, 32)] + [spec(r, bits) for r in (.3, .5, .7, 1.) for bits in (2, 4, 8)])


def decode_grid(winners):
    # Representative interactions; deliberately not a claim of exhaustive search.
    choices = [(w, c, 1.) for w in (32, 64, 128) for c in (2, 4, 8)]
    choices += [(w, 4, k) for w in (64, 128) for k in (.5, .75)]
    return unique([dict(s, recent_window=w, cold_bits=c, decode_keep_ratio=k)
                   for s in winners for w, c, k in choices] + winners)


def settings(args, base, final=False):
    a = dict(vars(base))
    context = 32768 if args.model == "qwen" else 4096
    shapes = [(1024, 1024, "short"), (16384, 256, "long_input"), (16384, 2048, "joint")]
    if final:
        shapes += [(1024, 4096, "long_output"), (32000, 256, "long_input"), (24576, 4096, "joint")]
    if context == 4096:
        shapes = [(1024, 1024, "short"), (3072, 256, "long_input"), (1024, 3072, "long_output")]
    a.update(tasks=list(b.TASK_LIMITS) if final else SCREEN_TASKS,
             samples=args.final_samples if final else args.screen_samples,
             sample_offset=args.screen_samples if final else 0,
             quality_lengths=([1536, 3072, 8192, 16384] if final else [3072, 16384]) if context > 4096 else [1536, 3072],
             performance_shapes=shapes, repeats=args.final_repeats if final else args.screen_repeats,
             phases=["capacity", "needle", "performance", "quality"] if final else ["performance", "quality"],
             lengths=([1024, 8192, 16384, 24576, 32000] if context > 4096 else [1024, 3072]),
             max_context=context, needle_samples=3, seed=args.seed, resume=True)
    return a


def expected_keys(a, data_dir):
    keys = set()
    for task in a["tasks"]:
        count = len(b.load_rows(Path(data_dir) / "data" / (task + ".jsonl")))
        if a["sample_offset"] + a["samples"] > count:
            raise ValueError(f"{task}: not enough documents for disjoint screening/verification")
        ids = b.sample_indices(count, a["samples"], a["seed"], task, a["sample_offset"])
        keys.update(f"quality/{task}/{length}/{i}" for length in a["quality_lengths"] for i in ids)
    for n, m, group in a["performance_shapes"]:
        keys.update(f"performance/{group}/{n}/{m}/{r}" for r in range(1, a["repeats"] + 1))
    return keys


def geometric(xs):
    return math.exp(st.mean(math.log(x) for x in xs))


def paired_intervals(rows, refs, expected):
    """Resample documents within tasks; all context lengths of a document stay together."""
    tasks = defaultdict(lambda: defaultdict(list))
    for key in sorted(expected):
        r = rows[key]
        if r["phase"] == "quality":
            tasks[r["task"]][r["sample_index"]].append(r["score"] - refs[key]["score"])
    rng = random.Random(1234)
    result = {}
    for name, summary in (("qa", False), ("summary", True)):
        groups = [[st.mean(v) for v in docs.values()] for t, docs in tasks.items() if (t in b.SUMMARY_TASKS) == summary]
        estimates = sorted(st.mean(st.mean(rng.choices(g, k=len(g))) for g in groups) for _ in range(1000))
        result[name + "_delta_ci95"] = [estimates[24], estimates[974]]
    return result


def metrics(records, baseline, expected, s, loss_limit, intervals=False):
    """Reject missing/failed/mismatched pairs rather than averaging a favorable subset."""
    result = {"name": label(s), "spec": s, "valid": False, "feasible": False}
    rows = {r["case_id"]: r for r in records}
    refs = {r["case_id"]: r for r in baseline}
    if len(rows) != len(records) or len(refs) != len(baseline):
        raise ValueError("Duplicate records")
    bad = [k for k in sorted(expected) if k not in rows or k not in refs or
           rows[k]["status"] != "ok" or refs[k]["status"] != "ok"]
    if bad:
        return dict(result, reason="missing/failed required cases", bad_cases=bad)
    grouped, refgroup = defaultdict(list), defaultdict(list)
    bylength = defaultdict(list)
    byshape = defaultdict(list)
    for key in sorted(expected):
        r, f = rows[key], refs[key]
        if not r.get("input_hash") or r["input_hash"] != f.get("input_hash") or r.get("answers") != f.get("answers"):
            raise ValueError(f"Input/answer mismatch: {label(s)} {key}")
        if r["phase"] == "quality":
            grouped[r["task"]].append(r["score"])
            if any(not isinstance(x.get("score"), (int, float)) or not math.isfinite(x["score"]) for x in (r, f)):
                raise ValueError(f"Invalid quality score: {key}")
            refgroup[r["task"]].append(f["score"])
            bylength[(r["task"], r["length_budget"])].append((r["score"], f["score"]))
        else:
            if r["output_tokens"] != f["output_tokens"] or r["output_tokens"] != r["requested_output_tokens"]:
                raise ValueError("Performance generation did not reach the fixed output length")
            if any(not math.isfinite(x[field]) or x[field] <= 0 for x in (r, f)
                   for field in ("decode_tokens_per_s", "final_kv_bytes", "total_s", "peak_allocated_bytes")):
                raise ValueError(f"Invalid performance measurement: {key}")
            byshape[(r["input_tokens"], r["output_tokens"], r["sweep"])].append((r, f))
    tasks = {t: {"score": st.mean(v), "full": st.mean(refgroup[t]), "n": len(v)} for t, v in grouped.items()}
    group_scores = {}
    for name, summary in (("qa", False), ("summary", True)):
        ts = [t for t in tasks if (t in b.SUMMARY_TASKS) == summary]
        group_scores[name] = st.mean(tasks[t]["score"] for t in ts)
        group_scores[name + "_loss"] = st.mean(tasks[t]["full"] - tasks[t]["score"] for t in ts)
    shapes = []
    for (n, m, group), pairs in sorted(byshape.items()):
        speed = st.median(r["decode_tokens_per_s"] for r, f in pairs)
        refspeed = st.median(f["decode_tokens_per_s"] for r, f in pairs)
        kv = st.median(r["final_kv_bytes"] for r, f in pairs)
        refkv = st.median(f["final_kv_bytes"] for r, f in pairs)
        shapes.append(dict(input=n, output=m, group=group, tokens_s=speed,
                           speed_vs_full=speed / refspeed, kv_ratio=kv / refkv, kv_mib=kv / 2**20,
                           prefill_kv_mib=st.median(r["prefill_kv_bytes"] for r, f in pairs) / 2**20,
                           total_s=st.median(r["total_s"] for r, f in pairs),
                           prefill_s=st.median(r["prefill_and_compress_s"] for r, f in pairs),
                           peak_gib=max(r["peak_allocated_bytes"] for r, f in pairs) / 2**30))
    loss = max(group_scores["qa_loss"], group_scores["summary_loss"])
    needles = [r for k, r in rows.items() if r.get("phase") == "needle" and r["status"] == "ok"
               and k in refs and refs[k]["status"] == "ok" and r.get("input_hash") == refs[k].get("input_hash")]
    extra = paired_intervals(rows, refs, expected) if intervals else {}
    return dict(result, valid=True, feasible=loss <= loss_limit, **group_scores, **extra,
                loss=loss, speed=geometric([x["speed_vs_full"] for x in shapes]),
                kv=geometric([x["kv_ratio"] for x in shapes]),
                peak_gib=max(x["peak_gib"] for x in shapes), tasks=tasks, shapes=shapes,
                quality_by_length=[dict(task=t, length=n, score=st.mean(x for x, y in v), full=st.mean(y for x, y in v), n=len(v)) for (t, n), v in sorted(bylength.items())],
                needle=dict(paired_n=len(needles), score=st.mean(r["score"] for r in needles) if needles else None,
                            full=st.mean(refs[r["case_id"]]["score"] for r in needles) if needles else None),
                failed_cases=[r["case_id"] for r in records if r["status"] != "ok" and r.get("phase") != "capacity"])


def pareto(rows):
    def dominates(a, z):
        av, zv = (a["loss"], -a["speed"], a["kv"]), (z["loss"], -z["speed"], z["kv"])
        return all(x <= y for x, y in zip(av, zv)) and any(x < y for x, y in zip(av, zv))
    return [r for r in rows if not any(dominates(other, r) for other in rows if other is not r)]


def select(rows, count):
    valid = [r for r in rows if r["valid"] and r["spec"]["method"] != "full"]
    if not valid:
        raise RuntimeError("No complete candidate. Inspect logs and failed cases; do not bypass validation.")
    feasible = [r for r in valid if r["feasible"]]
    if not feasible:
        # Still permit an honest final check of the least damaging configurations.
        pool = sorted(valid, key=lambda r: (r["loss"], -r["speed"], r["kv"]))
        return pool[:count]
    front = pareto(feasible)
    chosen = []
    for key in (lambda r: (r["loss"], -r["speed"]), lambda r: (-r["speed"], r["kv"]), lambda r: (r["kv"], r["loss"])):
        best = min(front, key=key)
        if best not in chosen:
            chosen.append(best)
    chosen += [r for r in sorted(feasible, key=lambda r: (r["loss"], -r["speed"])) if r not in chosen]
    return chosen[:count]


def execute(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        try:
            for line in p.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if p.wait():
                raise RuntimeError(f"Subprocess failed. Inspect {log_path}")
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


def report(out, stages, limit):
    b.atomic_json(out / "comparison.json", stages)
    e = html.escape
    blocks = ["<h1>PhaseKV 参数搜索</h1>",
        f"<p>质量门槛：QA F1 与摘要 ROUGE-L 的分组任务均值，各自相对 Full 下降不超过 {limit:g} 分（0–100）。"
        "速度与 KV 比值为固定形状的等权几何平均；详细表列出实际值。热区为 FP16。低 KV 不等于低总显存。</p>",
        "<p>prefill 与 decode 共用筛选文档；final 使用另一批文档。分阶段搜索并非全组合穷举，筛选排名不能当作最终结论。"
        "门槛按样本均值判断，不表示统计显著或对所有任务都达标。</p>"]
    for stage, rows in stages.items():
        blocks += [f"<h2>{e(stage)}</h2><table><tr><th>配置</th><th>QA F1</th><th>摘要 ROUGE-L</th><th>最大分组下降</th><th>速度 / Full</th><th>KV / Full</th><th>峰值 GiB</th><th>门槛</th></tr>"]
        for r in rows:
            cells = ([r["name"], f'{r["qa"]:.2f}', f'{r["summary"]:.2f}', f'{r["loss"]:.2f}',
                      f'{r["speed"]:.2f}×', f'{100*r["kv"]:.1f}%', f'{r["peak_gib"]:.2f}', "通过" if r["feasible"] else "未通过"]
                     if r["valid"] else [r["name"], "不可比较：" + r.get("reason", "incomplete")])
            blocks.append("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in cells) + "</tr>")
        blocks.append("</table>")
        for r in rows:
            if not r["valid"]:
                continue
            blocks.append(f'<details><summary>{e(r["name"])}：任务与速度明细</summary><pre>' +
                          e(json.dumps({k: r[k] for k in ("spec", "tasks", "quality_by_length", "shapes", "needle", "qa_delta_ci95", "summary_delta_ci95", "failed_cases") if k in r}, ensure_ascii=False, indent=2)) + "</pre></details>")
    final = stages.get("final", [])
    marker = out / "final" / "stage_complete.json"
    complete = marker.exists() and json.loads(marker.read_text(encoding="utf-8")) == sorted(r["name"] for r in final)
    feasible = [r for r in final if complete and r["valid"] and r["feasible"] and r["spec"]["method"] == "phasekv"]
    recs = {}
    if feasible:
        for name, key in (("quality", lambda r: r["loss"]), ("speed", lambda r: -r["speed"]), ("compression", lambda r: r["kv"])):
            r = min(feasible, key=key)
            recs[name] = r["name"]
            b.atomic_json(out / f"recommended_{name}.json", {"methods": [r["spec"]]})
        blocks.append("<h2>满足门槛的复核候选</h2><p>" + e(str(recs)) + "</p>")
    elif final and complete:
        blocks.append("<h2>复核结果：没有 PhaseKV 候选满足质量门槛</h2><p>本轮不推荐部署配置，请检查任务明细并扩大搜索或提高保留比例。</p>")
    else:
        blocks.append("<p>复核尚未全部完成，暂不导出推荐配置。</p>")
    b.atomic_json(out / "recommendations.json", recs)
    blocks.append("<p>P=prompt 保留比例，B=prompt bit，W=热区窗口，C=冷区 bit，D=冷却批次保留比例。"
                  "固定 obs_window=32、lazy_g=16、group_size=32、sink_tokens=0。原始日志、配置与结果保存在各阶段目录。</p>")
    (out / "report.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PhaseKV 参数搜索</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:36px auto;padding:20px;color:#172b3a}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd5df}th{background:#e9f2f6}pre{white-space:pre-wrap}details{margin:12px 0}p{line-height:1.7}</style>' + "\n".join(blocks) + '</html>', encoding="utf-8")


def run_stage(out, name, specs, a, profile, manifest, stages, limit):
    stage = out / name
    stage.mkdir(exist_ok=True)
    expected = expected_keys(a, a["data_dir"])
    stage_plan = {"specs": specs, "settings": a, "expected": sorted(expected)}
    plan_file = stage / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text(encoding="utf-8")) != json.loads(json.dumps(stage_plan)):
        raise ValueError("Stage plan changed. Use a new output directory.")
    b.atomic_json(plan_file, stage_plan)
    rows, baseline = [], None
    stages[name] = rows
    for i, s in enumerate([FULL] + unique(specs)):
        slug = label(s)
        folder = stage / slug
        folder.mkdir(exist_ok=True)
        print(f"\n=== {name}: {i+1}/{len(specs)+1} {slug} ===", flush=True)
        validation_key = b.digest({"spec": s, "manifest": manifest["hash"]})
        vf = out / "validation" / validation_key
        vf.mkdir(parents=True, exist_ok=True)
        vp, vr = vf / "job.json", vf / "report.json"
        if not vr.exists() or json.loads(vr.read_text(encoding="utf-8")).get("status") != "passed":
            b.atomic_json(vp, [dict(profile, methods=[s])])
            execute([sys.executable, "-u", str(b.ROOT / "scripts/benchmark_4090d.py"), "__validate__", str(vp), str(vr)], vf / "validation.log")
        rp, done = folder / "records.jsonl", folder / "complete.json"
        job = dict(name=profile["name"], model_path=manifest["checkpoint"][profile["name"]]["resolved_path"],
                   architecture=profile["architecture"], spec=s, args=a,
                   protocol_hash=b.digest({"manifest": manifest["hash"], "stage": stage_plan, "spec": s}), records_path=str(rp))
        if done.exists():
            if json.loads(done.read_text(encoding="utf-8")) != {"hash": b.file_hash(rp), "protocol": job["protocol_hash"]}:
                raise ValueError(f"Completed records modified: {rp}")
        else:
            jp = folder / "job.json"
            b.atomic_json(jp, job)
            execute([sys.executable, "-u", str(b.ROOT / "scripts/benchmark_4090d.py"), "__worker__", str(jp)], folder / "run.log")
            b.atomic_json(done, {"hash": b.file_hash(rp), "protocol": job["protocol_hash"]})
        records = b.read_records(rp)
        if baseline is None:
            baseline = records
        row = metrics(records, baseline, expected, s, limit, intervals=name == "final")
        rows.append(row)
        report(out, stages, limit)
        if s == FULL and not row["valid"]:
            raise RuntimeError("Full baseline incomplete. Cannot compare this stage.")
    b.atomic_json(stage / "stage_complete.json", sorted(r["name"] for r in rows))
    report(out, stages, limit)
    return rows


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["qwen", "llama"], default="qwen")
    p.add_argument("--model-path", help="Optional path to original local model checkpoint")
    p.add_argument("--out", default=str(b.ROOT / "results/phasekv_parameter_sweep"))
    p.add_argument("--data-dir", default=str(b.ROOT / ".cache/longbench"))
    for name, default in (("screen-samples", 8), ("final-samples", 32), ("screen-repeats", 1), ("final-repeats", 3), ("top-prefill", 2), ("top-final", 3), ("seed", 1234)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--max-quality-drop", type=float, default=1.)
    p.add_argument("--until", choices=["prefill", "decode", "final"], default="final")
    for flag in ("dry-run", "resume", "report-only"):
        p.add_argument("--" + flag, action="store_true")
    a = p.parse_args(argv)
    if min(a.screen_samples, a.final_samples, a.screen_repeats, a.final_repeats, a.top_prefill, a.top_final) < 1 or not math.isfinite(a.max_quality_drop) or a.max_quality_drop < 0:
        p.error("Counts must be positive; quality drop must be finite and nonnegative")
    a.out, a.data_dir = str(Path(a.out).resolve()), str(Path(a.data_dir).resolve())
    return a


def main(argv=None):
    args = arguments(argv)
    out = Path(args.out)
    if args.report_only:
        old = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        report(out, json.loads((out / "comparison.json").read_text(encoding="utf-8")), old["settings"]["max_quality_drop"])
        return
    base_argv = ["--models", args.model, "--methods", "full,phasekv", "--out", args.out, "--data-dir", args.data_dir]
    if args.model_path:
        base_argv += ["--" + args.model, args.model_path]
    base = b.parse_args(base_argv)
    profile = b.profiles_for(base)[0]
    grid = prefill_grid()
    screen, final = settings(args, base), settings(args, base, True)
    for s in grid + decode_grid([spec()]):
        b.validate_spec(s, base.chunk_size)
    plan = dict(prefill_configs=len(grid), decode_configs_at_most=13 * args.top_prefill,
                final_configs_at_most=args.top_final + 1, baseline="Full rerun once per stage; current PhaseKV configuration included in final verification",
                screening_quality_cases_per_config=len(screen["tasks"]) * len(screen["quality_lengths"]) * args.screen_samples,
                final_quality_cases_per_config=len(final["tasks"]) * len(final["quality_lengths"]) * args.final_samples,
                screening=screen, final=final, prefill_grid=grid,
                note="Staged search, not exhaustive. Final documents exclude the screening prefix. Includes numerical validation, fixed-length generation and warmup.")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    if mf.exists() and not args.resume:
        raise FileExistsError("Output exists: add --resume with identical settings, or choose a new --out")
    lock = out / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        b.atomic_json(out / "search_plan.json", plan)
        runner = str(b.ROOT / "scripts/benchmark_4090d.py")
        execute([sys.executable, runner, "__check_environment__"], out / "environment.log")
        b.check_sources()
        data = b.prepare_data(args.data_dir, list(b.TASK_LIMITS))
        expected_keys(final, args.data_dir)
        identity_job, identity = out / "identity_job.json", out / "identity.json"
        b.atomic_json(identity_job, [profile])
        execute([sys.executable, runner, "__identity__", str(identity_job), str(identity)], out / "identity.log")
        sources = b.source_hashes()
        sources.update({name: b.file_hash(b.ROOT / name) for name in ("scripts/sweep_phasekv.py", "run_phasekv_sweep.sh")})
        manifest = dict(version=VERSION, settings={k: v for k, v in vars(args).items() if k not in ("resume", "dry_run", "report_only", "until")},
                        profile=profile, data=data, source_hashes=sources, checkpoint=json.loads(identity.read_text(encoding="utf-8")))
        manifest["hash"] = b.digest(manifest)
        if mf.exists() and json.loads(mf.read_text(encoding="utf-8"))["hash"] != manifest["hash"]:
            raise ValueError("Code/data/checkpoint/settings changed. Use a new --out; do not mix results.")
        b.atomic_json(mf, manifest)
        stages = json.loads((out / "comparison.json").read_text(encoding="utf-8")) if (out / "comparison.json").exists() else {}
        rows = run_stage(out, "prefill", grid, screen, profile, manifest, stages, args.max_quality_drop)
        winners = select(rows, args.top_prefill)
        b.atomic_json(out / "selected_prefill.json", winners)
        if args.until == "prefill":
            return
        rows = run_stage(out, "decode", decode_grid([r["spec"] for r in winners]), screen, profile, manifest, stages, args.max_quality_drop)
        winners = select(rows, args.top_final)
        b.atomic_json(out / "selected_decode.json", winners)
        if args.until == "decode":
            return
        run_stage(out, "final", unique([spec(.3, 2, 32)] + [r["spec"] for r in winners]), final, profile, manifest, stages, args.max_quality_drop)
        print(f"\nComplete: {out / 'report.html'}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Resume with the same options plus --resume.", file=sys.stderr)
        sys.exit(130)
