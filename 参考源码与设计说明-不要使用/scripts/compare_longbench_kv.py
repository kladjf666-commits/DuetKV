"""Fresh Qwen comparison using the saved LongBench sample/configuration protocol.

Records every prediction and both actual KV tensor bytes and logical bytes.
Quantization methods use packed storage. Fake-quantization entry points are disabled.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts import run_longbench as lb
from eval import metrics, runtime
from eval.compressors import build_compressor
from eval.compression_trace import TRACE_VERSION


def forbid_fake_quantization():
    from baselines import _common
    from eval.compressors import common
    from src.phasekv import quant

    def forbidden(*args, **kwargs):
        raise RuntimeError("Fake quantization is forbidden in this benchmark")

    _common.grouped_fake_quant = forbidden
    common.fake_quantize = forbidden
    for name in ("_grouped_affine_quant_dequant", "quantize_key", "quantize_value", "quantize_kv"):
        setattr(quant, name, forbidden)


def generate(model, tokenizer, ids, spec, max_new_tokens):
    name = spec["method"]
    params = {k: v for k, v in spec.items() if k != "method"}
    if name in ("kivi", "phasekv"):
        from baselines.packed_runtime import generate_packed
        return generate_packed(model, tokenizer, ids, method=name, params=params,
                               max_new_tokens=max_new_tokens, trace_compression=True)
    compressor = build_compressor(name, **params)
    return runtime.generate_with_compression(model, tokenizer, ids, compressor,
                                              max_new_tokens=max_new_tokens, greedy=True, seed=1234,
                                              trace_compression=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def summarize(records, methods, tasks):
    refs = {r["sample_id"]: r for r in records if r["method"] == "full"}
    rows = []
    for method in methods:
        name = method["method"]
        selected = [r for r in records if r["method"] == name]
        if not selected:
            continue
        task_scores = {t: [r["qa_f1"] for r in selected if r["task"] == t] for t in tasks}
        task_means = {t: 100 * statistics.mean(v) for t, v in task_scores.items() if v}
        matches = [metrics.token_match_rate(r["generated_ids"], refs[r["sample_id"]]["generated_ids"])
                   for r in selected if r["sample_id"] in refs]
        mean = lambda key: statistics.mean(r[key] for r in selected)
        rows.append({
            "method": name, "params": {k: v for k, v in method.items() if k != "method"},
            "num_samples": len(selected), "qa_f1": statistics.mean(task_means.values()),
            "per_task": task_means,
            "logical_kv_saving_pct": mean("logical_kv_saving_pct"),
            "physical_kv_saving_pct": mean("physical_kv_saving_pct"),
            "mean_physical_kv_mib": mean("physical_kv_bytes") / 2**20,
            "peak_allocated_mib": max(r["peak_gpu_bytes"] for r in selected) / 2**20,
            "mean_wall_ms": mean("wall_ms"),
            "token_match_vs_full": statistics.mean(matches) if matches and name != "full" else None,
            "compression_triggers": {
                "observed_samples": sum("compression_trace" in r for r in selected),
                "missing_trace_samples": sum("compression_trace" not in r for r in selected),
                "triggered_samples": sum(r.get("compression_trace", {}).get("status") == "triggered" for r in selected),
                **{stage: {
                    "triggered_samples": sum(r.get("compression_trace", {}).get("summary", {}).get(stage, {}).get("triggered_steps", 0) > 0 for r in selected),
                    "triggered_steps": sum(r.get("compression_trace", {}).get("summary", {}).get(stage, {}).get("triggered_steps", 0) for r in selected),
                } for stage in ("prefill", "decode")},
            },
        })
    return rows


def save_report(prefix, meta, records, methods, tasks, complete):
    rows = summarize(records, methods, tasks)
    payload = {"meta": {**meta, "complete": complete}, "rows": rows}
    prefix.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Qwen2-1.5B · LongBench KV 对比", "",
             f"状态：{'完成' if complete else '进行中'}。模型：`{meta['model']}`。",
             f"每任务 {meta['n_samples_per_task']} 条；正文截断 {meta['max_length']} token；BF16 模型、eager、贪心生成。", "",
             "Full/驱逐方法的 KV 为 BF16；打包方法的 scale/min 和高精度残差为 FP16（均为 16-bit），沿用已有 packed kernel 精度。",
             "KIVI 和 PhaseKV 使用真实低比特打包；伪量化入口被设为立即报错。其他压缩方法使用 BF16 token 驱逐。",
             "KIVI 的 K 按通道分组，V 按 token 分组；采用官方独立 K/V 残差更新时机。PhaseKV 使用现有真实打包缓存。",
             "KV 节省以同一条生成轨迹已经写入缓存的 token 数为基准（prompt + generated - 1），修正旧表的一 token 偏差。",
             "峰值为 PyTorch allocated，包含模型及注意力中间量；KV 字节不含 SCOPE 的重要度等辅助状态。",
             "墙钟时间包含压缩维护和测量开销；不代表单独 attention kernel 的速度。", "",
             "| 方法 | 样本 | QA F1 | 逻辑 KV 节省% | 实际 KV 节省% | 平均 KV MiB | 峰值 MiB | 平均每题 ms | 对 full token 一致率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        match = "—" if r["token_match_vs_full"] is None else f"{r['token_match_vs_full']:.4f}"
        lines.append(f"| {r['method']} | {r['num_samples']} | {r['qa_f1']:.2f} | "
                     f"{r['logical_kv_saving_pct']:.2f} | {r['physical_kv_saving_pct']:.2f} | "
                     f"{r['mean_physical_kv_mib']:.2f} | {r['peak_allocated_mib']:.1f} | "
                     f"{r['mean_wall_ms']:.1f} | {match} |")
    lines.extend(["", "逐题预测、答案、生成 token 和字节数见同名 `.predictions.jsonl`。",
                  "SCOPE 当前明确为 ALLKV + Slide 的 GQA 适配：当前 query 评分，保护全部 prompt；未实现 Adaptive/Discontinuous 或压缩 prompt 插件。旧版累计评分实现不可称为官方 SCOPE。",
                  "SnapKV、SCOPE、PyramidKV 使用仓库中的 GQA 适配实现。PyramidKV 为保持平均预算采用线性插值取整，与官方实现的整数步长取整有所不同。",
                  "量化方法使用本项目的 Triton 内核，不是官方 KIVI CUDA 扩展的性能复现；内核和模型适配已单独通过 GPU 数值验证。",
                  "原 7B 结果不能用于推断本次 1.5B 的精度变化。重新取得同一数据集，并保存了文件及输入 token 哈希；旧实验没有数据哈希，无法证明下载版本逐字节相同。", ""])
    lines.extend(["## 实际压缩触发记录", "",
                  "逐题 compression_trace 记录实际存储状态变化；prefill 为第0步，decode 从第1次缓存追加计数。",
                  "触发步数按整次模型维护统计；token 数逐层累加，K/V量化分别统计，不乘head数。未采集与未触发分开标记。", "",
                  "| 方法 | 已采集/缺失 | 触发样本 | prefill触发步 | decode触发步 |",
                  "|---|---:|---:|---:|---:|"])
    for r in rows:
        t = r["compression_triggers"]
        lines.append(f"| {r['method']} | {t['observed_samples']}/{t['missing_trace_samples']} | {t['triggered_samples']} | {t['prefill']['triggered_steps']} | {t['decode']['triggered_steps']} |")
    prefix.with_suffix(".md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/m1_longbench_7b.json")
    parser.add_argument("--longbench-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results/qwen2_1p5b_longbench_kv_20261023")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-methods", help="Run a subset, retaining the full protocol for later resume")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run with access to the host GPU")
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.manual_seed(1234)
    torch.set_num_threads(4)
    forbid_fake_quantization()
    reference = json.loads(args.reference.read_text())
    tasks = reference["meta"]["tasks"]
    n_samples = args.n_samples or reference["meta"]["n_samples_per_task"]
    max_length = reference["meta"]["max_length_truncate"]
    methods = [{"method": r["method"], **r["params"]} for r in reference["rows"]]
    lb.LONGBENCH_DIR = str(args.longbench_dir)
    data_hashes = {t: hashlib.sha256((args.longbench_dir / "data" / f"{t}.jsonl").read_bytes()).hexdigest()
                   for t in tasks}
    protocol = {"model": args.model, "tasks": tasks, "n_samples_per_task": n_samples,
                "max_length": max_length, "methods": methods, "data_hashes": data_hashes,
                "dtype": "bfloat16", "attention": "eager", "greedy": True, "seed": 1234,
                "storage_protocol": "real_packed_v1_no_fake_quant",
                "trace_protocol": TRACE_VERSION,
                "scope_implementation": "scope_slide_current_query_v2"}
    protocol_hash = digest(protocol)
    prediction_path = args.out.with_suffix(".predictions.jsonl")
    records = []
    if prediction_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use a new --out or --resume: {prediction_path}")
        records = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
        if any(r["protocol_hash"] != protocol_hash for r in records):
            raise ValueError("Saved predictions use a different model/data/configuration")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = runtime.load_model_and_tokenizer(
        args.model, dtype="bfloat16", device="cuda", attn_implementation="eager")
    dims = lb._kv_dims(model.config)
    samples = []
    for task in tasks:
        data = lb.load_task(task, n_samples)
        if len(data) != n_samples:
            raise ValueError(f"Insufficient {task} samples")
        for index, sample in enumerate(data):
            ids = lb.build_input_ids(tokenizer, task, sample, max_length)
            samples.append({"task": task, "sample_id": f"{task}:{index}",
                            "input_ids": ids, "input_hash": digest(ids.tolist()),
                            "answers": sample["answers"], "max_new": lb.DATASET2MAXLEN[task]})
    meta = {**protocol, "protocol_hash": protocol_hash, "gpu": torch.cuda.get_device_name(),
            "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "kv_dims": dims,
            "input_hashes": {s["sample_id"]: s["input_hash"] for s in samples},
            "storage_modes": {"full": "bf16", "kivi": "packed uint8 codes + fp16 scales/mins/residuals",
                              "snapkv": "bf16 token eviction", "scope": "bf16 decode eviction",
                              "pyramidkv": "bf16 per-layer token eviction",
                              "phasekv": "token eviction + packed uint8 codes + fp16 hot cache"}}
    for record in records:
        if record["input_hash"] != meta["input_hashes"][record["sample_id"]]:
            raise ValueError("Tokenizer input differs from saved predictions")
    done = {(r["method"], r["sample_id"]) for r in records}
    print(f"GPU={meta['gpu']}; samples={len(samples)}; methods={len(methods)}", flush=True)
    with prediction_path.open("a", buffering=1) as handle:
        for spec in methods:
            name = spec["method"]
            if args.only_methods and name not in args.only_methods.split(","):
                continue
            pending = [s for s in samples if (name, s["sample_id"]) not in done]
            if not pending:
                continue
            gc.collect()
            torch.cuda.empty_cache()
            generate(model, tokenizer, pending[0]["input_ids"], spec, 4)
            for index, sample in enumerate(pending):
                torch.cuda.synchronize()
                start = time.perf_counter()
                gen = generate(model, tokenizer, sample["input_ids"], spec, sample["max_new"])
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - start) * 1000
                written_tokens = gen["prompt_len"] + len(gen["generated_ids"]) - 1
                reference_bytes = metrics.kv_cache_dtype_bytes(*dims, written_tokens)
                # This legacy field sums retained FP16 tensors, so it is physical KV here.
                physical = gen["full_cache_bytes"]
                logical = gen["final_cache_bytes"]
                record = {"protocol_hash": protocol_hash, "method": name,
                          "sample_id": sample["sample_id"], "task": sample["task"],
                          "input_hash": sample["input_hash"], "answers": sample["answers"],
                          "prediction": gen["text"].strip(), "generated_ids": gen["generated_ids"],
                          "prompt_tokens": gen["prompt_len"], "written_tokens": written_tokens,
                          "qa_f1": lb.best_qa_f1(gen["text"].strip(), sample["answers"]),
                          "physical_kv_bytes": physical, "logical_kv_bytes": logical,
                          "full_reference_bytes": reference_bytes,
                          "physical_kv_saving_pct": 100 * (1 - physical / reference_bytes),
                          "logical_kv_saving_pct": 100 * (1 - logical / reference_bytes),
                          "peak_gpu_bytes": gen["peak_gpu_bytes"], "wall_ms": wall_ms}
                record["compression_trace"] = gen["compression_trace"]
                if "storage_evidence" in gen:
                    record["storage_evidence"] = gen["storage_evidence"]
                if name == "full" and physical != reference_bytes:
                    raise AssertionError("Full KV accounting mismatch")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                if (index + 1) % 16 == 0 or index + 1 == len(pending):
                    print(f"{name}: {index+1}/{len(pending)} new samples", flush=True)
                    save_report(args.out, meta, records, methods, tasks, False)
            row = next(r for r in summarize(records, methods, tasks) if r["method"] == name)
            print(f"{name}: QA_F1={row['qa_f1']:.2f}; physical saving={row['physical_kv_saving_pct']:.2f}%", flush=True)
    complete = len(records) == len(samples) * len(methods)
    save_report(args.out, meta, records, methods, tasks, complete)
    print(f"Completed: {args.out}.json / .md / .predictions.jsonl", flush=True)


if __name__ == "__main__":
    main()
