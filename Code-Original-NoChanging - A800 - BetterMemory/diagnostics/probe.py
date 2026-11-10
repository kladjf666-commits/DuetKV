"""A800 batch-3 decode diagnostic. All in-model timings exclude prefill."""
import argparse
import contextlib
import gzip
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import Engine, run as formal_run
from PhaseKV.phasekv_cache import PhaseKVCache
from PhaseKV.kernels import segmented_attention as seg
from transformers.cache_utils import DynamicCache
import ablation_kernels
import flash_attn

LAYERS = (0, 10, 20, 31)
PHASE = "phasekv"
FULL = "full"
median = statistics.median


def save(out, report):
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(out)


def reuse_failed_run(out, report):
    """Reuse only the v1.1 failure after both clean passes and the Full profile."""
    keys = ("gpu", "cc", "sm_count", "torch", "model", "batch", "prompt", "windows",
            "core_sha256", "input_sha256", "scope")
    expected = json.loads(json.dumps({k: report[k] for k in keys}))
    windows = {f"{lo}-{hi}" for lo, hi in report["windows"]}
    for path in sorted(out.parent.parent.glob("*/summary.json"), reverse=True):
        if path == out:
            continue
        log = path.with_name("run.log")
        if not log.is_file() or "ValueError: This benchmark packs all prompt positions" not in log.read_text(errors="replace"):
            continue
        prior = json.loads(path.read_text())
        if prior.get("status") != "running" or any(prior.get(k) != v for k, v in expected.items()):
            continue
        methods = prior.get("methods", {})
        full, phase = methods.get(FULL, {}), methods.get(PHASE, {})
        if (set(full.get("clean", {})) == set(full.get("instrumented", {})) == windows
                and "formal_minus_legacy_ms_per_step" in full.get("same_gpu_driver_ab", {})
                and set(phase) == {"clean"} and set(phase["clean"]) == windows):
            report.update(methods={FULL: full, PHASE: phase}, resumed_from=str(path))
            print(f"Reused Full and PhaseKV clean measurements from {path}", flush=True)
            return
    print("No matching failed run found; starting a fresh measurement.", flush=True)


def compare_drivers(engine, method, rows):
    # Only the historical loop changes; weights, cache, kernel, inputs and A800 are shared.
    sys.path.insert(0, str(HERE / "legacy"))
    from batch_speed import measure
    output = 281
    sample = dict(rows=rows, length_budget=8192, max_new_tokens=output)
    arm = "phasekv_triton" if method == PHASE else "full_fa2"
    # Engine.start resolves speed vs. quality on a copy of the base configuration.
    options = dict(engine.effective)
    if method == PHASE and (options["prefill_keep_ratio"], options["decode_keep_ratio"]) != (1.0, 0.8):
        raise ValueError(f"Unexpected effective PhaseKV speed options: {options}")
    results = {}
    for driver in (("formal", "legacy") if method == FULL else ("legacy", "formal")):
        if driver == "legacy":
            r = measure(engine.model, sample, 3, arm, options, {})
            hashes = r["output_sha256_by_row"]
        else:
            r = formal_run(engine, rows, output, [], False, {})
            if r["effective_options"] != options:
                raise RuntimeError(f"{method}: effective options changed between drivers")
            hashes = [hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in r["output_ids"]]
        results[driver] = dict(ms_per_step=1000*r["decode_s"]/r["decode_steps"],
                               kv_bytes=r["final_kv_bytes"], output_sha256=hashes)
        engine.release()
    a,b = results["formal"],results["legacy"]
    if a["output_sha256"] != b["output_sha256"] or a["kv_bytes"] != b["kv_bytes"]:
        raise RuntimeError(f"{method}: formal and legacy drivers produced different outputs or cache sizes: {results}")
    results["formal_minus_legacy_ms_per_step"] = a["ms_per_step"] - b["ms_per_step"]
    results["effective_options"] = options
    return results


class LaunchHook:
    """Intercept the already-selected Triton launcher, preserving its grid and kwargs."""
    def __init__(self, original, decorate):
        self.original, self.decorate = original, decorate

    def __getitem__(self, grid):
        return self.decorate(self.original[grid])


class Timings:
    """Disjoint current-stream envelopes and host call times, zero in-scope synchronizations."""
    def __init__(self, steps):
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(steps * (4 * 32 + 1) * 2)]
        for event in self.events:
            event.record()
        torch.cuda.synchronize()
        self.reset()

    def reset(self):
        self.records = []
        self.index = 0
        self.step = 0

    def wrap(self, target, label):
        def measured(*args, **kwargs):
            start, end = self.events[self.index:self.index+2]
            self.index += 2
            start.record()
            cpu = time.thread_time_ns()
            value = target(*args, **kwargs)
            cpu = (time.thread_time_ns() - cpu) / 1e6
            end.record()
            self.records.append((self.step, label, start, end, cpu))
            return value
        return measured

    @contextlib.contextmanager
    def install(self, method):
        with contextlib.ExitStack() as stack:
            def add(owner, attr, label):
                orig = getattr(owner, attr)
                stack.enter_context(patch.object(owner, attr, self.wrap(orig, label)))
            if method == PHASE:
                for attr, label in (("_segmented_attention", "packed_main"), ("_merge_segments", "merge")):
                    hook = LaunchHook(getattr(seg, attr), lambda fn, label=label: self.wrap(fn, label))
                    stack.enter_context(patch.object(seg, attr, hook))
                add(seg, "ensure_workspace", "descriptors")
                add(PhaseKVCache, "append_hot", "append_hot")
                add(PhaseKVCache, "finish_decode_step", "finish_cache")
            else:
                add(DynamicCache, "update", "dense_cache_update")
                add(flash_attn, "flash_attn_func", "fa2")
            yield

    def result(self, count, wall_ms, method):
        rows = defaultdict(lambda: defaultdict(lambda: [0., 0., 0]))
        for step, label, start, end, cpu in self.records:
            row = rows[step][label]
            row[0] += start.elapsed_time(end)
            row[1] += cpu
            row[2] += 1
        expected = ({"packed_main":32, "merge":32, "descriptors":32, "append_hot":32, "finish_cache":1}
                    if method == PHASE else {"dense_cache_update":32, "fa2":32})
        if len(rows) != count or any({k:v[2] for k,v in r.items()} != expected for r in rows.values()):
            raise RuntimeError("Unexpected launch counts: instrumentation missed the active path")
        totals = {k: dict(stream_ms_per_step=sum(v[k][0] for v in rows.values()) / count,
                          host_thread_ms_per_step=sum(v[k][1] for v in rows.values()) / count)
                  for k in expected}
        return dict(stages=totals, per_step=rows,
                    other_stream_or_host_gap_ms=wall_ms-sum(v["stream_ms_per_step"] for v in totals.values()))


def experiment(engine, method, ids, windows, tape=None, timer=None, variant=None, capture=False):
    engine.name = method
    for layer in engine.model.model.layers:
        layer.self_attn.method = "phasekv_triton" if method == PHASE else "full_fa2"
    logits = engine.start(ids, 2048, False, {})
    token = logits[:, -1].argmax(-1, keepdim=True)
    generated = [token]
    reports, sample = {}, []
    front = seg.segmented_phasekv_decode
    def tuned(q, tiers, heads):
        return front(q, tiers, heads, block_n=variant["block_n"], num_warps=4,
                     fuse_packed_fma=variant["fuse_packed_fma"])
    def grab(q, tiers, heads):
        i = len(sample_calls)
        sample_calls.append(i)
        if i in LAYERS:
            sample.append((q.detach().clone(), tiers))
        return front(q, tiers, heads)
    sample_calls = []
    first_steps = {lo for lo, _ in windows}
    last_steps = {hi for _, hi in windows}
    with contextlib.ExitStack() as stack:
        if variant:
            stack.enter_context(patch.object(ablation_kernels, "segmented_phasekv_decode", tuned))
        for step in range(1, windows[-1][1]+1):
            if step in first_steps:
                lo = step
                active = contextlib.ExitStack()
                stack.enter_context(active)
                if timer:
                    timer.reset()
                    active.enter_context(timer.install(method))
                torch.cuda.synchronize()
                start = time.perf_counter()
            if timer:
                timer.step = step
            # Snapshot on a separate untimed step, never charge the extra clones to decode.
            source = tape[step-1] if tape is not None else token
            logits = engine.step(source)
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
            if step in last_steps:
                torch.cuda.synchronize()
                ms = (time.perf_counter()-start)*1000/(step-lo+1)
                active.close()
                key = f"{lo}-{step}"
                reports[key] = dict(ms_per_step=ms, aggregate_tokens_s=len(ids)*1000/ms)
                if timer:
                    reports[key].update(timer.result(step-lo+1, ms, method))
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite logits")
        if capture:
            with patch.object(ablation_kernels, "segmented_phasekv_decode", grab):
                engine.step(tape[-1] if tape is not None else token)
            if len(sample_calls)!=32 or len(sample)!=len(LAYERS):
                raise RuntimeError("Did not capture the four real layer inputs")
    matches = torch.equal(torch.cat(generated,1),torch.cat(tape,1)) if tape is not None else True
    engine.release()
    return reports, generated, matches, sample


def graph_of(fn):
    # All buffers and workspaces have already been built. Hold callable/tensors alive.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def graph_measure(graphs):
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in graphs]
    samples = {name: [] for name in graphs}
    names = list(graphs)
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    for rep in range(5):
        for j in range(len(names)):
            name = names[(j+rep)%len(names)]
            start, end = events[j]
            start.record()
            for _ in range(16):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end)/(16*len(LAYERS)))
    return {name: dict(median_ms_layer=median(vals), samples_ms_layer=vals)
            for name, vals in samples.items()}


def replay(sample, *, prompt_test=False):
    from triton.runtime.errors import OutOfResources
    dense = []
    for q, t in sample:
        k, v = ablation_kernels.materialize(t)
        def kv(x):
            return x.reshape(3, 8, t.physical_len, 128).transpose(1,2)
        dense.append((q.reshape(3,1,32,128),kv(k),kv(v)))
    def fa(args):
        return flash_attn.flash_attn_func(*args, dropout_p=0., causal=True)
    refs = [fa(x).reshape(96,128) for x in dense]
    graphs = {"fa2_same_dequantized_kv": graph_of(lambda: [fa(x) for x in dense])}
    info = {}
    owners = []  # Keep every captured input, partial and output live across graph replay.
    configs = [("tile8",None,False), ("prompt_mma",None,False)] if prompt_test else [
        ("tile8",None,False), ("tile8_fma",8,True), ("tile16",16,False), ("tile32",32,False)]
    for tag, tile, fuse in configs:
        first, second = [], []
        def capture(pool):
            def wrap(fn):
                def launch(*args, **kwargs):
                    pool.append(lambda: fn(*args, **kwargs))
                    return fn(*args, **kwargs)
                return launch
            return wrap
        try:
            with contextlib.ExitStack() as hooks:
                main = "_a800_segmented_attention" if tag == "prompt_mma" else "_segmented_attention"
                hooks.enter_context(patch.object(seg, main, LaunchHook(getattr(seg, main), capture(first))))
                hooks.enter_context(patch.object(seg,"_merge_segments",LaunchHook(seg._merge_segments,capture(second))))
                if tag == "prompt_mma":
                    for _, tiers in sample:
                        hooks.enter_context(patch.object(tiers, "optimizations", replace(tiers.optimizations, a800_prompt_mma=True)))
                outs = [seg.segmented_phasekv_decode(q,t,96,block_n=tile,num_warps=4 if tile else None,
                                                    fuse_packed_fma=fuse)
                        for q,t in sample]
        except OutOfResources as exc:
            if tag == "tile8":
                raise
            info[tag] = dict(passed=False,error=str(exc))
            continue
        if len(first)!=4 or len(second)!=4:
            raise RuntimeError("Unexpected single-layer launch pattern")
        owners.extend((first,second,outs))
        err = max(((y.float()-ref.float()).norm(dim=-1)/ref.float().norm(dim=-1).clamp_min(1e-6)).max().item()
                  for y,ref in zip(outs,refs))
        same = max((y.float()-ref.float()).abs().max().item() for y,ref in zip(outs,refs))
        if not all(torch.isfinite(x).all().item() for x in outs+refs):
            raise ValueError("Nonfinite layer replay")
        info[tag] = dict(block_n=tile or 8, num_warps=4, fuse_packed_fma=fuse,
                         prompt_splits=sample[0][1]._decode_workspace.prefix_count,
                         max_head_relative_l2=err,max_abs=same,passed=err<=.005)
        if tag == "tile8" and not info[tag]["passed"]:
            raise ValueError(f"Baseline packed output disagrees with materialized FA2: {err}")
        for name, ops in [("main",first),("merge",second),("pair",[f for pair in zip(first,second) for f in pair])]:
            graphs[f"{tag}_{name}"] = graph_of(lambda: [f() for f in ops])
    stats = graph_measure(graphs)
    best = min((tag for tag in info if info[tag]["passed"]),key=lambda tag:stats[tag+"_pair"]["median_ms_layer"])
    return dict(configs=info,timing=stats,winner=best,
                cold_segments=len(sample[0][1].cold),hot_tokens=sample[0][1].hot_len,
                layer_ids=LAYERS, retained_tokens=sample[0][1].physical_len)


def test_prompt_mma(engine, ids, windows, report, out):
    base_options = dict(engine.options)
    def select(enabled):
        engine.options = dict(base_options, optimizations=dict(base_options["optimizations"], a800_prompt_mma=enabled))

    select(False)
    print("PhaseKV: capturing four layers and baseline greedy tokens", flush=True)
    _, tape, _, sample = experiment(engine, PHASE, ids, windows, capture=True)
    micro = report["layer_replay"] = replay(sample, prompt_test=True)
    sample.clear()
    save(out, report)
    for name in ("tile8", "prompt_mma"):
        config = micro["configs"][name]
        timing = micro["timing"].get(name + "_pair", {})
        print(name, "ms/layer", timing.get("median_ms_layer"), "numerical check", config, flush=True)
    if not micro["configs"]["prompt_mma"]["passed"]:
        report["status"] = "layer_check_failed"
        save(out, report)
        raise SystemExit("Candidate failed its layer check. See " + str(out))

    engine.options = {}
    _, full_tape, _, _ = experiment(engine, FULL, ids, windows)
    full, _, same, _ = experiment(engine, FULL, ids, windows, full_tape)
    if not same:
        raise ValueError("Full greedy replay mismatch")
    report.update(full=full, runs=[])
    save(out, report)
    select(True)
    experiment(engine, PHASE, ids, windows, tape)  # warm up all encountered layouts
    expected = torch.cat(tape, dim=1)
    for name in ("baseline", "candidate", "candidate", "baseline"):
        select(name == "candidate")
        measured, tokens, same, _ = experiment(engine, PHASE, ids, windows, tape)
        mismatches = torch.ne(torch.cat(tokens, dim=1), expected).nonzero().tolist()
        if name == "baseline" and not same:
            raise ValueError("Baseline greedy replay mismatch")
        report["runs"].append(dict(method=name, windows=measured, greedy_matches=same,
                                   mismatch_count=len(mismatches), first_mismatch=mismatches[:1]))
        save(out, report)
        print(name, measured, "greedy mismatches", len(mismatches), flush=True)
    totals = report["comparison"] = {}
    for key in full:
        baseline, candidate = (median(r["windows"][key]["ms_per_step"] for r in report["runs"] if r["method"] == name)
                               for name in ("baseline", "candidate"))
        totals[key] = dict(baseline_ms=baseline, candidate_ms=candidate, full_ms=full[key]["ms_per_step"],
                           saved_ms_per_step=baseline-candidate, candidate_minus_full_ms=candidate-full[key]["ms_per_step"])
        print(key, totals[key], flush=True)
    report["greedy_matches"] = all(r["greedy_matches"] for r in report["runs"])
    report["status"] = "complete" if report["greedy_matches"] else "greedy_mismatch"
    save(out, report)
    print("Results:", out, flush=True)
    if not report["greedy_matches"]:
        raise SystemExit("Candidate changed greedy tokens on the baseline replay.")


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--prompt-mma",action="store_true",help="Test the SM80 packed-prompt candidate")
    ap.add_argument("--resume", action="store_true", help="Reuse a matching v1.1 prompt-policy failure")
    ap.add_argument("--late",action="store_true",help="Also include steps 1993-2008 in every pass")
    args=ap.parse_args()
    torch.manual_seed(20261029)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    props=torch.cuda.get_device_properties(0)
    if not ("A800" in props.name or "RTX 4090" in props.name) or torch.cuda.device_count()!=1:
        raise ValueError(f"Expected one visible A800 or comparison 4090, found {props.name}")
    model=os.environ.get("PHASEKV_MODEL","/data/Llama-3.1-8B-Instruct/snapshots/master")
    with gzip.open(HERE/"data/speed_rows.jsonl.gz","rt") as f:
        rows=[json.loads(next(f))["input_ids"][:8192] for _ in range(3)]
    ids=torch.tensor(rows,device="cuda")
    windows=[(137,152),(265,280)]+([(1993,2008)] if args.late else [])
    args.out.mkdir(exist_ok=True,parents=True)
    out=args.out/"summary.json"
    report=dict(status="running",gpu=props.name,sm_count=props.multi_processor_count,
                cc=torch.cuda.get_device_capability(),model=model,torch=str(torch.__version__),
                batch=3,prompt=8192,windows=windows,scope="decode only; stream interval includes launch gaps",
                core_sha256=hashlib.sha256(Path(seg.__file__).read_bytes()).hexdigest(),
                input_sha256=[hashlib.sha256(json.dumps(r).encode()).hexdigest() for r in rows],methods={})
    if args.prompt_mma:
        if args.resume or tuple(report["cc"]) != (8, 0):
            raise SystemExit("--prompt-mma requires a fresh SM80 run")
        report.update(experiment="a800_prompt_mma", prefill_keep_ratio=1.0, decode_keep_ratio=0.8)
        save(out,report)
        return test_prompt_mma(Engine(PHASE,model), ids, windows, report, out)
    if args.resume:
        reuse_failed_run(out, report)
    save(out,report)
    engine=Engine(PHASE,model)  # identical wrappers, select the Full or packed branch for each pass
    phase_options=dict(engine.options)
    timer=Timings(16)
    for method in (FULL,PHASE):
        row = report["methods"].setdefault(method, {})
        if method == FULL and "instrumented" in row:
            continue
        engine.options = phase_options if method==PHASE else {}
        print(f"{method}: warmup and build greedy replay tokens",flush=True)
        _, tape, _, sample=experiment(engine,method,ids,windows,capture=(method==PHASE and "clean" in row))
        if "clean" not in row:
            clean, _, same, sample=experiment(engine,method,ids,windows,tape,capture=(method==PHASE))
            if not same:
                raise ValueError(f"{method}: clean greedy replay mismatch")
            row["clean"]=clean
            save(out,report)
            print(f"{method} uninstrumented: {clean}",flush=True)
        clean=row["clean"]
        profile, _, same, _=experiment(engine,method,ids,windows,tape,timer=timer)
        if not same:
            raise ValueError(f"{method}: instrumentation changed greedy output")
        for k in clean:
            profile[k]["timing_overhead_pct"]=100*(profile[k]["ms_per_step"]/clean[k]["ms_per_step"]-1)
        row["instrumented"]=profile
        save(out,report)
        if method==PHASE:
            report["comparison"] = {}
            for k in clean:
                p, f = (report["methods"][m] for m in (PHASE,FULL))
                gap = p["clean"][k]["ms_per_step"]-f["clean"][k]["ms_per_step"]
                ps, fs = (x["instrumented"][k]["stages"] for x in (p,f))
                stream = lambda stages, names: sum(stages[n]["stream_ms_per_step"] for n in names)
                attn = stream(ps, ["packed_main","merge"])-stream(fs, ["fa2"])
                cache = stream(ps, ["descriptors","append_hot","finish_cache"])-stream(fs, ["dense_cache_update"])
                overhead = max(abs(x["instrumented"][k]["timing_overhead_pct"]) for x in (p,f))
                report["comparison"][k] = dict(clean_phase_minus_full_ms=gap,
                    instrumented_attention_delta_ms=attn,instrumented_cache_delta_ms=cache,
                    max_abs_timing_overhead_pct=overhead,
                    interpretation="use isolated replay; trace overhead > 10%" if overhead>10 else "stream envelopes; host gaps may be included")
                print("Matched-window comparison",k,report["comparison"][k],flush=True)
            save(out,report)
        row["same_gpu_driver_ab"]=compare_drivers(engine,method,rows)
        print(f"{method} A800 formal minus legacy loop, ms/step:",
              row["same_gpu_driver_ab"]["formal_minus_legacy_ms_per_step"],flush=True)
        save(out,report)
        if method==PHASE:
            micro=row["layer_replay"]=replay(sample)
            save(out,report)
            print("Four-layer replay ms per layer:",{k:round(v["median_ms_layer"],4) for k,v in micro["timing"].items()},flush=True)
            sample.clear()
            best=micro["winner"]
            if best!="tile8" and micro["timing"][best+"_pair"]["median_ms_layer"]<.97*micro["timing"]["tile8_pair"]["median_ms_layer"]:
                variant=micro["configs"][best]
                # Warm every encountered layout before the clean A/B, then bracket candidate with baseline.
                experiment(engine,method,ids,windows,tape,variant=variant)
                alt, _, agrees, _=experiment(engine,method,ids,windows,tape,variant=variant)
                after, _, reference_ok, _=experiment(engine,method,ids,windows,tape)
                if not reference_ok:
                    raise ValueError("Reference replay mismatch after candidate")
                row["candidate"]=dict(variant=best,clean=alt,all_greedy_tokens_equal=agrees,baseline_after=after)
                save(out,report)
    report["status"]="complete"
    save(out,report)
    for k in report["methods"][PHASE]["clean"]:
        a,b=(report["methods"][m]["clean"][k]["ms_per_step"] for m in (PHASE,FULL))
        print(f"{k}: PhaseKV {a:.3f}, Full {b:.3f}, gap {a-b:+.3f} ms/step",flush=True)
    print("Results:",out,flush=True)


if __name__=="__main__":
    main()
