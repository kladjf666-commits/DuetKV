"""Fixed-input A/B microbenchmarks; GPU events and synchronized wall time.

The saved two-launch implementation is the comparator. No profiler runs inside
official timing. --cpu-audit reports storage/equivalence only, never GPU speed.
"""
import argparse
from dataclasses import replace
import importlib
import itertools
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import phasekv_optimization_support as support


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", nargs="+", default=["baseline", "p1", "p2a", "p2b", "p3", "p4", "p4_gqa", "p5"])
    p.add_argument("--prompts", type=int, nargs="+", default=support.PROMPTS)
    p.add_argument("--cold-counts", type=int, nargs="+", default=support.COLD_COUNTS)
    p.add_argument("--hot-lengths", type=int, nargs="+", default=support.HOT_LENGTHS)
    p.add_argument("--head-layouts", nargs="+", default=["28:4", "32:32"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--lifecycle-steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--overrides", default="{}", help="JSON switch overrides, recorded in fingerprint")
    p.add_argument("--cpu-audit", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--profile", action="store_true", help="Separate trace-only run; no official timing")
    a = p.parse_args(argv)
    configs = support.load_config_module().stage_configs()
    if set(a.stages) - configs.keys() or a.rounds < 3 or a.iterations < 1 or a.lifecycle_steps < 16:
        p.error("Known stages, >=3 rounds, >=1 iterations, >=16 lifecycle steps required")
    if min(a.prompts) < 32 or min(a.cold_counts) < 0 or not all(0 <= n <= 143 for n in a.hot_lengths):
        p.error("Invalid prompt/cold/hot sizes")
    overrides = json.loads(a.overrides)
    a.configs = {name: (configs[name] if name == "baseline" else replace(configs[name], **overrides)).to_dict()
                 for name in a.stages}
    return a


def measure(call, iterations=1):
    import torch
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    wall = (time.perf_counter() - started) * 1000 / iterations
    return {"gpu_ms": start.elapsed_time(end) / iterations, "wall_ms": wall,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "temporary_peak_bytes": max(0, torch.cuda.max_memory_allocated() - allocated)}


def cache_factory(inputs, config, frozen=False):
    import torch
    from transformers.cache_utils import DynamicCache
    from PhaseKV import phasekv_cache as current
    module = support.load_baseline() if frozen else current
    k, v, cold, hk, hv, _, original = inputs
    dense = DynamicCache()
    dense.update(k, v, 0)
    budget = round(original * .7)
    # Synthetic already-selected indices: reserve the observation tail exactly.
    # No claim that these synthetic values model attention-based importance.
    indices = torch.cat((torch.arange(budget - 32, device=k.device),
                         torch.arange(original - 32, original, device=k.device)))
    indices = indices.view(1, 1, -1).expand(1, k.shape[1], -1)
    options = {} if frozen else {"optimizations": config}
    cache = module.PhaseKVCache.from_dense_prefill(dense, prompt_len=original,
        keep_indices=[indices], **support.TARGET, **options)
    layer = cache.layers[0]
    for ck, cv in cold:
        kwargs = {} if frozen else {"fast": config["fused_quantize"], "pool": layer.cold_pool,
                                    "channel_contiguous_k": config.get("channel_contiguous_k", False)}
        layer.cold.append(module._pack_kv(ck, cv, 4, 32, **kwargs))
    layer.hot_len = hk.shape[2]
    start = 137 if not frozen and config["hot_ring"] else 0
    if not frozen:
        layer.hot_start = start
    for i in range(layer.hot_len):
        position = (start + i) % 144
        layer.hot_k[:, :, position:position + 1].copy_(hk[:, :, i:i + 1])
        layer.hot_v[:, :, position:position + 1].copy_(hv[:, :, i:i + 1])
    return cache


def inputs_for(prompt, count, hot, heads, kvheads, seed, device):
    import torch
    generator = torch.Generator(device=device).manual_seed(seed)
    def random(length, dtype=torch.float16):
        return torch.randn(1, kvheads, length, 128, device=device, dtype=dtype, generator=generator)
    return (random(prompt, torch.bfloat16), random(prompt, torch.bfloat16),
            [(random(16), random(16)) for _ in range(count)], random(hot), random(hot),
            torch.randn(heads, 128, device=device, dtype=torch.float16, generator=generator), prompt)


def assert_storage_equal(base, actual):
    import torch
    a, b = base.layers[0], actual.layers[0]
    for x, y in zip((a.prompt, *a.cold), (b.prompt, *b.cold)):
        for px, py in ((x.key, y.key), (x.value, y.value)):
            for field in ("packed", "scale", "mins"):
                torch.testing.assert_close(getattr(px, field).flatten().contiguous().view(torch.uint8),
                                           getattr(py, field).flatten().contiguous().view(torch.uint8), rtol=0, atol=0)
    for x, y in zip(a.hot_views(), b.hot_views()):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def prepared_calls(q, layer, frozen):
    """Launch-only component calls with exactly the production argument layout."""
    import torch
    import triton
    from PhaseKV.kernels.launch_config import plan_kivi_decode
    from PhaseKV.kernels.attention_triton import _device_sm_count
    prefix = "phasekv_saved_baseline" if frozen else "PhaseKV"
    module = importlib.import_module(prefix + ".kernels.segmented_attention")
    original_kernel = frozen
    if not frozen:
        from PhaseKV.optimizations import uses_optimized_attention
        if not uses_optimized_attention(layer.optimizations):
            module = importlib.import_module("PhaseKV.kernels.segmented_baseline")
            original_kernel = True
    decode = lambda: module.segmented_phasekv_decode(q, layer, q.shape[0])
    decode()  # build descriptors, initialize partials and compile both launches
    w = layer._decode_workspace
    heads, hd = q.shape
    hk, hv = layer.hot_k, layer.hot_v
    config = plan_kivi_decode(hk.shape[2], heads, hd, _device_sm_count(q.device))
    count = w.partial_count
    options, merge_options, grid_heads = {}, {}, heads
    if not original_kernel:
        gqa = layer.optimizations.gqa_reuse and heads // hk.shape[1] > 1
        grid_heads = hk.shape[1] if gqa else heads
        options = dict(CAPACITY=w.capacity, PREFIX=w.prefix_count, COLD_GROUP=w.cold_group,
            NROWS=w.valid_count, FOUR_BIT=layer.optimizations.specialized_4bit and w.compatible_4bit,
            QPER=triton.next_power_of_2(heads // hk.shape[1]) if gqa else 1,
            HOT_CAP=hk.shape[2] if layer.optimizations.hot_ring else 0, HOT_START=layer.hot_start,
            CHANNEL_CONTIGUOUS_K=layer.optimizations.channel_contiguous_k)
        merge_options = dict(CAPACITY=w.capacity)
    out = torch.empty_like(q)
    def attention():
        return module._segmented_attention[(grid_heads, count)](q, w.descriptors, hk, hv,
            w.maxima, w.denominators, w.accumulators, count - 1, count, layer.hot_len,
            *q.stride(), *hk.stride()[1:], *hv.stride()[1:], HD=hd, GROUP=heads // hk.shape[1],
            BN=config.block_n, DW=24, num_warps=config.num_warps, enable_fp_fusion=False, **options)
    def merge():
        return module._merge_segments[(heads,)](w.maxima, w.denominators, w.accumulators,
            out, count, HD=hd, BP=16, num_warps=4, enable_fp_fusion=False, **merge_options)
    kernel = attention()
    resources = {name: getattr(kernel, name, None) for name in ("n_regs", "n_spills")}
    resources["shared_bytes"] = getattr(getattr(kernel, "metadata", None), "shared", None)
    resources["occupancy"] = "not inferred; use separate Nsight/profile run"
    return {"decode": decode, "attention": attention, "merge": merge}, resources


def lifecycle(inputs, config, frozen, steps):
    """Fresh state per round; each token attends before finish_decode_step cools."""
    import torch
    cache = cache_factory(inputs, config, frozen)
    layer = cache.layers[0]
    module = importlib.import_module(("phasekv_saved_baseline" if frozen else "PhaseKV") + ".kernels.segmented_attention")
    q = inputs[5]
    token = inputs[0][:, :, :1].to(torch.float16)
    samples = {"ordinary": [], "cooling": []}
    for _ in range(steps):
        before = len(layer.cold)
        def step():
            cache.append_hot(0, token, -token)
            module.segmented_phasekv_decode(q, layer, q.shape[0])
            cache.finish_decode_step()
        timing = measure(step)
        samples["cooling" if len(layer.cold) > before else "ordinary"].append(timing)
    return samples


def cooling_maintenance(inputs, config, frozen):
    """Isolate finish_decode_step after a real append + attention at the boundary."""
    import torch
    cache = cache_factory(inputs, config, frozen)
    module = importlib.import_module(("phasekv_saved_baseline" if frozen else "PhaseKV") + ".kernels.segmented_attention")
    layer, q = cache.layers[0], inputs[5]
    token = inputs[0][:, :, :1].to(torch.float16)
    while layer.hot_len < 143:
        cache.append_hot(0, token, -token)
        module.segmented_phasekv_decode(q, layer, q.shape[0])
        cache.finish_decode_step()
    cache.append_hot(0, token, -token)
    module.segmented_phasekv_decode(q, layer, q.shape[0])
    return measure(cache.finish_decode_step)


def summarize_samples(samples):
    return {key: {metric: statistics.median(x[metric] for x in values) for metric in values[0]}
            for key, values in samples.items() if values}


def percentiles(values):
    values = sorted(values)
    if not values:
        return None
    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lo, hi = int(position), min(int(position) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {"p50": percentile(.5), "p95": percentile(.95)}


def main(argv=None):
    a = arguments(argv)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    report = {"protocol": support.PROTOCOL, "settings": settings, "source_hashes": support.source_hashes(),
              "baseline_hashes": json.loads((support.BASELINE / "sha256.json").read_text()),
              "quantization": support.TARGET, "prefill_keep_ratio": .7, "obs_window": 32, "sink_tokens": 0,
              "timing_note": "event time includes submission gaps; wall time includes synchronization; profiler separate",
              "memory_note": "microbenchmark keeps comparator caches resident; absolute peaks include them; temporary delta and per-cache owned storage are separate; model peaks come from e2e"}
    report["fingerprint"] = support.digest(report)
    if a.dry_run:
        print(json.dumps(report, indent=2)); return
    if a.output.exists():
        raise FileExistsError("Choose a fresh output; microbenchmarks never resume across code or flag changes")
    support.bootstrap()
    import torch
    from PhaseKV.phasekv_cache import _pack_kv
    from models.packed_runtime import storage_evidence
    device = "cpu" if a.cpu_audit else "cuda"
    report["environment"] = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if device == "cuda" and not torch.cuda.is_available():
        report.update(status="not_run_no_cuda", rows=[])
        support.write_json(a.output, report)
        raise SystemExit("CUDA unavailable; GPU timings were not produced")
    if device == "cuda":
        import triton
        report["environment"].update(gpu=torch.cuda.get_device_name(), triton=triton.__version__)
    report.update(status="running", rows=[])
    for prompt, count, hot, layout in itertools.product(a.prompts, a.cold_counts, a.hot_lengths, a.head_layouts):
        heads, kvheads = map(int, layout.split(":"))
        inputs = inputs_for(prompt, count, hot, heads, kvheads, a.seed, device)
        base = cache_factory(inputs, a.configs.get("baseline", {}), True)
        row = {"shape": dict(prompt_input=prompt, resident_prompt=round(prompt * .7), cold_segments=count,
                              hot_length=hot, heads=heads, kv_heads=kvheads, head_dim=128), "stages": {}}
        caches, calls = {}, {}
        for name in a.stages:
            cache = base if name == "baseline" else cache_factory(inputs, a.configs[name])
            assert_storage_equal(base, cache)
            caches[name] = cache
            evidence = {"physical_bytes": cache.physical_nbytes()} if name == "baseline" else storage_evidence(cache)
            stage = row["stages"][name] = dict(switches=a.configs[name], storage=evidence, rounds=[])
            if device == "cuda":
                calls[name], stage["kernel_resources"] = prepared_calls(inputs[5], cache.layers[0], name == "baseline")
                for _ in range(3):
                    calls[name]["decode"]()
        if device == "cuda":
            baseline_module = importlib.import_module("phasekv_saved_baseline.kernels.segmented_attention")
            expected = baseline_module.segmented_phasekv_decode(inputs[5], base.layers[0], heads)
            for name in a.stages:
                actual = calls[name]["decode"]()
                torch.testing.assert_close(actual, expected, atol=.002, rtol=.01)
                errors = (actual.float() - expected.float()).abs().flatten()
                row["stages"][name]["error"] = dict(max=float(errors.max()), mean=float(errors.mean()),
                    p50=float(torch.quantile(errors, .5)), p95=float(torch.quantile(errors, .95)))
            if a.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for name in a.stages:
                        with torch.profiler.record_function(name):
                            calls[name]["decode"]()
                trace = a.output.with_suffix(f".{len(report['rows'])}.trace.json")
                trace.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(trace))
                row["trace"] = str(trace)
            else:
                for repeat in range(a.rounds):
                    order = a.stages if repeat % 2 == 0 else list(reversed(a.stages))
                    for name in order:
                        values = {key: measure(call, a.iterations) for key, call in calls[name].items()}
                        pack = support.load_baseline()._pack_kv if name == "baseline" else _pack_kv
                        kwargs = {} if name == "baseline" else {"fast": a.configs[name]["fused_quantize"]}
                        # Resident prompt pack and fixed 16-token cooling batch, same FP16 inputs.
                        for label, length in (("prefill_pack", round(prompt * .7)), ("cold_pack", 16)):
                            k = inputs[0][:, :, :length].to(torch.float16)
                            v = inputs[1][:, :, :length].to(torch.float16)
                            call = lambda: pack(k, v, 4, 32, **kwargs)
                            for _ in range(3): call()
                            values[label] = measure(call, a.iterations)
                        # Measure metadata update alone after appending one immutable block.
                        growth = cache_factory(inputs, a.configs[name], name == "baseline")
                        gl = growth.layers[0]
                        gcalls, _ = prepared_calls(inputs[5], gl, name == "baseline")
                        new_segment = pack(k[:, :, :16], v[:, :, :16], 4, 32, **kwargs)
                        gl.cold.append(new_segment)
                        plan = importlib.import_module(("phasekv_saved_baseline" if name == "baseline" else "PhaseKV") + ".kernels.segmented_plan")
                        builder = plan.build_workspace if name == "baseline" else plan.ensure_workspace
                        from PhaseKV.kernels.attention_triton import _device_sm_count
                        from PhaseKV.kernels.launch_config import plan_kivi_decode
                        cfg = plan_kivi_decode(144, heads, 128, _device_sm_count(inputs[5].device))
                        values["descriptor_growth"] = measure(lambda: builder(gl, heads, 128, inputs[5].device,
                            _device_sm_count(inputs[5].device), gl._decode_workspace.key, None, cfg.block_n, cfg.num_warps))
                        # Warm the entire append/attention/cooling sequence before measuring.
                        if repeat == 0:
                            lifecycle(inputs, a.configs[name], name == "baseline", max(160, a.lifecycle_steps))
                        values["lifecycle"] = lifecycle(inputs, a.configs[name], name == "baseline", a.lifecycle_steps)
                        values["cooling_maintenance"] = cooling_maintenance(inputs, a.configs[name], name == "baseline")
                        row["stages"][name]["rounds"].append(values)
                # A standalone copy diagnostic quantifies the operation removed
                # by P5. It never changes the KV/query used by attention timings.
                copy_k = torch.zeros((1, kvheads, 144, 128), device=device, dtype=torch.float16)
                copy_v = torch.zeros_like(copy_k)
                def compact_hot():
                    copy_k[:, :, :128].copy_(copy_k[:, :, 16:144].clone())
                    copy_v[:, :, :128].copy_(copy_v[:, :, 16:144].clone())
                for _ in range(3): compact_hot()
                row["reference_hot_compaction_copy"] = [measure(compact_hot, a.iterations) for _ in range(a.rounds)]
                for name in a.stages:
                    stage = row["stages"][name]
                    rounds = stage["rounds"]
                    stage["medians"] = {key: summarize_samples({key: [r[key] for r in rounds]})[key]
                                        for key in rounds[0] if key != "lifecycle"}
                    stage["partial_count"] = caches[name].layers[0]._decode_workspace.partial_count
                    stage["lifecycle_percentiles_ms"] = {kind: {metric: percentiles(
                        [step[metric] for r in rounds for step in r["lifecycle"][kind]])
                        for metric in ("gpu_ms", "wall_ms")} for kind in ("ordinary", "cooling")}
                    if name != "baseline":
                        stage["storage"] = storage_evidence(caches[name])
                    stage["scratch_bytes"] = sum(t.numel() * t.element_size() for t in
                        (caches[name].layers[0]._decode_workspace.maxima,
                         caches[name].layers[0]._decode_workspace.denominators,
                         caches[name].layers[0]._decode_workspace.accumulators))
        report["rows"].append(row)
        support.write_json(a.output, report)
        print(json.dumps(row["shape"]), flush=True)
    report["status"] = "cpu_storage_only" if a.cpu_audit else "profile_only" if a.profile else "measured"
    support.write_json(a.output, report)


if __name__ == "__main__":
    main()
