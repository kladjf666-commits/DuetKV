"""Separate model-step diagnostics. Event instrumentation is excluded from formal throughput runs."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts import phasekv_optimization_support as support
from scripts.bench_phasekv_optimizations import percentiles
import compare_phasekv_optimizations as comparison


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--stage", default="p5", choices=list(support.load_config_module().stage_configs()))
    p.add_argument("--prompt", type=int, default=1024)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    comparison.configure()
    b, t = comparison.b, comparison.t
    args = b.parse_args(["--models", "qwen", "--methods", "phasekv", "--qwen", a.model_path])
    profile = b.profiles_for(args)[0]
    label = "PhaseKV" if a.stage == "p5" else "PhaseKV-" + a.stage
    spec = next(spec for name, spec, _ in comparison.configs() if name == label)
    support.bootstrap()
    import torch
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, tokenizer = b.load_model(profile)
    warm_ids, _ = b.synthetic_input(tokenizer, a.prompt, 1234)
    with t.worker_policy("matched"):
        # Warm the same full trajectory, including descriptor-capacity growth
        # and the cold-group threshold, before installing diagnostic events.
        b.generate(model, tokenizer, warm_ids.to("cuda"), spec, a.tokens, 512, True)
    state, events = {}, []
    original_build = b.build_cache

    def build(*args, **kwargs):
        cache, compressor = original_build(*args, **kwargs)
        state["cache"] = cache
        finish = cache.finish_decode_step
        def measured_finish():
            before = len(cache.layers[0].cold)
            finish()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events.append((state["start"], end, len(cache.layers[0].cold) > before,
                           (time.perf_counter() - state["host_start"]) * 1000))
        cache.finish_decode_step = measured_finish
        return cache, compressor

    def before_forward(module, inputs, kwargs):
        cache = state.get("cache")
        if cache is not None and kwargs.get("past_key_values") is cache:
            state["start"] = torch.cuda.Event(enable_timing=True)
            state["start"].record()
            state["host_start"] = time.perf_counter()

    b.build_cache = build
    hook = model.register_forward_pre_hook(before_forward, with_kwargs=True)
    try:
        with t.worker_policy("matched"):
            result = b.generate(model, tokenizer, warm_ids.to("cuda"), spec, a.tokens, 512, True)
    finally:
        hook.remove()
        b.build_cache = original_build
    torch.cuda.synchronize()
    rows = [{"decode_step": i + 1, "cooling": cooling, "gpu_ms": start.elapsed_time(end),
             "host_submission_ms": host} for i, (start, end, cooling, host) in enumerate(events)]
    summary = {kind: percentiles([r["gpu_ms"] for r in rows if r["cooling"] == cooling])
               for kind, cooling in (("ordinary", False), ("cooling", True))}
    report = {"protocol": support.PROTOCOL + "-separate-step-trace", "spec": spec,
        "prompt": a.prompt, "tokens": a.tokens, "source_hashes": support.source_hashes(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "warning": "Diagnostic instrumentation only. Do not use this run for official tokens/s.",
        "wall_note": "host_submission_ms is asynchronous submission latency, not synchronized wall time",
        "gpu_percentiles_ms": summary, "steps": rows, "storage": result["storage_evidence"],
        "peak_allocated_bytes": result["peak_allocated_bytes"], "peak_reserved_bytes": result["peak_reserved_bytes"]}
    report["fingerprint"] = support.digest({k: report[k] for k in ("protocol", "spec", "prompt", "tokens", "source_hashes")})
    support.write_json(a.output, report)


if __name__ == "__main__":
    main()
