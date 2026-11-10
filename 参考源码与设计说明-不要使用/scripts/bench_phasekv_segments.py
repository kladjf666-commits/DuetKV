"""Compare old/new PhaseKV attention, including Python submission overhead.

Uses resident synthetic KV, not model generation. No CUDA graphs: hiding CPU
launch overhead would hide the very bottleneck this change is meant to fix.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if "eval" not in sys.modules and importlib.util.find_spec("eval") is None:
    from scripts.benchmark_4090d import bootstrap_algorithms
    bootstrap_algorithms()

import torch
import triton

from PhaseKV.phasekv_cache import LayerTiers, PackedKVSegment
from PhaseKV.kernels.packing import quantize_pack
from PhaseKV.kernels.attention_triton import (
    fused_phasekv_decode, segmented_phasekv_decode_reference,
)


def measure(call, iterations):
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, nargs="+", default=[0, 16, 125, 189, 253])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/phasekv_segments.json"))
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.heads, args.kv_heads, args.head_dim) <= 0 or min(args.segments) < 0:
        parser.error("counts/dimensions must be positive; segment counts must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; no estimated timings are reported")
    torch.manual_seed(42)

    def segment(length, bits):
        shape = (1, args.kv_heads, length, args.head_dim)
        return PackedKVSegment(
            quantize_pack(torch.randn(shape, device="cuda", dtype=torch.float16), bits, 2, 32),
            quantize_pack(torch.randn(shape, device="cuda", dtype=torch.float16), bits, 3, 32))

    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "triton": triton.__version__, "shape": [args.heads, args.kv_heads, args.head_dim],
              "measurement": "single-layer synchronized wall time, includes Python submission; excludes model and cooling",
              "rows": []}
    for count in args.segments:
        shape = (1, args.kv_heads, 48, args.head_dim)
        tiers = LayerTiers(segment(307, 2), cold=[segment(16, 4) for _ in range(count)],
                           hot_k=torch.randn(shape, device="cuda", dtype=torch.float16),
                           hot_v=torch.randn(shape, device="cuda", dtype=torch.float16), hot_len=47)
        q = torch.randn(args.heads, args.head_dim, device="cuda", dtype=torch.float16)
        calls = {"old": lambda: segmented_phasekv_decode_reference(q, tiers, args.heads),
                 "new": lambda: fused_phasekv_decode(q, tiers, args.heads)}
        old, new = calls["old"](), calls["new"]()
        torch.testing.assert_close(new, old, atol=.002, rtol=.01)
        error = float((old.float() - new.float()).abs().max().item())
        for call in calls.values():
            for _ in range(3):
                call()
        times = {name: [] for name in calls}
        for repeat in range(args.repeats):
            # Alternate ordering to reduce clock/temperature ordering bias.
            for name in (["old", "new"] if repeat % 2 == 0 else ["new", "old"]):
                times[name].append(measure(calls[name], args.iterations))
        # Include descriptor construction as a separate measured operation;
        # steady-state timings above deliberately reuse the unchanged cache.
        def rebuild():
            tiers._decode_workspace = None
            return calls["new"]()
        rebuild_ms = measure(rebuild, min(5, args.iterations))
        medians = {name: statistics.median(samples) for name, samples in times.items()}
        row = {"cold_segments": count, "max_abs_error": error,
               "samples_ms": times, "median_ms": medians,
               "speedup_single_layer": medians["old"] / medians["new"],
               "new_with_descriptor_rebuild_ms": rebuild_ms}
        result["rows"].append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
