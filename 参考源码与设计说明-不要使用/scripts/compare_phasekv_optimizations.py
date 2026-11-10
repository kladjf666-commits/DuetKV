"""Opt-in extension of the existing five-method, same-round evaluation protocol.

The old runner, scoring, tolerances and non-PhaseKV paths are reused unchanged.
Every switch and the saved baseline hashes enter the protocol fingerprint.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import compare_phasekv_same_round as same
from scripts import phasekv_optimization_support as support

b, t = same.b, same.t
ORIGINAL_BUILD = b.build_cache
ORIGINAL_VALIDATE = b.validate_spec
ORIGINAL_INSTALL = b.install_adapters
ORIGINAL_HASHES = b.source_hashes
ORIGINAL_CONFIGS = same.configs
ORIGINAL_REPORT = same.report
ORIGINAL_ARGUMENTS = same.arguments


def configs():
    original = ORIGINAL_CONFIGS()
    controls = original[:-1]
    spec = original[-1][1]
    stages = support.load_config_module().stage_configs()
    variants = []
    for name, switches in stages.items():
        label = "PhaseKV" if name == "p5" else "PhaseKV-" + name
        variants.append((label, dict(spec, optimizations=switches.to_dict(),
                                     saved_baseline=name == "baseline"), "matched"))
    return controls + variants


def validate_spec(spec, chunk_size):
    clean = {k: v for k, v in spec.items() if k not in ("optimizations", "saved_baseline")}
    ORIGINAL_VALIDATE(clean, chunk_size)
    if spec["method"] != "phasekv" and set(spec) != set(clean):
        raise ValueError("Optimization switches only apply to PhaseKV")
    if "optimizations" in spec:
        support.load_config_module().resolve_config(spec["optimizations"])
    if type(spec.get("saved_baseline", False)) is not bool:
        raise ValueError("saved_baseline must be bool")
    if spec.get("saved_baseline") and any(v for k, v in spec.get("optimizations", {}).items()
                                           if k not in ("page_slots", "cold_group")):
        raise ValueError("Saved baseline cannot enable optimizations")


def build_cache(model, output, length, spec):
    if spec["method"] != "phasekv" or "optimizations" not in spec:
        return ORIGINAL_BUILD(model, output, length, spec)
    from PhaseKV.phasekv_cache import PhaseKVCache
    from PhaseKV.prompt_tier import select_prompt_indices
    keep = select_prompt_indices(output.attentions, prompt_len=length, model_config=model.config,
        prefill_keep_ratio=spec["prefill_keep_ratio"], sink_tokens=spec["sink_tokens"],
        obs_window=spec["obs_window"], pool_kernel=7, pooling="maxpool")
    options = {key: spec[key] for key in support.TARGET}
    if spec.get("saved_baseline"):
        cls = support.load_baseline().PhaseKVCache
    else:
        cls = PhaseKVCache
        options["optimizations"] = spec["optimizations"]
    return cls.from_dense_prefill(output.past_key_values, prompt_len=length, keep_indices=keep, **options), None


def install_adapters(model):
    """Teach existing adapters the isolated baseline type, without changing their math."""
    result = ORIGINAL_INSTALL(model)
    import importlib
    from PhaseKV.phasekv_cache import PhaseKVCache
    from PhaseKV.kernels import attention_triton
    baseline = support.load_baseline()
    for family in ("Qwen2.qwen2_attention", "Llama.llama_attention"):
        module = importlib.import_module("models." + family)
        module.PhaseKVCache = (PhaseKVCache, baseline.PhaseKVCache)
    if not getattr(attention_triton.fused_phasekv_decode, "_optimization_dispatch", False):
        original = attention_triton.fused_phasekv_decode
        frozen = importlib.import_module("phasekv_saved_baseline.kernels.segmented_attention")
        def dispatch(q, tiers, heads, **kwargs):
            if isinstance(tiers, baseline.LayerTiers):
                return frozen.segmented_phasekv_decode(q, tiers, heads, **kwargs)
            return original(q, tiers, heads, **kwargs)
        dispatch._optimization_dispatch = True
        attention_triton.fused_phasekv_decode = dispatch
    return result


def hashes():
    result = ORIGINAL_HASHES()
    result.update(support.source_hashes())
    for path in support.BASELINE.rglob("*.py"):
        if "PhaseKV" in path.relative_to(support.BASELINE).parts:
            result[path.relative_to(ROOT).as_posix()] = b.file_hash(path)
    result[(support.BASELINE / "sha256.json").relative_to(ROOT).as_posix()] = b.file_hash(support.BASELINE / "sha256.json")
    return result


def arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--out" or arg.startswith("--out=") for arg in argv):
        argv += ["--out", str(ROOT / "results/phasekv_optimization_20261027/e2e")]
    args = ORIGINAL_ARGUMENTS(argv)
    if args.repeats < 3:
        raise ValueError("Optimization evaluation requires at least three measured repeats")
    return args


def report(out, manifest):
    result = ORIGINAL_REPORT(out, manifest)
    baseline = "PhaseKV-baseline"
    comparisons = {}
    if baseline in result:
        base_rows = b.read_records(out / "runs" / baseline / "records.jsonl")
        for name, spec, _ in manifest["plan"]:
            if name in result and spec["method"] == "phasekv":
                rows = b.read_records(out / "runs" / name / "records.jsonl")
                comparisons[name] = same.comparison_metrics(rows, base_rows, set(manifest["expected"]),
                    spec, manifest["settings"]["max_quality_drop"], intervals=True)
    b.atomic_json(out / "optimization_ablation.json", {"against": baseline, "comparisons": comparisons,
        "cooling_step_percentiles": "separate microbenchmark lifecycle; no per-step profiler in model timing"})
    return result


def configure():
    b.build_cache, b.validate_spec, b.install_adapters, b.source_hashes = build_cache, validate_spec, install_adapters, hashes
    same.configs, same.arguments, same.report = configs, arguments, report
    same.HERE = Path(__file__).resolve()
    same.VERSION = support.PROTOCOL + "-same-round"


def main():
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == "__worker__":
        job = t.read(sys.argv[2])
        with t.worker_policy(job["prefill_policy"], job.get("matched_baseline")):
            b.worker(job)
    elif len(sys.argv) > 1 and sys.argv[1] == "__validate__":
        t.validate_job(t.read(sys.argv[2]), sys.argv[3])
    else:
        same.main()


if __name__ == "__main__":
    main()
