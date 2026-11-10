"""Shared experiment plumbing; production algorithms never import this module."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/phasekv_optimization_20261027/stages/00_baseline"
PROTOCOL = "phasekv-optimizations-v1-fixed-query-bf16-fp16-w128-g16"
TARGET = dict(prompt_bits=4, cold_bits=4, group_size=32, recent_window=128,
              lazy_g=16, decode_keep_ratio=1.0)
PROMPTS = [1024, 4096, 8192, 16384, 24576, 32000]
COLD_COUNTS = [0, 1, 7, 16, 64, 128, 247, 512]
HOT_LENGTHS = [0, 1, 127, 128, 143]
PERFORMANCE_SHAPES = [(1024, 256), (1024, 1024), (1024, 4096), (4096, 256),
                      (8192, 256), (16384, 256), (16384, 2048), (24576, 4096), (32000, 256)]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes():
    paths = list((ROOT / "PhaseKV").rglob("*.py")) + [ROOT / "models/packed_runtime.py"]
    paths += list((ROOT / "scripts").glob("*phasekv*optimization*.py"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def load_config_module():
    """The controller can read switches without importing torch/eval/PhaseKV."""
    name = "phasekv_experiment_config"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "PhaseKV/optimizations.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_baseline():
    """Load the saved *two-launch* implementation under an isolated package name.

    Check every file before executing it, so a corrupt/edited baseline cannot
    silently become a different comparator. No source restoration or git reset.
    """
    name = "phasekv_saved_baseline"
    hashes = json.loads((BASELINE / "sha256.json").read_text(encoding="utf-8"))
    for relative, expected in hashes.items():
        if relative.startswith("PhaseKV/"):
            path = BASELINE / "source" / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Baseline hash mismatch: {relative}")
    for suffix, path in (("", "PhaseKV"), (".kernels", "PhaseKV/kernels")):
        if name + suffix not in sys.modules:
            package = types.ModuleType(name + suffix)
            package.__path__ = [str(BASELINE / "source" / path)]
            sys.modules[name + suffix] = package
    return importlib.import_module(name + ".phasekv_cache")


def bootstrap():
    from scripts.benchmark_4090d import bootstrap_algorithms
    if "eval" not in sys.modules:
        bootstrap_algorithms()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
