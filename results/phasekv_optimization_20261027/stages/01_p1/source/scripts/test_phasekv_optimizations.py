"""Run required regressions with standalone-package bootstrap; save actual outcomes."""
import argparse
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_4090d import bootstrap_algorithms

PATTERNS = ["test_phasekv_cache.py", "test_phasekv_segmented_decode.py",
            "test_phasekv_launch_config.py", "test_kivi_packed.py",
            "test_qwen2_attention_adapter.py", "test_llama_opt_adapters.py",
            "test_phasekv_fused_quantize.py", "test_phasekv_paged_storage.py",
            "test_phasekv_hot_ring.py", "test_phasekv_optimization_protocol.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patterns", nargs="*", default=PATTERNS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh evidence file")
    bootstrap_algorithms()
    # Historical tests still import the removed baselines/ forwarding package.
    # Alias only in this test process; production imports remain unchanged.
    import importlib
    import types
    legacy = types.ModuleType("baselines")
    legacy.__path__ = []
    sys.modules.setdefault("baselines", legacy)
    sys.modules.setdefault("baselines.kivi_packed", importlib.import_module("KIVI.packed_cache"))
    suite = unittest.TestSuite()
    for pattern in args.patterns:
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    import torch
    report = {"tests_run": result.testsRun, "failures": [(str(t), m) for t, m in result.failures],
              "errors": [(str(t), m) for t, m in result.errors],
              "skips": [(str(t), m) for t, m in result.skipped],
              "successful": result.wasSuccessful(), "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "patterns": args.patterns}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
