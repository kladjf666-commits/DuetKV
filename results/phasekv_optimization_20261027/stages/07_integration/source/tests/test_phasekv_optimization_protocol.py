"""Independent controls, fingerprints, frozen comparator, and same-round isolation."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from scripts import phasekv_optimization_support as support
support.bootstrap()
from scripts.bench_phasekv_optimizations import inputs_for, cache_factory, assert_storage_equal
from PhaseKV.optimizations import OptimizationConfig, stage_configs


class TestOptimizationProtocol(unittest.TestCase):
    def test_default_and_invalid_switches(self):
        original = support.BASELINE / "source/PhaseKV/kernels/segmented_attention.py"
        preserved = support.ROOT / "PhaseKV/kernels/segmented_baseline.py"
        self.assertEqual(original.read_bytes(), preserved.read_bytes())
        self.assertEqual(stage_configs()["baseline"], OptimizationConfig())
        for invalid in ({"cold_group": 3}, {"page_slots": 0}, {"hot_ring": "false"}):
            with self.assertRaises(ValueError):
                OptimizationConfig(**invalid)
        base = OptimizationConfig().to_dict()
        self.assertNotEqual(support.digest(base), support.digest({**base, "hot_ring": True}))

    def test_every_stage_matches_saved_storage(self):
        inputs = inputs_for(1024, 7, 143, 28, 4, 1234, "cpu")
        baseline = cache_factory(inputs, {}, True)
        for name, config in stage_configs().items():
            assert_storage_equal(baseline, cache_factory(inputs, config.to_dict()))

    def test_individual_switches_can_be_disabled(self):
        from dataclasses import replace
        final = stage_configs()["p5"]
        inputs = inputs_for(1024, 7, 143, 28, 4, 12, "cpu")
        baseline = cache_factory(inputs, {}, True)
        for field in ("fused_quantize", "reuse_workspace", "paged_cold", "specialized_4bit", "hot_ring"):
            assert_storage_equal(baseline, cache_factory(inputs, replace(final, **{field: False}).to_dict()))
        assert_storage_equal(baseline, cache_factory(inputs, replace(final, cold_group=1).to_dict()))

    def test_e2e_plan_preserves_other_algorithms_and_full_matrix(self):
        script = support.ROOT / "scripts/compare_phasekv_optimizations.py"
        run = subprocess.run([sys.executable, str(script), "--dry-run"], capture_output=True, text=True, check=True)
        plan = json.loads(run.stdout)
        groups = {name: (spec, policy) for name, spec, policy in plan["groups"]}
        self.assertEqual(groups["KIVI"][0], {"method": "kivi", "bits": 2, "group_size": 32, "recent_window": 64})
        self.assertEqual(groups["SnapKV"][0], {"method": "snapkv", "prefill_keep_ratio": .3, "obs_window": 32})
        self.assertTrue(groups["PhaseKV-baseline"][0]["saved_baseline"])
        self.assertTrue(groups["PhaseKV"][0]["optimizations"]["hot_ring"])
        shapes = {(n, m) for n, m, _ in plan["settings"]["performance_shapes"]}
        self.assertEqual(shapes, set(support.PERFORMANCE_SHAPES))
