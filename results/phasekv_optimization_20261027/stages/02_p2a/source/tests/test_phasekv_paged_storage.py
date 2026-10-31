"""Capacity reuse and independently audited backing storage."""
import sys
import unittest
import torch
from scripts.benchmark_4090d import bootstrap_algorithms
if "eval" not in sys.modules:
    bootstrap_algorithms()
from PhaseKV.optimizations import OptimizationConfig
from PhaseKV.kernels.segmented_plan import ensure_workspace
from test_phasekv_segmented_decode import make_segment
from PhaseKV.phasekv_cache import LayerTiers


class TestWorkspaceCapacity(unittest.TestCase):
    def test_incremental_growth_preserves_pointers_and_rows(self):
        tiers = LayerTiers(make_segment(17), hot_k=torch.empty(1, 2, 144, 32),
                           hot_v=torch.empty(1, 2, 144, 32),
                           optimizations=OptimizationConfig(reuse_workspace=True))
        def ensure(key=()):
            return ensure_workspace(tiers, 4, 32, torch.device("cpu"), 46, key, 1, 8, 4)
        workspace = ensure()
        first = workspace.descriptors[0].clone()
        old_capacity, old_scratch = workspace.capacity, workspace.accumulators.data_ptr()
        for i in range(300):
            tiers.cold.append(make_segment(16))
            result = ensure()
            self.assertIs(result, workspace)
            if result.capacity == old_capacity:
                self.assertEqual(result.accumulators.data_ptr(), old_scratch)
            old_capacity, old_scratch = result.capacity, result.accumulators.data_ptr()
            self.assertTrue(torch.equal(result.descriptors[0], first))
            self.assertEqual(result.valid_count, i + 2)
            self.assertEqual(len(result.owners), i + 2)
        self.assertIsNot(ensure(("new-stream",)), workspace)
        tiers.prompt_extensions.append(make_segment(5))
        self.assertEqual(ensure().extension_count, 1)
