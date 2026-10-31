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
from PhaseKV.phasekv_cache import PhaseKVCache
from transformers.cache_utils import DynamicCache
from models.packed_runtime import storage_evidence


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


class TestPagedCold(unittest.TestCase):
    def test_pages_keep_old_bytes_and_count_unused_capacity_once(self):
        dense = DynamicCache()
        dense.update(torch.randn(1, 2, 17, 32, dtype=torch.float16),
                     torch.randn(1, 2, 17, 32, dtype=torch.float16), 0)
        args = dict(prompt_len=17, prompt_bits=4, recent_window=4, lazy_g=2)
        base = PhaseKVCache.from_dense_prefill(dense, **args)
        cache = PhaseKVCache.from_dense_prefill(dense, **args,
            optimizations=OptimizationConfig(paged_cold=True, page_slots=4, reuse_workspace=True))
        first_pointer = None
        for step in range(36):
            token = torch.randn(1, 2, 1, 32, dtype=torch.float16)
            for c in (base, cache):
                c.append_hot(0, token, token)
                c.finish_decode_step()
            for a, b in zip(base.layers[0].cold, cache.layers[0].cold):
                for pa, pb in ((a.key, b.key), (a.value, b.value)):
                    for name in ("packed", "scale", "mins"):
                        self.assertTrue(torch.equal(getattr(pa, name), getattr(pb, name)))
            if cache.layers[0].cold:
                pointer = cache.layers[0].cold[0].key.packed.data_ptr()
                if first_pointer is not None:
                    self.assertEqual(first_pointer, pointer)
                first_pointer = pointer
            owners = { (str(t.device), t.untyped_storage()._cdata): t.untyped_storage().nbytes()
                       for _, t in cache.storage_tensors() }
            self.assertEqual(sum(owners.values()), cache.physical_nbytes())
            evidence = storage_evidence(cache)
            self.assertEqual(evidence["physical_bytes"], sum(owners.values()))
            self.assertEqual(evidence["effective_payload_bytes"] + evidence["page_free_bytes"]
                             + evidence["hot_free_bytes"], evidence["physical_bytes"])
        cache.reorder_cache(torch.tensor([0, 0]))
        self.assertEqual(storage_evidence(cache)["page_free_bytes"], 0)
        cache.begin_prompt_chunk(3)
        token = token.expand(2, -1, -1, -1)
        for _ in range(3):
            cache.append_prompt_token(0, token, token)
        cache.finish_prompt_chunk()
        for _ in range(4):
            cache.append_hot(0, token, token)
            cache.finish_decode_step()
        storage_evidence(cache)
