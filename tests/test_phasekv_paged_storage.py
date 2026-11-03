"""Capacity reuse and independently audited backing storage."""
import sys
import unittest
from unittest.mock import patch
import torch
from scripts.benchmark_4090d import bootstrap_algorithms
if "eval" not in sys.modules:
    bootstrap_algorithms()
from PhaseKV.optimizations import OptimizationConfig
from PhaseKV.kernels.segmented_plan import ensure_workspace, compatible_4bit
from test_phasekv_segmented_decode import make_segment
from PhaseKV.phasekv_cache import LayerTiers
from PhaseKV.phasekv_cache import PhaseKVCache
from transformers.cache_utils import DynamicCache
from models.packed_runtime import storage_evidence


class TestWorkspaceCapacity(unittest.TestCase):
    def test_real_cooling_keeps_workspace_and_appends_visible_rows(self):
        dense = DynamicCache()
        dense.update(torch.randn(1, 2, 33, 32, dtype=torch.float16),
                     torch.randn(1, 2, 33, 32, dtype=torch.float16), 0)
        cache = PhaseKVCache.from_dense_prefill(dense, prompt_len=33, recent_window=2, lazy_g=2,
            optimizations=OptimizationConfig(reuse_workspace=True, paged_cold=True))
        layer = cache.layers[0]
        workspace = ensure_workspace(layer, 4, 32, torch.device("cpu"), 46, (), 1, 8, 4)
        pointer = workspace.accumulators.data_ptr()
        for _ in range(4):
            token = torch.randn(1, 2, 1, 32, dtype=torch.float16)
            cache.append_hot(0, token, token)
            cache.finish_decode_step()
            self.assertIs(layer._decode_workspace, workspace)
        self.assertEqual(workspace.cold_count, 0)
        ensure_workspace(layer, 4, 32, torch.device("cpu"), 46, (), 1, 8, 4)
        self.assertEqual(workspace.cold_count, 1)
        self.assertEqual(workspace.accumulators.data_ptr(), pointer)
        self.assertEqual(int(workspace.descriptors[1, 0]), layer.cold[0].key.packed.data_ptr())

    def test_specialization_rejects_incompatible_bits_and_groups(self):
        self.assertTrue(compatible_4bit(make_segment(33, bits=(4, 4)), prompt=True))
        self.assertTrue(compatible_4bit(make_segment(16, bits=(4, 4)), prompt=False))
        self.assertFalse(compatible_4bit(make_segment(15, bits=(4, 4)), prompt=False))
        self.assertFalse(compatible_4bit(make_segment(33, bits=(2, 4)), prompt=True))
        self.assertFalse(compatible_4bit(make_segment(33, bits=(4, 4), gs=8), prompt=True))

    def test_grouped_schedule_covers_all_descriptors(self):
        for count in (0, 1, 7, 16, 64, 128, 247, 512):
            for requested in (1, 4, 8, 16):
                tiers = LayerTiers(make_segment(33), cold=[make_segment(16) for _ in range(count)],
                    hot_k=torch.empty(1, 2, 144, 32), hot_v=torch.empty(1, 2, 144, 32),
                    optimizations=OptimizationConfig(reuse_workspace=True, cold_group=requested))
                w = ensure_workspace(tiers, 4, 32, torch.device("cpu"), 46, (), 1, 8, 4)
                visited = []
                for part in range(w.partial_count - 1):
                    start = part if part < w.prefix_count else w.prefix_count + (part - w.prefix_count) * w.cold_group
                    end = min(start + (1 if part < w.prefix_count else w.cold_group), w.valid_count)
                    visited.extend(range(start, end))
                self.assertEqual(visited, list(range(w.valid_count)))
                self.assertLessEqual(w.partial_count, w.valid_count + 1)

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
    @unittest.skipUnless(torch.cuda.is_available(), "requires actual packed CUDA decode")
    def test_batched_keep80_matches_each_layer_across_wrap_and_page_growth(self):
        from PhaseKV.kernels.segmented_attention import segmented_phasekv_decode
        from PhaseKV import phasekv_cache as runtime

        torch.manual_seed(173)
        dense = DynamicCache()
        for layer_id in range(4):
            dense.update(*(torch.randn(1, 2, 33, 128, dtype=torch.float16, device="cuda")
                           for _ in range(2)), layer_id)
        options = OptimizationConfig(paged_cold=True, fused_quantize=True, reuse_workspace=True,
                                     channel_contiguous_k=True, specialized_4bit=True, hot_ring=True,
                                     gqa_reuse=True, page_slots=2)
        ref, batch = (PhaseKVCache.from_dense_prefill(dense, prompt_len=33, prompt_bits=4,
                      recent_window=3, lazy_g=16, decode_keep_ratio=.8, optimizations=options)
                      for _ in range(2))
        original_gather = runtime._gather_kv
        q = torch.randn(8, 128, dtype=torch.float16, device="cuda")
        for step in range(51):
            for layer_id in range(4):
                k, v = (torch.randn(1, 2, 1, 128, dtype=torch.float16, device="cuda") for _ in range(2))
                # Exact ties in one head, distinct keys reveal a changed tie choice.
                v[:, 0].zero_()
                v[:, 0, 0, step % 128] = 1
                for cache in (ref, batch):
                    cache.append_hot(layer_id, k, v)
            with patch.object(ref, "_cool_batch", return_value=False), \
                    patch.object(runtime, "_gather_kv", wraps=original_gather) as old_gather:
                ref.finish_decode_step()
            with patch.object(runtime, "_gather_kv", wraps=original_gather) as new_gather:
                batch.finish_decode_step()
            if old_gather.called:
                self.assertEqual(old_gather.call_count, 4)
                self.assertEqual(new_gather.call_count, 1)
                expected = torch.cat([call.args[2] for call in old_gather.call_args_list])
                self.assertTrue(torch.equal(expected, new_gather.call_args.args[2]))
                self.assertEqual(batch.physical_nbytes(), ref.physical_nbytes())
                for a, b in zip(ref.layers, batch.layers):
                    self.assertEqual((a.hot_len, a.hot_start), (b.hot_len, b.hot_start))
                    for old, new in zip(a.cold, b.cold):
                        for x, y in ((old.key, new.key), (old.value, new.value)):
                            for name in ("packed", "scale", "mins"):
                                left = getattr(x, name).reshape(-1).contiguous().view(torch.uint8)
                                right = getattr(y, name).reshape(-1).contiguous().view(torch.uint8)
                                self.assertTrue(torch.equal(left, right))
                    self.assertTrue(torch.equal(segmented_phasekv_decode(q, a, 8),
                                                segmented_phasekv_decode(q, b, 8)))
        self.assertEqual(len(batch.layers[0].cold), 3)
        for cache in (ref, batch):
            cache.reorder_cache(torch.tensor([0, 0], device="cuda"))
            # A changed batch size releases the old page and resumes scalar cooling.
            for _ in range(16):
                for layer_id in range(4):
                    cache.append_hot(layer_id, k.expand(2, -1, -1, -1), v.expand(2, -1, -1, -1))
                cache.finish_decode_step()
        self.assertIsNone(batch._batched_cold_pool)
        for a, b in zip(ref.layers, batch.layers):
            self.assertEqual(b.cold[-1].key.orig_shape, (2, 2, 13, 128))
            self.assertTrue(torch.equal(a.cold[-1].key.packed, b.cold[-1].key.packed))
        self.assertEqual(batch.physical_nbytes(), ref.physical_nbytes())

    def test_channel_contiguous_key_preserves_codes_during_cooling_and_prompt_extension(self):
        dense = DynamicCache()
        tensors = [torch.randn(1, 2, 67, 128, dtype=torch.float16) for _ in range(2)]
        dense.update(*tensors, 0)
        caches = [PhaseKVCache.from_dense_prefill(dense, prompt_len=67, prompt_bits=4, cold_bits=4,
                    recent_window=2, lazy_g=2, optimizations=OptimizationConfig(
                        paged_cold=True, page_slots=2, hot_ring=True, channel_contiguous_k=flag))
                  for flag in (False, True)]
        for cache in caches:
            for i in range(8):
                cache.append_hot(0, *(t[:, :, i:i+1] for t in tensors))
                cache.finish_decode_step()
            cache.begin_prompt_chunk(3)
            for i in range(3):
                cache.append_prompt_token(0, *(t[:, :, i:i+1] for t in tensors))
            cache.finish_prompt_chunk()
        old, new = (c.layers[0] for c in caches)
        self.assertGreater(len(new.cold), 2)
        self.assertTrue(new.prompt_extensions)
        self.assertEqual(caches[0].physical_nbytes(), caches[1].physical_nbytes())
        for a, b in zip((old.prompt, *old.cold, *old.prompt_extensions),
                        (new.prompt, *new.cold, *new.prompt_extensions), strict=True):
            for name in ('packed', 'scale', 'mins'):
                self.assertEqual(getattr(b.key, name).stride(-2), 1)
                self.assertTrue(torch.equal(getattr(a.key, name), getattr(b.key, name)))
                self.assertTrue(torch.equal(getattr(a.value, name), getattr(b.value, name)))

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
