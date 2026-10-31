"""Descriptor/lifecycle checks on CPU and actual kernels on CUDA.

Run directly, even in the standalone checkout without eval/:
  python -m unittest discover -s tests -p test_phasekv_segmented_decode.py -v
For CPU kernel interpretation (not a GPU performance/compilation check):
  TRITON_INTERPRET=1 python -m unittest discover -s tests -p test_phasekv_segmented_decode.py -v
"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if "eval" not in sys.modules and importlib.util.find_spec("eval") is None:
    from scripts.benchmark_4090d import bootstrap_algorithms
    bootstrap_algorithms()

import torch
from transformers.cache_utils import DynamicCache

from PhaseKV.phasekv_cache import PhaseKVCache, LayerTiers, PackedKVSegment
from PhaseKV.kernels.packing import quantize_pack, unpack_low_bits
from PhaseKV.kernels.segmented_plan import build_workspace, segment_rows, DESCRIPTOR_WIDTH
from PhaseKV.optimizations import OptimizationConfig


def make_segment(length, device="cpu", bits=(2, 4), hd=32, hkv=2, gs=32):
    shape = (1, hkv, length, hd)
    return PackedKVSegment(
        quantize_pack(torch.randn(shape, device=device, dtype=torch.float16), bits[0], 2, gs),
        quantize_pack(torch.randn(shape, device=device, dtype=torch.float16), bits[1], 3, gs))


def dense_float(packed):
    # Match existing PhaseKV semantics: apply FP16 scale/min in FP32, without
    # rounding dequantized KV back to FP16 before attention.
    codes = unpack_low_bits(packed.packed, packed.bits, packed.axis_len).float()
    groups = torch.arange(packed.axis_len, device=codes.device) // packed.group_size
    return (codes * packed.scale.float()[..., groups]
            + packed.mins.float()[..., groups]).movedim(-1, packed.axis)


def reference(q, tiers):
    segments = [tiers.prompt, *tiers.prompt_extensions, *tiers.cold]
    k = torch.cat([dense_float(s.key) for s in segments] + [tiers.hot_views()[0].float()], dim=2)[0]
    v = torch.cat([dense_float(s.value) for s in segments] + [tiers.hot_views()[1].float()], dim=2)[0]
    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
    v = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)
    scores = torch.einsum("hd,hsd->hs", q.float(), k) / q.shape[1] ** .5
    return torch.einsum("hs,hsd->hd", scores.softmax(-1), v)


class TestSegmentDescriptors(unittest.TestCase):
    def test_splits_cover_each_segment_once_and_preserve_bytes(self):
        for length in (1, 7, 16, 31, 129, 513):
            seg = make_segment(length)
            before = [x.clone() for p in (seg.key, seg.value) for x in (p.packed, p.scale, p.mins)]
            for splits in (None, 1, 3, 8):
                rows = segment_rows(seg, 4, 32, torch.device("cpu"), 46, splits, 8)
                self.assertTrue(all(len(r) == DESCRIPTOR_WIDTH for r in rows))
                self.assertEqual([n for r in rows for n in range(r[6], r[7])], list(range(length)))
                self.assertEqual(rows[0][:6], [x.data_ptr() for p in (seg.key, seg.value)
                                              for x in (p.packed, p.scale, p.mins)])
                self.assertEqual(rows[0][8:12], [2, 4, min(32, length), 32])
            after = [x for p in (seg.key, seg.value) for x in (p.packed, p.scale, p.mins)]
            for a, b in zip(before, after):
                self.assertTrue(torch.equal(a, b))

    def test_workspace_invalidated_only_when_frozen_segments_change(self):
        dense = DynamicCache()
        dense.update(torch.randn(1, 2, 8, 8, dtype=torch.float16),
                     torch.randn(1, 2, 8, 8, dtype=torch.float16), 0)
        cache = PhaseKVCache.from_dense_prefill(dense, prompt_len=8, group_size=4,
                                               recent_window=2, lazy_g=2)
        layer = cache.layers[0]
        marker = object()
        layer._decode_workspace = marker
        token = torch.randn(1, 2, 1, 8, dtype=torch.float16)
        for _ in range(3):
            cache.append_hot(0, token, token)
            cache.finish_decode_step()
            self.assertIs(layer._decode_workspace, marker)
        cache.append_hot(0, token, token)
        cache.finish_decode_step()
        self.assertIsNone(layer._decode_workspace)
        layer._decode_workspace = marker
        cache.begin_prompt_chunk(2)
        for _ in range(2):
            cache.append_prompt_token(0, token, token)
        cache.finish_prompt_chunk()
        self.assertIsNone(layer._decode_workspace)
        layer._decode_workspace = marker
        cache.reorder_cache(torch.tensor([0]))
        self.assertIsNone(layer._decode_workspace)


INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
HAS_TRITON = importlib.util.find_spec("triton") is not None


@unittest.skipUnless(HAS_TRITON and (torch.cuda.is_available() or INTERPRET),
                     "requires CUDA + Triton or TRITON_INTERPRET=1")
class TestSegmentedKernels(unittest.TestCase):
    @unittest.skipIf(INTERPRET, "actual CUDA optimization dispatch required")
    @torch.inference_mode()
    def test_grouped_and_specialized_match_dense(self):
        from PhaseKV.kernels.attention_triton import fused_phasekv_decode
        for heads, kvheads in ((28, 4), (32, 32)):
            for count in (0, 1, 7, 16, 64, 128, 247, 512):
                tiers = LayerTiers(make_segment(717, "cuda", (4, 4), 128, kvheads),
                    cold=[make_segment(16, "cuda", (4, 4), 128, kvheads) for _ in range(count)],
                    hot_k=torch.randn(1, kvheads, 144, 128, device="cuda", dtype=torch.float16),
                    hot_v=torch.randn(1, kvheads, 144, 128, device="cuda", dtype=torch.float16), hot_len=127)
                q = torch.randn(heads, 128, device="cuda", dtype=torch.float16)
                expected = reference(q, tiers)
                for group in (1, 4, 8, 16):
                    for specialized, reuse in ((False, False), (True, False), (True, True)):
                        tiers.optimizations = OptimizationConfig(reuse_workspace=True, cold_group=group,
                            specialized_4bit=specialized, gqa_reuse=reuse)
                        actual = fused_phasekv_decode(q, tiers, heads)
                        torch.testing.assert_close(actual.float(), expected, atol=.002, rtol=.01)

    def execute(self, q, tiers, options):
        from PhaseKV.kernels.segmented_attention import _segmented_attention, _merge_segments
        from PhaseKV.kernels.launch_config import plan_kivi_decode
        if not INTERPRET:
            from PhaseKV.kernels.attention_triton import fused_phasekv_decode
            return fused_phasekv_decode(q, tiers, q.shape[0], **options)
        hq, hd = q.shape
        config = plan_kivi_decode(tiers.hot_k.shape[2], hq, hd, 46, **options)
        workspace = build_workspace(tiers, hq, hd, q.device, 46, (),
                                    options.get("num_splits"), config.block_n, config.num_warps)
        count = workspace.partial_count
        out = torch.empty_like(q)
        _segmented_attention[(hq, count)](
            q, workspace.descriptors, tiers.hot_k, tiers.hot_v, workspace.maxima,
            workspace.denominators, workspace.accumulators, count - 1, count, tiers.hot_len,
            *q.stride(), *tiers.hot_k.stride()[1:], *tiers.hot_v.stride()[1:],
            HD=hd, GROUP=hq // tiers.hot_k.shape[1], BN=config.block_n, DW=DESCRIPTOR_WIDTH)
        _merge_segments[(hq,)](workspace.maxima, workspace.denominators,
                              workspace.accumulators, out, count, HD=hd, BP=16)
        return out

    @torch.inference_mode()
    def test_mixed_bits_tails_hot_and_many_segments_match_dense(self):
        torch.manual_seed(19)
        device = "cpu" if INTERPRET else "cuda"
        # Mixed K/V bits, partial groups/bytes, empty and strided hot buffers,
        # multi-round prompt extensions, GQA, long reductions and split tails.
        cases = [(0, 0, 8, 2, 4), (1, 1, 32, 4, 4), (5, 7, 64, 8, 2),
                 (33, 33, 128, 2, 8), (253, 47, 128, 2, 4)]
        for count, hot, hd, pb, cb in cases:
            with self.subTest(cold=count, hot=hot, hd=hd):
                tiers = LayerTiers(make_segment(37, device, (pb, pb), hd),
                    prompt_extensions=[make_segment(7, device, (8, 2), hd)],
                    cold=[make_segment(1 + (i % 16), device, (cb, cb), hd) for i in range(count)],
                    hot_k=torch.randn(1, 2, 96, hd, device=device, dtype=torch.float16)[:, :, ::2],
                    hot_v=torch.randn(1, 2, 96, hd, device=device, dtype=torch.float16)[:, :, ::2],
                    hot_len=hot)
                q = torch.randn(4, hd * 2, device=device, dtype=torch.float16)[:, ::2]
                expected = reference(q, tiers)
                for options in ({}, {"num_splits": 3, "block_n": 16, "num_warps": 4}):
                    actual = self.execute(q, tiers, options)
                    torch.testing.assert_close(actual.float(), expected, atol=.002, rtol=.01)
                    if not INTERPRET:
                        tiers.optimizations = OptimizationConfig(reuse_workspace=True,
                            cold_group=8, specialized_4bit=True, gqa_reuse=True)
                        actual = self.execute(q, tiers, options)
                        torch.testing.assert_close(actual.float(), expected, atol=.002, rtol=.01)
                        tiers.optimizations = OptimizationConfig()
                if not INTERPRET:
                    from PhaseKV.kernels.attention_triton import segmented_phasekv_decode_reference
                    old = segmented_phasekv_decode_reference(q, tiers, 4)
                    torch.testing.assert_close(actual, old, atol=.002, rtol=.01)

    @unittest.skipIf(INTERPRET, "workspace dispatch/lifecycle requires CUDA")
    @torch.inference_mode()
    def test_cached_workspace_survives_steps_and_refreshes_on_cooling(self):
        from PhaseKV.kernels.attention_triton import fused_phasekv_decode
        dense = DynamicCache()
        dense.update(torch.randn(1, 2, 17, 32, device="cuda", dtype=torch.float16),
                     torch.randn(1, 2, 17, 32, device="cuda", dtype=torch.float16), 0)
        cache = PhaseKVCache.from_dense_prefill(dense, prompt_len=17, group_size=8,
                                               recent_window=4, lazy_g=3, decode_keep_ratio=.5)
        layer = cache.layers[0]
        q = torch.randn(4, 32, device="cuda", dtype=torch.float16)
        previous = None
        for _ in range(25):
            token = torch.randn(1, 2, 1, 32, device="cuda", dtype=torch.float16)
            cache.append_hot(0, token, token)
            result = fused_phasekv_decode(q, layer, 4)
            torch.testing.assert_close(result.float(), reference(q, layer), atol=.002, rtol=.01)
            if previous is not None:
                self.assertIs(layer._decode_workspace, previous)
            cache.finish_decode_step()
            previous = layer._decode_workspace
        cache.begin_prompt_chunk(3)
        for _ in range(3):
            cache.append_prompt_token(0, token, token)
        cache.finish_prompt_chunk()
        cache.reorder_cache(torch.tensor([0], device="cuda"))
        result = fused_phasekv_decode(q, layer, 4)
        torch.testing.assert_close(result.float(), reference(q, layer), atol=.002, rtol=.01)


if __name__ == "__main__":
    unittest.main()
