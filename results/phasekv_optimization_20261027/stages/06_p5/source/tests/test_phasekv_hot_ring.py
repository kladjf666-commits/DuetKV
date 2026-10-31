"""Token-by-token equivalence of contiguous and circular hot buffers."""
import importlib.util
import sys
import unittest
from unittest.mock import patch
import torch
from transformers.cache_utils import DynamicCache
from scripts.benchmark_4090d import bootstrap_algorithms
if "eval" not in sys.modules:
    bootstrap_algorithms()
from PhaseKV.phasekv_cache import PhaseKVCache, LayerTiers
from PhaseKV.optimizations import OptimizationConfig


class RingContract:
    def run_lifecycle(self, device, keep):
        torch.manual_seed(82)
        dense = DynamicCache()
        dense.update(torch.randn(1, 2, 33, 32, device=device, dtype=torch.bfloat16),
                     torch.randn(1, 2, 33, 32, device=device, dtype=torch.bfloat16), 0)
        args = dict(prompt_len=33, prompt_bits=4, group_size=32,
                    recent_window=5, lazy_g=3, decode_keep_ratio=keep)
        base = PhaseKVCache.from_dense_prefill(dense, **args)
        ring = PhaseKVCache.from_dense_prefill(dense, **args,
            optimizations=OptimizationConfig(fused_quantize=True, paged_cold=True,
                reuse_workspace=True, cold_group=4, specialized_4bit=True, hot_ring=True))
        batch = 1

        def compare():
            a, b = base.layers[0], ring.layers[0]
            self.assertEqual(base.true_seq_length, ring.true_seq_length)
            self.assertEqual(base.get_seq_length(), ring.get_seq_length())
            self.assertEqual(a.hot_len, b.hot_len)
            for x, y in zip(a.hot_views(), b.hot_views()):
                self.assertTrue(torch.equal(x, y))
            for x, y in zip((a.prompt, *a.prompt_extensions, *a.cold),
                            (b.prompt, *b.prompt_extensions, *b.cold)):
                for px, py in ((x.key, y.key), (x.value, y.value)):
                    for name in ("packed", "scale", "mins"):
                        self.assertTrue(torch.equal(getattr(px, name).contiguous().view(torch.uint8),
                                                    getattr(py, name).contiguous().view(torch.uint8)))

        for step in range(120):
            token = torch.randn(batch, 2, 1, 32, device=device, dtype=torch.float16)
            if step in (11, 29, 46):
                length = {11: 1, 29: 17, 46: 4}[step]
                for c in (base, ring):
                    c.begin_prompt_chunk(length)
                    for _ in range(length):
                        c.append_prompt_token(0, token, token)
                    c.finish_prompt_chunk()
                compare()
            if step == 78 and device == "cpu":
                for c in (base, ring):
                    c.reorder_cache(torch.tensor([0, 0], device=device))
                batch = 2
                token = token.expand(2, -1, -1, -1)
            for c in (base, ring):
                c.append_hot(0, token, token)
            compare()
            if device == "cuda":
                from PhaseKV.kernels.segmented_attention import segmented_phasekv_decode
                from test_phasekv_segmented_decode import reference
                q = torch.randn(4, 32, device=device, dtype=torch.float16)
                expected = reference(q, base.layers[0])
                with patch.object(LayerTiers, "hot_views", side_effect=AssertionError("decode materialized hot")):
                    actual = segmented_phasekv_decode(q, ring.layers[0], 4)
                torch.testing.assert_close(actual.float(), expected, atol=.002, rtol=.01)
            for c in (base, ring):
                c.finish_decode_step()
            compare()


class TestHotRingCPU(unittest.TestCase, RingContract):
    def test_wrap_prompt_growth_reorder_and_selection(self):
        for keep in (1., .5):
            with self.subTest(keep=keep):
                self.run_lifecycle("cpu", keep)


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"),
                     "requires actual CUDA + Triton")
class TestHotRingCUDA(unittest.TestCase, RingContract):
    def test_cuda_ring_attention_and_quantize(self):
        for keep in (1., .5):
            self.run_lifecycle("cuda", keep)


if __name__ == "__main__":
    unittest.main()
