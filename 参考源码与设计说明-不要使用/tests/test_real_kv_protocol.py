"""Check eviction semantics and the benchmark's fake-quantization guard."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache

import baselines
from eval.compressors import build_compressor


CONFIG = SimpleNamespace(num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)


def make_cache(length=64):
    cache = DynamicCache()
    for layer in range(3):
        key = torch.arange(length, dtype=torch.bfloat16).reshape(1, 1, length, 1).expand(1, 2, length, 32).clone()
        cache.update(key, -key, layer)
    return cache


class TestEvictionProtocol(unittest.TestCase):
    def test_snapkv_keeps_top_votes_and_recent_tokens_per_head(self):
        cache = make_cache()
        attn = torch.zeros(1, 4, 4, 64)
        attn[:, :2, :, 8:20] = 1
        attn[:, 2:, :, 30:42] = 1
        comp = build_compressor("snapkv", max_cache_tokens=16, obs_window=4, pool_kernel=1)
        comp.on_prefill(cache, attentions=(attn,) * 3, prompt_len=64, model_config=CONFIG)
        expected = [list(range(8, 20)) + list(range(60, 64)),
                    list(range(30, 42)) + list(range(60, 64))]
        for key, value in zip(cache.key_cache, cache.value_cache):
            self.assertEqual(key[0, :, :, 0].int().tolist(), expected)
            self.assertTrue(torch.equal(value, -key))
            self.assertEqual(key.dtype, torch.bfloat16)

    def test_pyramid_budgets_decrease_and_preserve_mean(self):
        comp = build_compressor("pyramidkv", prefill_keep_ratio=0.3, obs_window=16, beta=20)
        budgets = comp._layer_select_k(461, 28, 1520, 16)
        self.assertEqual(budgets, sorted(budgets, reverse=True))
        self.assertAlmostEqual(sum(budgets) / 28, 445, places=6)
        cache = make_cache()
        attn = torch.ones(1, 4, 16, 64)
        comp.on_prefill(cache, attentions=(attn,) * 3, prompt_len=64, model_config=CONFIG)
        for key, value in zip(cache.key_cache, cache.value_cache):
            self.assertTrue(torch.equal(key, -value))
            self.assertTrue(torch.equal(key[0, 0, -16:, 0], torch.arange(48, 64).bfloat16()))

    def test_scope_really_evicts_generated_tokens_but_preserves_prompt(self):
        cache = make_cache(8)
        original = cache.key_cache[0].clone()
        comp = build_compressor("scope", recent_window=2, max_cache_tokens=4)
        comp.reset(CONFIG)
        comp.on_prefill(cache, prompt_len=8, model_config=CONFIG)
        for step in range(10):
            for layer in range(3):
                token = torch.full((1, 2, 1, 32), 8 + step, dtype=torch.bfloat16)
                cache.update(token, -token, layer)
            length = cache.get_seq_length()
            attn = torch.ones(1, 4, 1, length)
            comp.on_decode_step(cache, attentions=(attn,) * 3, step=step, model_config=CONFIG)
            self.assertLessEqual(cache.get_seq_length(), 12)
            self.assertTrue(torch.equal(cache.key_cache[0][:, :, :8], original))
        self.assertEqual(cache.key_cache[0][0, 0, -2:, 0].int().tolist(), [16, 17])

    def test_old_fake_quantization_entry_points_fail_closed(self):
        from scripts.compare_longbench_kv import forbid_fake_quantization
        from baselines import _common
        from src.phasekv import quant
        from eval.compressors import common
        names = ("_grouped_affine_quant_dequant", "quantize_key", "quantize_value", "quantize_kv")
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(_common, "grouped_fake_quant", _common.grouped_fake_quant))
            stack.enter_context(patch.object(common, "fake_quantize", common.fake_quantize))
            for name in names:
                stack.enter_context(patch.object(quant, name, getattr(quant, name)))
            forbid_fake_quantization()
            for name in ("kivi", "phasekv"):
                comp = build_compressor(name, bits=2, recent_window=32, prefill_keep_ratio=0.3)
                comp.reset(CONFIG)
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    comp.on_prefill(make_cache(96), attentions=(torch.ones(1, 4, 32, 96),) * 3,
                                    prompt_len=96, model_config=CONFIG)


if __name__ == "__main__":
    unittest.main()
