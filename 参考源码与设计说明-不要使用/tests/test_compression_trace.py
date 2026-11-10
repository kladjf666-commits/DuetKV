"""State-transition logging, actual activation boundaries, and SCOPE regression."""
import importlib.util
import math
import os
import unittest
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

import baselines
from eval.compression_trace import CompressionTrace, cache_state
from eval.compressors import build_compressor
from baselines.kivi_packed import PackedKIVICache
from src.phasekv.phasekv_cache import PhaseKVCache

CFG = SimpleNamespace(num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)


def dense(length=96):
    cache = DynamicCache()
    for i in range(2):
        k = torch.arange(length).half().reshape(1, 1, length, 1).expand(1, 2, length, 32).clone()
        cache.update(k, -k, i)
    return cache


def append(cache, position):
    for i in range(2):
        k = torch.full((1, 2, 1, 32), position, dtype=torch.float16)
        cache.update(k, -k, i)


class TestCompressionTrace(unittest.TestCase):
    def test_full_and_prefill_eviction_transitions(self):
        for name, params in (("full", {}), ("snapkv", {"max_cache_tokens": 32, "obs_window": 8}),
                             ("pyramidkv", {"max_cache_tokens": 32, "obs_window": 8})):
            c = build_compressor(name, **params)
            c.reset(CFG)
            cache = dense()
            trace = CompressionTrace(name)
            before = cache_state(cache)
            attn = (torch.ones(1, 2, 8, 96),) * 2
            cache = c.on_prefill(cache, attentions=attn, prompt_len=96, model_config=CFG)
            trace.observe("prefill", 0, before, cache_state(cache))
            before = cache_state(cache)
            append(cache, 96)
            cache = c.on_decode_step(cache, step=0, model_config=CFG)
            trace.observe("decode", 1, before, cache_state(cache), appended_tokens=1)
            result = trace.export()
            expected = 0 if name == "full" else 128
            self.assertEqual(result["summary"]["prefill"]["layer_token_totals"]["evicted_tokens"], expected)
            self.assertEqual(result["summary"]["decode"]["triggered_steps"], 0)
            self.assertEqual(result["status"], "baseline_no_compression" if name == "full" else "triggered")

    def test_scope_boundary_and_current_query_not_lifetime_score(self):
        c = build_compressor("scope", recent_window=1, max_cache_tokens=3)
        c.reset(CFG)
        cache = dense(4)
        trace = CompressionTrace("scope", c.describe())
        before = cache_state(cache)
        c.on_prefill(cache, prompt_len=4, model_config=CFG)
        trace.observe("prefill", 0, before, cache_state(cache))
        for step in range(4):
            before = cache_state(cache)
            append(cache, 4 + step)
            scores = torch.zeros(1, 2, 1, cache.get_seq_length())
            if step < 3:
                scores[..., 4] = 1  # formerly dominant token must not be protected forever
            else:
                scores[..., 5] = 0.6
                scores[..., 6] = 0.4
            c.on_decode_step(cache, attentions=(scores,) * 2, step=step, model_config=CFG)
            trace.observe("decode", step + 1, before, cache_state(cache), appended_tokens=1)
            if step == 2:
                self.assertEqual(trace.export()["status"], "not_triggered")
        self.assertEqual(cache.key_cache[0][0, 0, :, 0].int().tolist(), [0, 1, 2, 3, 5, 6, 7])
        result = trace.export()
        self.assertEqual(result["summary"]["decode"]["first_trigger_step"], 4)
        self.assertEqual(result["summary"]["decode"]["layer_token_totals"]["evicted_tokens"], 2)
        self.assertEqual(result["summary"]["decode"]["checks"], 4)

    def test_scope_rejects_unimplemented_or_ambiguous_parameters(self):
        for params in ({"decoding_metric": "adaptive"}, {"decode_keep_ratio": 0.5},
                       {"prefill_keep_ratio": 0.3}):
            with self.assertRaises(ValueError):
                build_compressor("scope", recent_window=2, max_cache_tokens=4, **params)

    def test_kivi_independent_flushes_are_observed(self):
        source = dense(63)
        trace = CompressionTrace("kivi")
        before = cache_state(source)
        cache = PackedKIVICache.from_dense_prefill(source)
        trace.observe("prefill", 0, before, cache_state(cache))
        self.assertEqual(trace.export()["status"], "not_triggered")
        for step in range(2):
            before = cache_state(cache)
            for i in range(2):
                token = torch.zeros(1, 2, 1, 32, dtype=torch.float16)
                cache.append(i, token, token)
                cache.finish_layer(i)
            cache.finish_decode_step()
            trace.observe("decode", step + 1, before, cache_state(cache), appended_tokens=1)
        result = trace.export()
        totals = result["summary"]["decode"]["layer_token_totals"]
        self.assertEqual(totals["quantized_k_tokens"], 128)
        self.assertEqual(totals["quantized_v_tokens"], 2)
        self.assertEqual(totals["evicted_tokens"], 0)
        self.assertEqual(result["events"][0]["actions"], ["key_quantization"])
        self.assertEqual(result["events"][1]["actions"], ["value_quantization"])
        self.assertEqual(result["final_state"]["physical_bytes"], cache.physical_nbytes())

    def test_phasekv_prompt_and_cooling_eviction_are_separate(self):
        source = dense(96)
        trace = CompressionTrace("phasekv")
        before = cache_state(source)
        keep = [torch.arange(32).reshape(1, 1, 32).expand(1, 2, 32)] * 2
        cache = PhaseKVCache.from_dense_prefill(source, prompt_len=96, keep_indices=keep,
                                               recent_window=8, lazy_g=4, decode_keep_ratio=0.5)
        trace.observe("prefill", 0, before, cache_state(cache))
        for step in range(16):
            before = cache_state(cache)
            for i in range(2):
                token = torch.full((1, 2, 1, 32), step, dtype=torch.float16)
                cache.append_hot(i, token, token)
            cache.finish_decode_step()
            trace.observe("decode", step + 1, before, cache_state(cache), appended_tokens=1)
        result = trace.export()
        pre = result["summary"]["prefill"]["layer_token_totals"]
        dec = result["summary"]["decode"]
        self.assertEqual(pre["evicted_tokens"], 128)
        self.assertEqual(pre["quantized_k_tokens"], 64)
        self.assertEqual(dec["first_trigger_step"], 12)
        self.assertEqual(dec["triggered_steps"], 2)
        self.assertEqual(dec["layer_token_totals"]["cooled_tokens"], 8)
        self.assertEqual(dec["layer_token_totals"]["evicted_tokens"], 8)
        self.assertEqual(dec["layer_token_totals"]["hot_tokens_processed"], 16)
        self.assertEqual(result["final_state"]["physical_bytes"], cache.physical_nbytes())

    @unittest.skipUnless(os.environ.get("SCOPE_OFFICIAL_KV_UTILS"), "Set official source path for differential check")
    def test_scope_slide_matches_official_allkv_selected_sets(self):
        spec = importlib.util.spec_from_file_location("scope_official", os.environ["SCOPE_OFFICIAL_KV_UTILS"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        official = module.ALLKVCluster(decoding_metric="slide", decoding_window_size=5,
                                       decoding_recent_size=2, num_hidden_layers=2)
        module.ALLKVCluster.max_capacity_prompt = 4
        comp = build_compressor("scope", recent_window=2, max_cache_tokens=5)
        comp.reset(CFG)
        cache = dense(4)
        comp.on_prefill(cache, prompt_len=4, model_config=CFG)
        rng = torch.Generator().manual_seed(41)
        # Independent random Q/K, deterministic token IDs in V; no equal-score ties.
        official_k = torch.randn(1, 2, 4, 32, generator=rng)
        official_v = cache.value_cache[0].float()
        cache.key_cache = [official_k.clone() for _ in range(2)]
        cache.value_cache = [official_v.clone() for _ in range(2)]
        for step in range(18):
            k = torch.randn(1, 2, 1, 32, generator=rng)
            v = torch.full_like(k, -(4 + step))
            q = torch.randn(1, 2, 1, 32, generator=rng)
            official_k = torch.cat((official_k, k), 2)
            official_v = torch.cat((official_v, v), 2)
            for i in range(2):
                cache.update(k, v, i)
            attn = (q @ cache.key_cache[0].transpose(2, 3) / math.sqrt(32)).softmax(-1)
            comp.on_decode_step(cache, attentions=(attn,) * 2, step=step, model_config=CFG)
            official_k, official_v = official.update_kv_in_decoding(official_k, q, official_v, None, 1)
            torch.testing.assert_close(cache.value_cache[0][..., 0].sort(-1).values,
                                       official_v[..., 0].sort(-1).values)


if __name__ == "__main__":
    unittest.main()
