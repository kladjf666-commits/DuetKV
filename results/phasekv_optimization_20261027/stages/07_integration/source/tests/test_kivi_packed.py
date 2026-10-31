"""Packed storage, official residual schedules and real CUDA attention checks."""
import unittest

import torch
from transformers.cache_utils import DynamicCache

from baselines.kivi_packed import PackedKIVICache
from baselines.packed_runtime import storage_evidence
from src.phasekv.kernels.packing import unpack_dequantize


def dense(length, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(91)
    cache = DynamicCache()
    for layer in range(2):
        k = torch.randn(1, 2, length, 128, device=device, dtype=torch.float16, generator=generator)
        v = torch.randn(1, 2, length, 128, device=device, dtype=torch.float16, generator=generator)
        cache.update(k, v, layer)
    return cache


def materialize(layer):
    def join(packed, residual):
        return torch.cat((unpack_dequantize(packed), residual), dim=2) if packed else residual
    return join(layer.key, layer.key_residual), join(layer.value, layer.value_residual)


class TestPackedKIVI(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Requires a real CUDA GPU")
    @torch.inference_mode()
    def test_qwen_bf16_adapter_matches_dense_quantized_reference(self):
        import copy
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from baselines.packed_runtime import install_packed_attention
        torch.manual_seed(12)
        config = Qwen2Config(vocab_size=97, hidden_size=256, intermediate_size=384,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=256, attn_implementation="eager")
        model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
        reference = copy.deepcopy(model)
        ids = torch.arange(63, device="cuda").unsqueeze(0)
        prefill = model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
        cache = PackedKIVICache.from_dense_prefill(prefill.past_key_values)
        install_packed_attention(model)
        for step in range(4):
            dc = DynamicCache()
            for index, layer in enumerate(cache.layers):
                k, v = materialize(layer)
                dc.update(k.bfloat16(), v.bfloat16(), index)
            token = torch.tensor([[64 + step]], device="cuda")
            positions = torch.tensor([[63 + step]], device="cuda")
            args = dict(input_ids=token, use_cache=True, position_ids=positions,
                        cache_position=positions[0])
            expected = reference(past_key_values=dc, **args).logits
            actual = model(past_key_values=cache, **args).logits
            cache.finish_decode_step()
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
            storage_evidence(cache)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires a real CUDA GPU")
    @torch.inference_mode()
    def test_phasekv_mixed_tiers_against_dense_reference(self):
        from src.phasekv.phasekv_cache import PhaseKVCache
        from src.phasekv.kernels.attention_triton import fused_phasekv_decode
        source = dense(461, "cuda")
        cache = PhaseKVCache.from_dense_prefill(source, prompt_len=461,
                                               prompt_bits=2, cold_bits=4,
                                               recent_window=32, lazy_g=16)
        for step in range(65):
            for index in range(2):
                key = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.float16)
                cache.append_hot(index, key, -key)
            cache.finish_decode_step()
        layer = cache.layers[0]
        query = torch.randn(12, 128, device="cuda", dtype=torch.float16)
        actual = fused_phasekv_decode(query, layer, 12)
        segments = [layer.prompt, *layer.cold]
        k = torch.cat([unpack_dequantize(s.key) for s in segments] + [layer.hot_views()[0]], dim=2)
        v = torch.cat([unpack_dequantize(s.value) for s in segments] + [layer.hot_views()[1]], dim=2)
        k, v = k[0].repeat_interleave(6, 0).float(), v[0].repeat_interleave(6, 0).float()
        weights = torch.einsum("hd,hsd->hs", query.float(), k).div(128**0.5).softmax(-1)
        expected = torch.einsum("hs,hsd->hd", weights, v).half()
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)
        storage_evidence(cache)

    def test_independent_residuals_and_actual_storage(self):
        for length in (31, 63, 64, 65, 127, 128, 1537):
            with self.subTest(length=length):
                source = dense(length)
                cache = PackedKIVICache.from_dense_prefill(source)
                layer = cache.layers[0]
                self.assertEqual(layer.key_residual.shape[2], length % 64)
                self.assertEqual(layer.value_residual.shape[2], min(length, 64))
                if layer.key:
                    self.assertEqual(layer.key.axis, 2)
                    self.assertEqual(layer.key.packed.dtype, torch.uint8)
                if layer.value:
                    self.assertEqual(layer.value.axis, 3)
                self.assertEqual(cache.get_seq_length(), length)
                for residual in (layer.key_residual, layer.value_residual):
                    self.assertEqual(residual.untyped_storage().nbytes(), residual.numel() * 2)
                evidence = storage_evidence(cache)
                self.assertFalse(evidence["dense_shadow"])
                if length >= 128:
                    self.assertLess(evidence["physical_bytes"], sum(t.numel()*2 for t in source.key_cache + source.value_cache))

    def test_flush_boundaries_preserve_prefix_and_lengths(self):
        cache = PackedKIVICache.from_dense_prefill(dense(127))
        original_codes = cache.layers[0].key.packed.clone()
        for step in range(66):
            for index in range(2):
                token = torch.full((1, 2, 1, 128), step / 8, dtype=torch.float16)
                cache.append(index, token, -token)
                cache.finish_layer(index)
            cache.finish_decode_step()
            self.assertEqual(cache.get_seq_length(), 128 + step)
            self.assertEqual(cache.true_seq_length, 128 + step)
            self.assertEqual(cache.layers[0].key_residual.shape[2], (128 + step) % 64)
            self.assertEqual(cache.layers[0].value_residual.shape[2], 64)
            storage_evidence(cache)
        self.assertTrue(torch.equal(cache.layers[0].key.packed[..., :original_codes.shape[-1]], original_codes))
        with self.assertRaisesRegex(RuntimeError, "dense"):
            cache.update(None)

    @unittest.skipUnless(torch.cuda.is_available(), "Requires a real CUDA GPU")
    def test_cuda_kernel_against_independent_dense_attention(self):
        from baselines.kivi_packed_kernel import kivi_decode
        torch.manual_seed(123)
        worst = 0.
        for bits in (2, 4, 8):
            for length in (31, 63, 64, 65, 127, 1537):
                cache = PackedKIVICache.from_dense_prefill(dense(length, "cuda"), bits=bits)
                for step in range(3):
                    key = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.float16)
                    value = torch.randn_like(key)
                    cache.append(0, key, value)
                    layer = cache.layers[0]
                    query = torch.randn(12, 128, device="cuda", dtype=torch.float16)
                    actual = kivi_decode(query, layer, bits=bits, group_size=32)
                    k, v = materialize(layer)
                    k = k[0].repeat_interleave(6, dim=0).float()
                    v = v[0].repeat_interleave(6, dim=0).float()
                    weights = torch.einsum("hd,hsd->hs", query.float(), k).div(128**0.5).softmax(-1)
                    expected = torch.einsum("hs,hsd->hd", weights, v).half()
                    error = (actual - expected).abs().max().item()
                    worst = max(worst, error)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue(torch.allclose(actual, expected, atol=0.002, rtol=0.01),
                                    f"bits={bits}, length={length}, step={step}, max_error={error}")
                    cache.finish_layer(0)
                    cache._pending.clear()
        print(f"KIVI GPU attention max absolute error: {worst:.8f}", flush=True)


if __name__ == "__main__":
    unittest.main()
