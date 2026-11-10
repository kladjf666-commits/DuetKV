"""Model bridges keep ordinary inference intact and accept native prefill KV."""
import unittest
from argparse import Namespace

import torch
from transformers import (
    LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM,
)
from transformers.cache_utils import DynamicCache

from llama.llama_attention import PhaseKVLlamaAttention, install_llama_phasekv_attention
from opt.opt_attention import PhaseKVOPTAttention, install_opt_phasekv_attention
from src.phasekv.model_runner import pack_prefill


class TestModelAdapters(unittest.TestCase):
    @staticmethod
    def _args(ratio):
        return Namespace(
            prefill_keep_ratio=ratio, prompt_bits=2, cold_bits=4,
            group_size=4, recent_window=4, lazy_g=2, decode_keep_ratio=1.0,
        )

    @torch.inference_mode()
    def test_llama_dense_forward_and_prompt_packing(self):
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
            pretraining_tp=1, _attn_implementation="eager",
        )).half().eval()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        before = model(input_ids=ids, use_cache=False).logits
        install_llama_phasekv_attention(model)
        install_llama_phasekv_attention(model)
        after = model(input_ids=ids, use_cache=False).logits
        self.assertTrue(torch.equal(before, after))
        self.assertIsInstance(model.model.layers[0].self_attn, PhaseKVLlamaAttention)

        prefill = model(input_ids=ids, past_key_values=DynamicCache(),
                        output_attentions=True, use_cache=True)
        cache = pack_prefill(prefill, prefill.past_key_values, model, 6, self._args(0.5))
        self.assertEqual(cache.true_seq_length, 6)
        self.assertEqual(cache.get_seq_length(), 4)
        self.assertEqual(len(cache.layers), 2)

    @torch.inference_mode()
    def test_opt_legacy_cache_and_prompt_packing(self):
        model = OPTForCausalLM(OPTConfig(
            vocab_size=97, hidden_size=32, word_embed_proj_dim=32,
            ffn_dim=64, num_hidden_layers=2, num_attention_heads=4,
            max_position_embeddings=64, _attn_implementation="eager",
        )).half().eval()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        before = model(input_ids=ids, use_cache=False).logits
        install_opt_phasekv_attention(model)
        install_opt_phasekv_attention(model)
        after = model(input_ids=ids, use_cache=False).logits
        self.assertTrue(torch.equal(before, after))
        self.assertIsInstance(model.model.decoder.layers[0].self_attn, PhaseKVOPTAttention)

        prefill = model(input_ids=ids, output_attentions=True, use_cache=True)
        self.assertIsInstance(prefill.past_key_values, tuple)
        dense = DynamicCache.from_legacy_cache(prefill.past_key_values)
        cache = pack_prefill(prefill, dense, model, 6, self._args(0.5))
        self.assertEqual(cache.true_seq_length, 6)
        self.assertEqual(cache.get_seq_length(), 4)
        self.assertEqual(len(cache.layers), 2)


if __name__ == "__main__":
    unittest.main()
