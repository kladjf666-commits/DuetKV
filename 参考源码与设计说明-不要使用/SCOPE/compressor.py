"""SCOPE Slide + ALLKV prefill, adapted to the local GQA cache interface.

Reference: https://github.com/Linking-ai/SCOPE/blob/main/model/kv_utils.py
ALLKVCluster.update_kv_in_decoding, decoding_metric='slide'.

The current query selects generated history; prompt and the recent generated
window are protected. No lifetime attention accumulation. This is NOT the whole
SCOPE framework: adaptive/discontinuous and compressed-prefill plug-ins are not
implemented here. Official repeated query-head caches are adapted by summing
query-head scores within each KV group; retained indices are sorted by time.
"""
from __future__ import annotations

import torch

from eval.interface import CompressionConfig, KVCompressor
from eval.compressors import head_ops as _common


class SCOPECompressor(KVCompressor):
    """Explicit ALLKV + Slide variant, using current-step attention."""

    requires_attention = True
    physically_evicts = True

    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        if config.extra.get("decoding_metric", "slide") != "slide":
            raise ValueError("This SCOPE adapter implements only decoding_metric='slide'")
        if config.recent_window is None or config.max_cache_tokens is None:
            raise ValueError("SCOPE Slide requires recent_window and a fixed max_cache_tokens decode budget")
        if config.max_cache_tokens <= config.recent_window:
            raise ValueError("SCOPE Slide decode budget must exceed the recent window")
        if config.decode_keep_ratio is not None:
            raise ValueError("decode_keep_ratio is not a SCOPE Slide parameter; use a fixed decode budget")
        if config.prefill_keep_ratio not in (None, 1.0):
            raise ValueError("This SCOPE adapter uses ALLKV prefill; compressed-prefill plug-ins are unsupported")
        self._prompt_len = 0

    def reset(self, model_config):
        self._prompt_len = 0

    def on_prefill(self, cache, *, attentions=None, prompt_len, model_config):
        self._prompt_len = _common.cache_seq_len(cache)
        return cache

    def on_decode_step(self, cache, *, attentions=None, step, model_config):
        seq_len = _common.cache_seq_len(cache)
        gen_count = seq_len - self._prompt_len
        budget = self.config.max_cache_tokens
        # At equality official code reorders all tokens but removes none. Keeping
        # time order already satisfies that case, so no gather is needed here.
        if gen_count <= budget:
            return cache
        if attentions is None:
            raise ValueError("SCOPE Slide requires current decode attention scores")
        num_kv_heads = _common.num_kv_heads_of(model_config)
        recent = self.config.recent_window
        keep_per_layer = []
        for layer in range(_common.cache_num_layers(cache)):
            attn = attentions[layer]
            if attn.shape[-2] != 1 or attn.shape[-1] != seq_len:
                raise ValueError("SCOPE Slide requires single-token decode attention matching the cache")
            scores = attn[..., self._prompt_len:].float().sum(dim=2)
            scores = _common.aggregate_to_kv_heads(scores, num_kv_heads)
            candidates = gen_count - recent
            top = scores[..., :candidates].topk(budget - recent, dim=-1).indices
            tail = torch.arange(candidates, gen_count, device=scores.device)
            tail = tail.view(1, 1, -1).expand(*scores.shape[:2], recent)
            gen_keep = torch.cat((top, tail), dim=-1).sort(dim=-1).values
            prompt = torch.arange(self._prompt_len, device=scores.device)
            prompt = prompt.view(1, 1, -1).expand(*scores.shape[:2], self._prompt_len)
            keep_per_layer.append(torch.cat((prompt, self._prompt_len + gen_keep), dim=-1))
        return _common.evict_per_head(cache, keep_per_layer)

    def describe(self):
        result = super().describe()
        result["implementation"] = {
            "revision": "scope_slide_current_query_v2",
            "variant": "ALLKV + Slide (GQA group-summed adaptation)",
            "score": "current_query_attention", "prefill": "full_prompt_preserved",
            "decode_budget": self.config.max_cache_tokens,
            "recent_window": self.config.recent_window,
            "unsupported": ["adaptive", "discontinuous", "compressed_prefill_plugins"],
        }
        return result
