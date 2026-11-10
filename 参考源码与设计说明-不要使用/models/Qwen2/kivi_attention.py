"""Qwen2 attention adapter for the packed KIVI cache."""
import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from KIVI.packed_cache import PackedKIVICache


class KIVIQwen2Attention(nn.Module):
    def __init__(self, phase_attention):
        super().__init__()
        self.original = phase_attention
        self.train(phase_attention.training)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None):
        if not isinstance(past_key_value, PackedKIVICache):
            return self.original(hidden_states=hidden_states, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_value=past_key_value,
                                 output_attentions=output_attentions, use_cache=use_cache,
                                 cache_position=cache_position, position_embeddings=position_embeddings)
        if hidden_states.shape[:2] != (1, 1) or output_attentions or not use_cache:
            raise ValueError("KIVI packed attention supports batch=1 single-token cached decode")
        base = self.original.original
        q = base.q_proj(hidden_states).view(1, 1, base.num_heads, base.head_dim).transpose(1, 2)
        k = base.k_proj(hidden_states).view(1, 1, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        v = base.v_proj(hidden_states).view(1, 1, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        if position_embeddings is None:
            cos, sin = base.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        cache = past_key_value
        cache.append(base.layer_idx, k, v)
        from KIVI.kernels import kivi_decode
        output = kivi_decode(q[0, :, 0], cache.layers[base.layer_idx],
                             bits=cache.bits, group_size=cache.group_size)
        cache.finish_layer(base.layer_idx)
        output = output.reshape(1, 1, base.hidden_size)
        return base.o_proj(output.to(hidden_states.dtype)), None, cache

