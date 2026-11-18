import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from DuetKV.duetkv_cache import DuetKVCache


class BatchedDuetKVAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.layer_idx = original.layer_idx
        self.train(original.training)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        cache = past_key_value
        if not isinstance(cache, DuetKVCache):
            return self.original(hidden_states=hidden_states, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_value=cache,
                                 output_attentions=output_attentions, use_cache=use_cache,
                                 cache_position=cache_position, position_embeddings=position_embeddings,
                                 **kwargs)
        batch, length = hidden_states.shape[:2]
        if (length != 1 or hidden_states.dtype != torch.float16
                or hidden_states.device.type != 'cuda' or self.training or not use_cache
                or output_attentions or attention_mask is not None):
            raise ValueError('Ablation decode requires unpadded CUDA FP16 eval, Q=1, use_cache')
        base = self.original
        q = base.q_proj(hidden_states).view(batch, 1, base.num_heads, base.head_dim).transpose(1, 2)
        k = base.k_proj(hidden_states).view(batch, 1, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        v = base.v_proj(hidden_states).view(batch, 1, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        if position_embeddings is None:
            if position_ids is None:
                raise ValueError('Absolute RoPE positions are required')
            cos, sin = base.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        from .kernels.segmented_attention import segmented_duetkv_decode

        if batch > 1:
            # KV head b*Hkv+h belongs only to request b. The GQA ratio stays unchanged.
            k = k.reshape(1, batch * base.num_key_value_heads, 1, base.head_dim)
            v = v.reshape_as(k)
        cache.append_hot(self.layer_idx, k, v)
        tiers = cache.layer_tiers(self.layer_idx)
        context = segmented_duetkv_decode(q.reshape(batch * base.num_heads, base.head_dim),
                                           tiers, batch * base.num_heads)
        context = context.reshape(batch, 1, base.num_heads * base.head_dim)
        return base.o_proj(context), None, cache
