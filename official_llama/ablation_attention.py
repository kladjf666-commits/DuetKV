"""Common decode adapter for the 2 x 2 cache/attention experiment."""
import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from PhaseKV.phasekv_cache import PhaseKVCache


class AblationAttention(nn.Module):
    def __init__(self, original, method):
        super().__init__()
        self.original = original
        self.method = method
        self.layer_idx = original.layer_idx
        self.train(original.training)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        cache = past_key_value
        packed = isinstance(cache, PhaseKVCache)
        # A populated dense cache with q_len=1 means decode only for Full.
        # PhaseKV's observation prefill (including length=1) must stay native.
        dense_decode = (self.method.startswith('full_') and cache is not None
                        and hidden_states.shape[1] == 1
                        and cache.get_seq_length(self.layer_idx) > 0)
        if not packed and not dense_decode:
            return self.original(hidden_states=hidden_states, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_value=cache,
                                 output_attentions=output_attentions, use_cache=use_cache,
                                 cache_position=cache_position, position_embeddings=position_embeddings,
                                 **kwargs)
        if (hidden_states.shape[:2] != (1, 1) or hidden_states.dtype != torch.float16
                or hidden_states.device.type != 'cuda' or self.training or not use_cache
                or output_attentions or attention_mask is not None):
            raise ValueError('Ablation decode requires unpadded CUDA FP16 eval, B=1, Q=1, use_cache')
        if packed != self.method.startswith('phasekv_'):
            raise ValueError('Cache policy and ablation method disagree')
        base = self.original
        q = base.q_proj(hidden_states).view(1, 1, base.num_heads, base.head_dim).transpose(1, 2)
        k = base.k_proj(hidden_states).view(1, 1, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        v = base.v_proj(hidden_states).view(1, 1, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        if position_embeddings is None:
            if position_ids is None:
                raise ValueError('Absolute RoPE positions are required')
            cos, sin = base.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        from ablation_kernels import dense_decode, materialize, segmented_phasekv_decode
        if packed:
            cache.append_hot(self.layer_idx, k, v)
            tiers = cache.layer_tiers(self.layer_idx)
            if self.method.endswith('_fa2'):
                k, v = materialize(tiers)
                cache._materialized_peak = max(getattr(cache, '_materialized_peak', 0),
                                               (k.numel() + v.numel()) * 2)
            else:
                context = segmented_phasekv_decode(q[0, :, 0, :], tiers, base.num_heads)
        else:
            k, v = cache.update(k, v, self.layer_idx,
                                {'sin': sin, 'cos': cos, 'cache_position': cache_position})
            if self.method.endswith('_triton'):
                context = dense_decode(q[0, :, 0, :], k, v, cache, self.layer_idx)
        if self.method.endswith('_fa2'):
            from flash_attn import flash_attn_func
            # Q=1 and all retained keys are past/current: bottom-right causal
            # alignment admits every key, even when prompt positions have gaps.
            context = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                      dropout_p=0.0, causal=True)
        context = context.reshape(1, 1, base.num_heads * base.head_dim)
        return base.o_proj(context), None, cache
