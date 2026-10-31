"""PhaseKV packed attention for Transformers 4.46.3 Llama-2/3.1 models.

The original module handles the first, dense prefill. Packed decode supports
one unpadded CUDA sequence and uses the model's absolute RoPE positions.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from PhaseKV.phasekv_cache import PhaseKVCache


class PhaseKVLlamaAttention(nn.Module):
    def __init__(self, original: LlamaAttention):
        super().__init__()
        self.original = original
        self.layer_idx = original.layer_idx
        self.train(original.training)

    def forward(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False,
        cache_position=None, position_embeddings=None, **kwargs,
    ):
        if not isinstance(past_key_value, PhaseKVCache):
            return self.original(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings,
                **kwargs,
            )

        cache = past_key_value
        base = self.original
        batch, q_len, _ = hidden_states.shape
        if batch != 1 or hidden_states.device.type != "cuda":
            raise ValueError("PhaseKV Llama 仅支持 CUDA、batch=1、无 padding")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("PhaseKV Llama 仅支持 FP16/BF16")
        if self.training or not use_cache or output_attentions:
            raise ValueError("PhaseKV Llama 需要 eval、use_cache=True、output_attentions=False")
        if self.layer_idx is None or base.config.pretraining_tp != 1:
            raise ValueError("PhaseKV Llama 需要 layer_idx 且 pretraining_tp=1")
        if base.num_heads % base.num_key_value_heads:
            raise ValueError("query heads 必须可被 KV heads 整除")

        if cache.prompt_chunk_len:
            if q_len != cache.prompt_chunk_len:
                raise ValueError("新增 prompt 长度与 begin_prompt_chunk 不一致")
            append = cache.append_prompt_token
        else:
            if q_len != 1:
                raise ValueError("普通 decode 需要 q_len=1；新增 prompt 请先 begin_prompt_chunk")
            append = cache.append_hot

        q = base.q_proj(hidden_states).view(batch, q_len, base.num_heads, base.head_dim).transpose(1, 2)
        k = base.k_proj(hidden_states).view(batch, q_len, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        v = base.v_proj(hidden_states).view(batch, q_len, base.num_key_value_heads, base.head_dim).transpose(1, 2)
        if position_embeddings is None:
            if position_ids is None:
                raise ValueError("必须提供真实绝对 position_ids")
            cos, sin = base.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

        from PhaseKV.kernels.attention_triton import fused_phasekv_decode

        tiers = cache.layer_tiers(self.layer_idx)
        parts = []
        for index in range(q_len):
            append(self.layer_idx, k[:, :, index:index + 1, :], v[:, :, index:index + 1, :])
            parts.append(fused_phasekv_decode(q[0, :, index, :], tiers, base.num_heads))
        context = torch.stack(parts, dim=0).reshape(batch, q_len, base.num_heads * base.head_dim)
        return base.o_proj(context.to(hidden_states.dtype)), None, cache


def install_llama_phasekv_attention(model) -> None:
    """Wrap Llama attention in place; safe to call more than once."""
    if getattr(model.config, "model_type", None) != "llama":
        raise TypeError("仅支持 model_type='llama'")
    if model.config._attn_implementation != "eager":
        raise TypeError("请以 attn_implementation='eager' 加载 Llama 模型")
    if model.config.pretraining_tp != 1:
        raise ValueError("尚不支持 pretraining_tp > 1")
    for layer in model.model.layers:
        original = layer.self_attn
        if isinstance(original, PhaseKVLlamaAttention):
            continue
        if type(original) is not LlamaAttention:
            raise TypeError(f"不支持的 Llama attention: {type(original).__name__}")
        layer.self_attn = PhaseKVLlamaAttention(original)
