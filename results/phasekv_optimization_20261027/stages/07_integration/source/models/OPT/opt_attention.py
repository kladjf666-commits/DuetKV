"""Packed PhaseKV attention and decoder bridge for Transformers 4.46.3 OPT.

OPT uses a legacy tuple cache and learned absolute positions. Its decoder must
pass the shared PhaseKVCache to every layer and embed the *true* token position
after cache eviction. Ordinary forwards retain the original implementation.
"""
from __future__ import annotations

from types import MethodType

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoder

from PhaseKV.phasekv_cache import PhaseKVCache


class PhaseKVOPTAttention(nn.Module):
    def __init__(self, original: OPTAttention, layer_idx: int):
        super().__init__()
        self.original = original
        self.layer_idx = layer_idx
        self.train(original.training)

    def forward(
        self, hidden_states, key_value_states=None, past_key_value=None,
        attention_mask=None, layer_head_mask=None, output_attentions=False,
        position_ids=None,
    ):
        if not isinstance(past_key_value, PhaseKVCache):
            return self.original(
                hidden_states=hidden_states, key_value_states=key_value_states,
                past_key_value=past_key_value, attention_mask=attention_mask,
                layer_head_mask=layer_head_mask, output_attentions=output_attentions,
                position_ids=position_ids,
            )

        cache = past_key_value
        base = self.original
        batch, q_len, _ = hidden_states.shape
        if batch != 1 or hidden_states.device.type != "cuda":
            raise ValueError("PhaseKV OPT 仅支持 CUDA、batch=1、无 padding")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("PhaseKV OPT 仅支持 FP16/BF16")
        if self.training or output_attentions or key_value_states is not None or layer_head_mask is not None:
            raise ValueError("PhaseKV OPT 需要 eval、self attention、无 head mask")
        if cache.prompt_chunk_len:
            if q_len != cache.prompt_chunk_len:
                raise ValueError("新增 prompt 长度与 begin_prompt_chunk 不一致")
            append = cache.append_prompt_token
        else:
            if q_len != 1:
                raise ValueError("普通 decode 需要 q_len=1；新增 prompt 请先 begin_prompt_chunk")
            append = cache.append_hot

        q = base._shape(base.q_proj(hidden_states), q_len, batch).to(torch.float16)
        k = base._shape(base.k_proj(hidden_states), q_len, batch).to(torch.float16)
        v = base._shape(base.v_proj(hidden_states), q_len, batch).to(torch.float16)

        from PhaseKV.kernels.attention_triton import fused_phasekv_decode

        tiers = cache.layer_tiers(self.layer_idx)
        parts = []
        for index in range(q_len):
            append(self.layer_idx, k[:, :, index:index + 1, :], v[:, :, index:index + 1, :])
            parts.append(fused_phasekv_decode(q[0, :, index, :], tiers, base.num_heads))
        context = torch.stack(parts, dim=0).reshape(batch, q_len, base.embed_dim)
        return base.out_proj(context.to(hidden_states.dtype)), None, cache


def _packed_decoder_forward(
    self, input_ids=None, attention_mask=None, head_mask=None,
    past_key_values=None, inputs_embeds=None, use_cache=None,
    output_attentions=None, output_hidden_states=None, return_dict=None,
    position_ids=None,
):
    if not isinstance(past_key_values, PhaseKVCache):
        return self._phasekv_original_forward(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
            position_ids=position_ids,
        )

    cache = past_key_values
    if self.training or use_cache is not True or output_attentions or output_hidden_states or head_mask is not None:
        raise ValueError("PhaseKV OPT 需要 eval、use_cache=True，且不输出 attentions/hidden states")
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("必须且只能提供 input_ids 或 inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    batch, q_len = inputs_embeds.shape[:2]
    if batch != 1 or inputs_embeds.device.type != "cuda":
        raise ValueError("PhaseKV OPT 仅支持 CUDA、batch=1")
    physical_len = cache.get_seq_length()
    if attention_mask is not None and (attention_mask.shape != (1, physical_len + q_len)
                                       or not bool(attention_mask.all())):
        raise ValueError("PhaseKV OPT 不支持 padding；attention_mask 必须全为 1 且匹配物理长度")
    if position_ids is None:
        position_ids = torch.arange(
            cache.true_seq_length, cache.true_seq_length + q_len,
            device=inputs_embeds.device,
        ).unsqueeze(0)
    if position_ids.shape != (1, q_len):
        raise ValueError("position_ids 必须为 [1, q_len] 的真实绝对位置")
    mask = torch.ones((1, physical_len + q_len), device=inputs_embeds.device, dtype=torch.long)
    positions = self.embed_positions(mask, physical_len, position_ids=position_ids)
    if self.project_in is not None:
        inputs_embeds = self.project_in(inputs_embeds)
    hidden_states = inputs_embeds + positions

    for layer in self.layers:
        hidden_states = layer(
            hidden_states, attention_mask=None, position_ids=position_ids,
            past_key_value=cache, output_attentions=False, use_cache=True,
        )[0]
    if self.final_layer_norm is not None:
        hidden_states = self.final_layer_norm(hidden_states)
    if self.project_out is not None:
        hidden_states = self.project_out(hidden_states)
    if return_dict is False:
        return hidden_states, cache
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=cache)


def install_opt_phasekv_attention(model) -> None:
    """Install OPT attention plus its decoder bridge; safe to call twice."""
    if getattr(model.config, "model_type", None) != "opt":
        raise TypeError("仅支持 model_type='opt'")
    if model.config._attn_implementation != "eager":
        raise TypeError("请以 attn_implementation='eager' 加载 OPT 模型")
    decoder = model.model.decoder
    if not isinstance(decoder, OPTDecoder):
        raise TypeError(f"不支持的 OPT decoder: {type(decoder).__name__}")
    if not hasattr(decoder, "_phasekv_original_forward"):
        decoder._phasekv_original_forward = decoder.forward
        decoder.forward = MethodType(_packed_decoder_forward, decoder)
    for index, layer in enumerate(decoder.layers):
        original = layer.self_attn
        if isinstance(original, PhaseKVOPTAttention):
            continue
        if type(original) is not OPTAttention:
            raise TypeError(f"不支持的 OPT attention: {type(original).__name__}")
        layer.self_attn = PhaseKVOPTAttention(original, index)
