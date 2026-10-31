"""Qwen2 generation with genuine packed KIVI or PhaseKV caches."""
import torch
from transformers.cache_utils import DynamicCache

from KIVI.packed_cache import PackedKIVICache
from models.Qwen2.kivi_attention import KIVIQwen2Attention
from models.Qwen2.qwen2 import build_phasekv_cache
from models.Qwen2.qwen2_attention import install_qwen2_phasekv_attention




def install_packed_attention(model):
    if getattr(model, "_real_kv_installed", False):
        return
    install_qwen2_phasekv_attention(model)
    for layer in model.model.layers:
        layer.self_attn = KIVIQwen2Attention(layer.self_attn)
    model._real_kv_installed = True


from models.packed_runtime import storage_evidence, generate_packed as _generate_packed


def generate_packed(model, tokenizer, input_ids, *, method, params, max_new_tokens,
                    trace_compression=False, stop_on_eos=True):
    return _generate_packed(model, tokenizer, input_ids, method=method, params=params,
                           max_new_tokens=max_new_tokens, trace_compression=trace_compression,
                           stop_on_eos=stop_on_eos, install_attention=install_packed_attention,
                           build_cache=build_phasekv_cache)
