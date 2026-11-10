"""Native Llama attention installation for the shared real packed KV runner."""
from .llama_attention import install_llama_phasekv_attention
from .kivi_attention import KIVILlamaAttention
from models.packed_runtime import build_phasekv_cache, storage_evidence
from models.packed_runtime import generate_packed as _generate_packed
from models.generation import stop_token_ids


def install_packed_attention(model):
    if model.config.model_type != 'llama':
        raise TypeError('Requires model_type=llama')
    if getattr(model, '_llama_real_kv_installed', False):
        return
    install_llama_phasekv_attention(model)
    for layer in model.model.layers:
        layer.self_attn = KIVILlamaAttention(layer.self_attn)
    model._llama_real_kv_installed = True


def generate_packed(model, tokenizer, input_ids, *, method, params, max_new_tokens,
                    trace_compression=False, stop_on_eos=True):
    return _generate_packed(model, tokenizer, input_ids, method=method, params=params,
                           max_new_tokens=max_new_tokens, trace_compression=trace_compression,
                           stop_on_eos=stop_on_eos, install_attention=install_packed_attention,
                           build_cache=build_phasekv_cache,
                           eos_token_ids=stop_token_ids(model, tokenizer))
