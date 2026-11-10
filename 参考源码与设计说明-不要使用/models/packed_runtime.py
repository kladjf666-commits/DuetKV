"""Shared packed generation loop; each model family supplies its own attention adapter."""
import torch
from transformers.cache_utils import DynamicCache
from KIVI.packed_cache import PackedKIVICache
from PhaseKV.phasekv_cache import PhaseKVCache
from PhaseKV.prompt_tier import select_prompt_indices


def build_phasekv_cache(prefill, model, prompt_len, *, prompt_bits=2, cold_bits=4,
                        prefill_keep_ratio=0.3, decode_keep_ratio=1.0, sink_tokens=0,
                        obs_window=32, recent_window=32, lazy_g=16, group_size=32,
                        optimizations=None):
    keep = None if prefill_keep_ratio == 1.0 else select_prompt_indices(
        prefill.attentions, prompt_len=prompt_len, model_config=model.config,
        prefill_keep_ratio=prefill_keep_ratio, sink_tokens=sink_tokens,
        obs_window=obs_window, pool_kernel=7, pooling="maxpool")
    return PhaseKVCache.from_dense_prefill(
        prefill.past_key_values, prompt_len=prompt_len, keep_indices=keep,
        prompt_bits=prompt_bits, cold_bits=cold_bits, group_size=group_size,
        recent_window=recent_window, lazy_g=lazy_g, decode_keep_ratio=decode_keep_ratio,
        optimizations=optimizations)


def storage_evidence(cache):
    if isinstance(cache, PackedKIVICache):
        packs = [p for layer in cache.layers for p in (layer.key, layer.value) if p is not None]
        residuals = [t for layer in cache.layers for t in (layer.key_residual, layer.value_residual)]
    else:
        packs = [p for layer in cache.layers
                 for segment in (layer.prompt, *layer.prompt_extensions, *layer.cold)
                 for p in (segment.key, segment.value)]
        residuals = [t for layer in cache.layers for t in (layer.hot_k, layer.hot_v)]
    if any(p.packed.dtype != torch.uint8 for p in packs):
        raise AssertionError("Packed payload must be uint8")
    if hasattr(cache, "key_cache") or hasattr(cache, "value_cache"):
        raise AssertionError("Packed cache exposes an unexpected dense KV shadow")
    if isinstance(cache, PhaseKVCache):
        evidence = cache.storage_breakdown()
        if evidence["physical_bytes"] != cache.physical_nbytes():
            raise AssertionError("Owned storage and physical_nbytes disagree")
        return {"cache_class": type(cache).__name__, "code_dtype": "torch.uint8",
                **evidence, "attention": "Triton reads packed codes; no full KV dequantization",
                "dense_shadow": False}
    code_bytes = sum(p.packed.untyped_storage().nbytes() for p in packs)
    metadata_bytes = sum(p.scale.untyped_storage().nbytes() + p.mins.untyped_storage().nbytes() for p in packs)
    residual_bytes = sum(t.untyped_storage().nbytes() for t in residuals)
    total = code_bytes + metadata_bytes + residual_bytes
    if total != cache.physical_nbytes():
        raise AssertionError(f"Reported packed bytes differ from tensor storage: {total} != {cache.physical_nbytes()}")
    return {"cache_class": type(cache).__name__, "code_dtype": "torch.uint8",
            "code_bytes": code_bytes, "metadata_bytes": metadata_bytes,
            "fp16_residual_capacity_bytes": residual_bytes, "physical_bytes": total,
            "attention": "Triton reads packed codes; no full KV dequantization",
            "dense_shadow": False}


@torch.inference_mode()
def generate_packed(model, tokenizer, input_ids, *, method, params, max_new_tokens,
                    trace_compression=False, stop_on_eos=True,
                    install_attention, build_cache, eos_token_ids=None):
    install_attention(model)
    eos_ids = set(eos_token_ids if eos_token_ids is not None else [tokenizer.eos_token_id])
    ids = input_ids.to("cuda")
    prompt_len = ids.shape[1]
    positions = torch.arange(prompt_len, device=ids.device)
    torch.cuda.reset_peak_memory_stats()
    prefill = model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True,
                    output_attentions=(method == "phasekv"), position_ids=positions.unsqueeze(0),
                    cache_position=positions)
    trace = None
    if trace_compression:
        from eval.compression_trace import CompressionTrace, cache_state
        trace = CompressionTrace(method)
        before = cache_state(prefill.past_key_values)
    if method == "kivi":
        cache = PackedKIVICache.from_dense_prefill(prefill.past_key_values,
                                                  bits=params.get("bits", 2),
                                                  group_size=params.get("group_size", 32),
                                                  residual_length=params.get("recent_window", 64))
    elif method == "phasekv":
        options = {"prompt_bits": 2, "cold_bits": 4, "prefill_keep_ratio": 0.3,
                   "decode_keep_ratio": 1., "sink_tokens": 0, "obs_window": 32,
                   "recent_window": 32, "lazy_g": 16, "group_size": 32}
        options.update(params)
        cache = build_cache(prefill, model, prompt_len, **options)
    else:
        raise ValueError(method)
    if trace:
        trace.implementation = ({"bits": cache.bits, "group_size": cache.group_size,
                                 "residual_length": cache.residual_length,
                                 "revision": "independent_kv_residual_packed_v1"}
                                if method == "kivi" else options)
        trace.observe("prefill", 0, before, cache_state(cache))
    def choose(logits):
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"Non-finite {method} logits; benchmark aborted")
        return logits.argmax(-1)

    next_token = choose(prefill.logits[:, -1])
    generated = [int(next_token.item())]
    del prefill
    for step in range(max_new_tokens - 1):
        if stop_on_eos and generated[-1] in eos_ids:
            break
        before = cache_state(cache) if trace else None
        output = model(input_ids=next_token.reshape(1, 1), past_key_values=cache,
                       use_cache=True, output_attentions=False,
                       position_ids=torch.tensor([[prompt_len + step]], device=ids.device),
                       cache_position=torch.tensor([cache.get_seq_length()], device=ids.device))
        cache.finish_decode_step()
        if trace:
            trace.observe("decode", step + 1, before, cache_state(cache), appended_tokens=1)
        next_token = choose(output.logits[:, -1])
        generated.append(int(next_token.item()))
        del output
    evidence = storage_evidence(cache)
    return {"generated_ids": generated, "text": tokenizer.decode(generated, skip_special_tokens=True),
            "prompt_len": prompt_len, "full_cache_bytes": evidence["physical_bytes"],
            "final_cache_bytes": evidence["physical_bytes"],
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "storage_evidence": evidence,
            **({"compression_trace": trace.export()} if trace else {})}
