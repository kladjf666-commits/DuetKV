"""Llama adapters. Native compression/attention code lives in runtime/, not here.

Only batch=1, no padding, greedy decode, explicit absolute positions are supported.
Never import the repository's historical KIVI/SnapKV/SCOPE replacements.
"""
from pathlib import Path
import importlib.util
import json
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime'
sys.path.insert(0, str(RUNTIME))


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snap_flash(self, q, k, v, mask, query_length, dropout=0., softmax_scale=None):
    """Restore the old bound FA2 helper for the supported unpadded path."""
    from flash_attn import flash_attn_func
    if mask is not None:
        raise ValueError('Native deployment supports unpadded equal-length rows only')
    return flash_attn_func(q, k, v, dropout_p=dropout,
                          softmax_scale=softmax_scale, causal=self.is_causal)


def scope_schedule(options, output_limit):
    options = dict(options)
    recent = options['decoding_recent_size']
    delta = (output_limit - recent) // (options['decoding_window_size'] - recent)
    # Upstream adaptive/discontinuous divide by delta. In the <=budget regime
    # there is no valid paper schedule; retain generated KV and label explicitly.
    if options['decoding_metric'] in ('adaptive', 'discontinuous') and delta <= 0:
        options['decoding_metric'] = 'None'
        options['delta'] = 1
        mode = 'prefill_only_short_output_boundary'
    else:
        options['delta'] = max(1, delta)
        mode = 'native_' + options['decoding_metric']
    return options, mode


def install(model, method, options):
    provenance = []
    if method == 'snapkv':
        sys.path.insert(0, str(RUNTIME / 'SnapKV'))
        module = module_at('native_snap_llama', RUNTIME / 'SnapKV/snapkv/monkeypatch/llama_hijack_4_37.py')
        for layer in model.model.layers:
            layer.self_attn.forward = MethodType(module.llama_flash_attn2_forward, layer.self_attn)
            layer.self_attn._flash_attention_forward = MethodType(snap_flash, layer.self_attn)
        provenance.append(str(Path(module.__file__).relative_to(ROOT)))
    elif method.startswith('scope'):
        sys.path.insert(0, str(RUNTIME / 'SCOPE'))
        module = module_at('native_scope_llama', RUNTIME / 'SCOPE/model/llama_model.py')
        for layer in model.model.layers:
            layer.self_attn.forward = MethodType(module.llama_flash_attn2_forward_H2O, layer.self_attn)
        provenance.append(str(Path(module.__file__).relative_to(ROOT)))
    elif method == 'phasekv':
        module = module_at('native_phase_llama', RUNTIME / 'phasekv_llama_attention.py')
        for layer in model.model.layers:
            # Same packed wrapper; dense prefix delegated to native HF FA2.
            layer.self_attn = module.PhaseKVLlamaAttention(layer.self_attn)
        provenance.append('runtime/phasekv_llama_attention.py')
    for key, value in options.items():
        setattr(model.config, key, value)
    return provenance


def load_model(model_path, method, *, tiny=False):
    import torch
    import transformers
    from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM
    config = json.loads((ROOT / 'config.json').read_text('utf-8'))
    if transformers.__version__ != config['transformers']:
        raise RuntimeError('Use the isolated setup_linux.sh environment: Transformers version mismatch')
    if torch.__version__.split('+')[0] != config['torch']:
        raise RuntimeError('PyTorch version mismatch; use setup_linux.sh')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU/fake-quant fallback is forbidden')
    if tiny:
        cfg = LlamaConfig(vocab_size=256, hidden_size=512, intermediate_size=1024,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=131072, rope_theta=500000.,
                          rope_scaling={'rope_type': 'llama3', 'factor': 8.,
                                        'low_freq_factor': 1., 'high_freq_factor': 4.,
                                        'original_max_position_embeddings': 8192},
                          pad_token_id=0, bos_token_id=1, eos_token_id=2)
    else:
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        if (cfg.model_type != 'llama' or cfg.hidden_size != 4096 or cfg.num_hidden_layers != 32
                or cfg.num_attention_heads != 32 or cfg.num_key_value_heads != 8
                or (cfg.rope_scaling or {}).get('rope_type', (cfg.rope_scaling or {}).get('type')) != 'llama3'):
            raise ValueError('Expected Llama-3.1-8B architecture and original llama3 RoPE configuration')
    cfg.torch_dtype = torch.float16
    cfg._attn_implementation = 'flash_attention_2'
    cfg.pretraining_tp = 1
    options = config['methods'][method]
    cls = LlamaForCausalLM
    origin = []
    if method == 'kivi':
        sys.path.insert(0, str(RUNTIME / 'KIVI'))
        native = module_at('native_kivi_llama', RUNTIME / 'KIVI/models/llama_kivi.py')
        cls = native.LlamaForCausalLM_KIVI
        for key, value in options.items():
            setattr(cfg, key, value)
        # Upstream legacy model reads this flag to select a 2-D flash mask.
        cfg._flash_attn_2_enabled = True
        origin.append('runtime/KIVI/models/llama_kivi.py')
    if tiny:
        # Explicit dtype at construction also satisfies HF FlashAttention checks.
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        try:
            model = cls(cfg).to(device='cuda', dtype=torch.float16).eval()
        finally:
            torch.set_default_dtype(old)
    else:
        model = cls.from_pretrained(model_path, config=cfg, torch_dtype=torch.float16,
                                    attn_implementation='flash_attention_2',
                                    local_files_only=True, low_cpu_mem_usage=True).to('cuda').eval()
    wrong = [(n, str(p.dtype), str(p.device)) for n, p in model.named_parameters()
             if p.dtype != torch.float16 or p.device.type != 'cuda']
    if wrong:
        raise RuntimeError(f'FP16/GPU-only contract violated: {wrong[:3]}')
    origin += install(model, method, options)
    return model, options, origin


def reset_sample(model, method, options, output_limit):
    effective = dict(options)
    mode = 'native'
    if method.startswith('scope'):
        effective, mode = scope_schedule(options, output_limit)
        from model.kv_utils import H2OKVCluster
        H2OKVCluster.current_decoding_step = 0
        H2OKVCluster.jump_step = 0
        H2OKVCluster.jump_layer = 0
    for key, value in effective.items():
        setattr(model.config, key, value)
    for layer in model.model.layers:
        attention = layer.self_attn
        if hasattr(attention, 'kv_cluster'):
            del attention.kv_cluster
        if method == 'snapkv' or method.startswith('scope'):
            attention.kv_seq_len = 0
    return effective, mode


def forward(model, ids, cache, start, method, **kwargs):
    import torch
    positions = torch.arange(start, start + ids.shape[1], device=ids.device)
    if method != 'kivi':
        kwargs['cache_position'] = positions
    result = model.model(input_ids=ids, past_key_values=cache, position_ids=positions[None],
                         use_cache=True, return_dict=True, **kwargs)
    # All methods project only the final hidden state. Avoid full N x vocab logits.
    logits = model.lm_head(result.last_hidden_state[:, -1:, :]).float()
    return logits, result.past_key_values, result.attentions


def phase_prefill(model, ids, options):
    """Native dense FA2 prefix + native eager last-observation chunk for scores.

    PhaseKV selects only the final 32 query rows, so no N x N tensor is required.
    Explicit causal mask preserves absolute-position semantics in the tail.
    """
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaAttention
    from PhaseKV.phasekv_cache import PhaseKVCache
    from PhaseKV.prompt_tier import select_prompt_indices
    length = ids.shape[1]
    observation = min(options['obs_window'], length)
    start = length - observation
    cache = DynamicCache()
    if start:
        _, cache, _ = forward(model, ids[:, :start], cache, 0, 'phasekv')
    originals = [layer.self_attn.original for layer in model.model.layers]
    saved = [attention.forward for attention in originals]
    mask = torch.zeros((1, 1, observation, length), device=ids.device, dtype=torch.float16)
    future = torch.arange(length, device=ids.device)[None] > torch.arange(start, length, device=ids.device)[:, None]
    mask.masked_fill_(future[None, None], torch.finfo(torch.float16).min)
    try:
        for attention in originals:
            attention.forward = MethodType(LlamaAttention.forward, attention)
        logits, dense, attentions = forward(model, ids[:, start:], cache, start, 'phasekv',
                                             attention_mask=mask, output_attentions=True)
    finally:
        for attention, previous in zip(originals, saved):
            attention.forward = previous
    keep = select_prompt_indices(attentions, prompt_len=length, model_config=model.config,
                                 prefill_keep_ratio=options['prefill_keep_ratio'],
                                 obs_window=options['obs_window'], sink_tokens=options['sink_tokens'],
                                 pool_kernel=7, pooling='maxpool')
    packing = {k: options[k] for k in ('prompt_bits', 'cold_bits', 'group_size',
                                      'recent_window', 'lazy_g', 'decode_keep_ratio', 'optimizations')}
    packed = PhaseKVCache.from_dense_prefill(dense, prompt_len=length, keep_indices=keep, **packing)
    return logits, packed


def cache_report(cache):
    """Physical unique storage, including view backing storage; never theory bytes."""
    import torch
    if hasattr(cache, 'storage_breakdown'):
        tensors = [tensor for _, tensor in cache.storage_tensors()]
        if any(t.is_floating_point() and t.dtype != torch.float16 for t in tensors):
            raise RuntimeError('Non-FP16 PhaseKV residual/quantization metadata storage')
        return {**cache.storage_breakdown(), 'tensor_dtypes': sorted({str(t.dtype) for t in tensors})}
    tensors = []

    def walk(value):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
    if hasattr(cache, 'key_cache'):
        walk(cache.key_cache)
        walk(cache.value_cache)
    else:
        walk(cache)
    if any(t.is_floating_point() and t.dtype != torch.float16 for t in tensors):
        raise RuntimeError('Non-FP16 native KV/quantization metadata storage')
    seen = set()
    size = 0
    for tensor in tensors:
        identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
        if identity not in seen:
            seen.add(identity)
            size += tensor.untyped_storage().nbytes()
    return {'physical_bytes': size, 'auxiliary_bytes': 0, 'total_owned_bytes': size,
            'tensor_dtypes': sorted({str(t.dtype) for t in tensors})}


def cache_trace(cache, method):
    """Cheap compression trigger evidence, kept outside timed decode."""
    if method == 'phasekv':
        layer = cache.layers[0]
        return {'prompt_tokens': layer.prompt.length, 'hot_tokens': layer.hot_len,
                'cold_segments': len(layer.cold), 'physical_tokens': layer.physical_len}
    if method == 'kivi':
        layer = cache[0]
        return {'key_quantized': layer[0] is not None, 'value_quantized': layer[4] is not None,
                'logical_tokens': layer[-1]}
    return {'physical_tokens': cache.key_cache[0].shape[-2],
            'stored_heads': cache.key_cache[0].shape[1]}
