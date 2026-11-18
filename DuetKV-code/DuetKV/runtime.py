"""Llama model loading and absolute-position forwards for DuetKV."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def load_model(model_path, method, *, tiny=False):
    import torch
    import transformers
    from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM
    config = json.loads((ROOT / 'config.json').read_text('utf-8'))
    if transformers.__version__ != config['transformers']:
        raise RuntimeError('Transformers version mismatch')
    if torch.__version__.split('+')[0] != config['torch']:
        raise RuntimeError('PyTorch version mismatch')
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
    if method != 'duetkv':
        raise ValueError(f'Unknown method: {method}')
    options = config['methods'][method]
    if tiny:
        # Explicit dtype at construction also satisfies HF FlashAttention checks.
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        try:
            model = LlamaForCausalLM(cfg).to(device='cuda', dtype=torch.float16).eval()
        finally:
            torch.set_default_dtype(old)
    else:
        model = LlamaForCausalLM.from_pretrained(model_path, config=cfg, torch_dtype=torch.float16,
                                    attn_implementation='flash_attention_2',
                                    local_files_only=True, low_cpu_mem_usage=True).to('cuda').eval()
    wrong = [(n, str(p.dtype), str(p.device)) for n, p in model.named_parameters()
             if p.dtype != torch.float16 or p.device.type != 'cuda']
    if wrong:
        raise RuntimeError(f'FP16/GPU-only contract violated: {wrong[:3]}')
    from .llama_attention import DuetKVLlamaAttention
    for layer in model.model.layers:
        layer.self_attn = DuetKVLlamaAttention(layer.self_attn)
    reset_sample(model, method, options, None)
    return model, options, ['DuetKV/llama_attention.py']


def reset_sample(model, method, options, output_limit):
    if method != 'duetkv':
        raise ValueError(f'Unknown method: {method}')
    for key, value in options.items():
        setattr(model.config, key, value)
    return dict(options), 'native'


def forward(model, ids, cache, start, method, **kwargs):
    import torch
    positions = torch.arange(start, start + ids.shape[1], device=ids.device)
    kwargs['cache_position'] = positions
    result = model.model(input_ids=ids, past_key_values=cache, position_ids=positions[None],
                         use_cache=True, return_dict=True, **kwargs)
    # All methods project only the final hidden state. Avoid full N x vocab logits.
    logits = model.lm_head(result.last_hidden_state[:, -1:, :]).float()
    return logits, result.past_key_values, result.attentions
