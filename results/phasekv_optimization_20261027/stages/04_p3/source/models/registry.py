"""Explicit profiles and architecture integrations; no silent model substitution."""
from importlib import import_module
import json
from pathlib import Path

MODELS_ROOT = Path(__file__).resolve().parent
FAMILIES = {'qwen2': 'models.Qwen2.integration', 'llama': 'models.Llama.integration'}


def profile_names():
    return sorted(p.parent.name for p in MODELS_ROOT.glob('*/profile.json'))


def load_profile(name):
    if name not in profile_names():
        raise ValueError(f'Unknown model profile {name!r}; available: {profile_names()}')
    folder = MODELS_ROOT / name
    profile = json.loads((folder/'profile.json').read_text())
    if profile['name'] != name or profile['family'] not in FAMILIES:
        raise ValueError('Profile name/family mismatch or unsupported architecture')
    if (profile['dtype'], profile['attention'], profile['batch_size']) != ('bfloat16','eager',1):
        raise ValueError('Current reproducible deployment requires BF16/eager/batch=1')
    profile['architecture'] = json.loads((folder/profile['architecture_file']).read_text())
    profile['compression'] = json.loads((folder/profile['methods_file']).read_text())
    validate_methods(profile, profile['compression']['methods'])
    return profile


def integration(profile):
    return import_module(FAMILIES[profile['family']])


def validate_methods(profile, methods):
    supported = integration(profile).SUPPORTED_METHODS
    names = [s['method'] for s in methods]
    if not names or len(set(names)) != len(names) or any(n not in supported for n in names):
        raise ValueError(f'Choose unique supported methods: {supported}')
    # Fail before loading weights if a config would be ignored or unsupported.
    from eval.compressors import build_compressor
    for spec in methods:
        name = spec['method']
        p = {k:v for k,v in spec.items() if k != 'method'}
        if name in ('full','snapkv','scope','pyramidkv'):
            allowed = {'full':set(), 'snapkv':{'max_cache_tokens','prefill_keep_ratio','obs_window','pool_kernel','pooling'},
                       'scope':{'max_cache_tokens','recent_window','decoding_metric'},
                       'pyramidkv':{'max_cache_tokens','prefill_keep_ratio','obs_window','pool_kernel','pooling','beta'}}[name]
            if set(p)-allowed: raise ValueError(f'Unsupported {name} parameters: {set(p)-allowed}')
            build_compressor(name, **p)
        elif name == 'kivi':
            if set(p)-{'bits','group_size','recent_window'}: raise ValueError('Unknown KIVI parameters')
            b,g,r = p.get('bits',2),p.get('group_size',32),p.get('recent_window',64)
            if b not in (2,4,8) or not isinstance(g,int) or not isinstance(r,int) or min(g,r)<1 or r%g or g%(8//b):
                raise ValueError('KIVI bit/group/residual configuration is invalid')
        else:
            allowed={'prompt_bits','cold_bits','prefill_keep_ratio','decode_keep_ratio','sink_tokens','obs_window','recent_window','lazy_g','group_size'}
            if set(p)-allowed: raise ValueError(f'Unknown PhaseKV parameters: {set(p)-allowed}')
            for key in ('prompt_bits','cold_bits'):
                if p.get(key,2) not in (2,4,8): raise ValueError('PhaseKV requires packed 2/4/8 bits')
            for key in ('group_size','recent_window','lazy_g','obs_window'):
                if not isinstance(p.get(key,32),int) or p.get(key,32)<1: raise ValueError(f'Invalid {key}')
            for key in ('prefill_keep_ratio','decode_keep_ratio'):
                if not 0 < p.get(key,1) <= 1: raise ValueError(f'Invalid {key}')
            if not isinstance(p.get('sink_tokens',0),int) or p.get('sink_tokens',0)<0:
                raise ValueError('sink_tokens must be a nonnegative integer')
            from PhaseKV.phasekv_cache import _validate_config
            _validate_config(p.get('prompt_bits',2),p.get('cold_bits',4),p.get('group_size',32),
                             p.get('recent_window',32),p.get('lazy_g',16),p.get('decode_keep_ratio',1.0))
        if name in ('kivi','phasekv'):
            cfg=profile['architecture']; hd=cfg['hidden_size']//cfg['num_attention_heads']
            if hd % p.get('group_size',32): raise ValueError('KV group size must divide head dimension')


def validate_model_config(profile, config):
    expected = profile['architecture']
    keys=('model_type','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads',
          'num_key_value_heads','vocab_size','tie_word_embeddings','max_position_embeddings')
    if profile['family'] == 'llama':
        keys += ('pretraining_tp','attention_bias','mlp_bias','rope_theta','rms_norm_eps','hidden_act')
    mismatches={k:(expected[k],getattr(config,k,None)) for k in keys if expected[k]!=getattr(config,k,None)}
    if profile['family'] == 'llama':
        def normalized_rope(value):
            if value is None:
                return None
            value = dict(value)
            if 'type' in value:
                value.setdefault('rope_type', value['type'])
                del value['type']
            return value
        rope = normalized_rope(getattr(config, 'rope_scaling', None))
        if rope != normalized_rope(expected['rope_scaling']):
            mismatches['rope_scaling'] = (expected['rope_scaling'], rope)
        hd = expected['hidden_size'] // expected['num_attention_heads']
        if getattr(config, 'head_dim', hd) != hd:
            mismatches['head_dim'] = (hd, config.head_dim)
    if mismatches: raise ValueError(f'Model architecture differs from named profile: {mismatches}')
    if getattr(config,'quantization_config',None):
        raise ValueError('Weight-quantized checkpoints are not part of this BF16 comparison')


def estimated_parameter_count(config):
    """Qwen2/Llama parameter count with family-specific biases and tied embeddings."""
    family = config['model_type']
    if family not in ('qwen2', 'llama'):
        raise ValueError(f'No parameter-count rule for {family}')
    h=config['hidden_size']; kv=config['num_key_value_heads']*(h//config['num_attention_heads'])
    embed=config['vocab_size']*h*(1 if config['tie_word_embeddings'] else 2)
    bias = h+2*kv if family == 'qwen2' else (2*h+2*kv)*bool(config.get('attention_bias', False))
    if family == 'llama' and config.get('mlp_bias', False):
        bias += 2*config['intermediate_size'] + h
    layer=2*h*h + 2*h*kv + bias + 3*h*config['intermediate_size'] + 2*h
    return embed + config['num_hidden_layers']*layer + h
