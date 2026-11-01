"""Equal-length, independent sequences sharing model weights on one GPU."""
from dataclasses import replace

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from ablation_attention import AblationAttention
from ablation_kernels import segmented_phasekv_decode
from PhaseKV.phasekv_cache import LayerTiers, PackedKVSegment, PhaseKVCache


class BatchAttention(AblationAttention):
    def forward(self, hidden_states, **kwargs):
        batch, length, _ = hidden_states.shape
        if batch == 1 or length != 1:
            return super().forward(hidden_states, **kwargs)
        cache = kwargs['past_key_value']
        base = self.original
        if self.training or kwargs.get('attention_mask') is not None or not kwargs['use_cache']:
            raise ValueError('Requires unpadded equal-length inference')
        hq, hk, hd = base.num_heads, base.num_key_value_heads, base.head_dim
        q = base.q_proj(hidden_states).reshape(batch, hq, 1, hd)
        k = base.k_proj(hidden_states).reshape(batch, hk, 1, hd)
        v = base.v_proj(hidden_states).reshape(batch, hk, 1, hd)
        q, k = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
        if self.method == 'full_fa2':
            from flash_attn import flash_attn_func
            k, v = cache.update(k, v, self.layer_idx)
            out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                  dropout_p=0., causal=True)
        elif self.method == 'phasekv_triton':
            # Consecutive query and KV head groups belong to the same sequence.
            # All B*Hq heads are scheduled by a single segmented attention launch.
            cache.append_hot(self.layer_idx, k.reshape(1, batch*hk, 1, hd), v.reshape(1, batch*hk, 1, hd))
            out = segmented_phasekv_decode(q.reshape(batch*hq, hd), cache.layer_tiers(self.layer_idx), batch*hq)
        else:
            raise ValueError(self.method)
        return base.o_proj(out.reshape(batch, 1, hq*hd)), None, cache


def merge_prefills(caches, phase, options):
    """Join independent prompt caches before timing. Consume inputs layer by layer."""
    if len(caches) == 1:
        return caches[0]
    if not phase:
        merged = DynamicCache()
        for i in range(len(caches[0].key_cache)):
            merged.update(torch.cat([c.key_cache[i] for c in caches]), torch.cat([c.value_cache[i] for c in caches]), i)
            for c in caches:
                c.key_cache[i] = c.value_cache[i] = None
        return merged

    def join(packed, channel_keys=False):
        item = packed[0]
        shape = item.orig_shape
        if any(p.orig_shape != shape or p.bits != item.bits or p.group_size != item.group_size for p in packed):
            raise ValueError('All sequences need matching packed geometry')
        fields = {}
        for name in ('packed', 'scale', 'mins'):
            tensors = [getattr(p, name) for p in packed]
            if channel_keys:
                # Concatenate in physical order. A plain logical cat can silently
                # revert K to token-contiguous storage and disable the fast path.
                fields[name] = torch.cat([t.transpose(-2, -1).contiguous() for t in tensors],
                                         dim=1).contiguous().transpose(-2, -1)
            else:
                fields[name] = torch.cat(tensors, dim=1)
        return replace(item, orig_shape=(1, len(packed)*shape[1], *shape[2:]), **fields)

    seen = caches[0].true_seq_length
    layers = []
    for i in range(len(caches[0].layers)):
        parts = [c.layers[i] for c in caches]
        if any(p.hot_len or p.cold or p.prompt_extensions for p in parts):
            raise ValueError('Only freshly prefilled cache rows can be folded')
        channel_keys = options['optimizations'].get('channel_contiguous_k', False)
        prompt = type(parts[0].prompt)(join([p.prompt.key for p in parts], channel_keys),
                                      join([p.prompt.value for p in parts]))
        hot_shape = (1, len(parts)*parts[0].hot_k.shape[1], *parts[0].hot_k.shape[2:])
        hot = parts[0].hot_k.new_empty(hot_shape)
        layers.append(type(parts[0])(prompt, hot_k=hot, hot_v=torch.empty_like(hot)))
        for c in caches:
            c.layers[i] = None
    return type(caches[0])(layers, seen_tokens=seen, **{k: options[k] for k in (
        'prompt_bits', 'cold_bits', 'group_size', 'recent_window', 'lazy_g', 'decode_keep_ratio', 'optimizations')})
