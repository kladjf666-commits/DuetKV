import torch
from PhaseKV.hetero_cache import cache_num_layers, num_kv_heads_of, evict_per_head, observation_votes, pool_scores, topk_per_head

def topk_keep_per_head(scores,select_k,prefix_len,recent_lo,recent_hi):
    assert scores.shape[-1]==prefix_len
    top=topk_per_head(scores,select_k)
    tail=torch.arange(recent_lo,recent_hi,device=scores.device).expand(*top.shape[:2],-1)
    return torch.cat((top,tail),dim=-1)
