"""B-local: replace the tail votes in B with causal, within-block attention votes."""
from contextlib import ExitStack
from functools import partial
from unittest.mock import patch

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from block70 import BLOCK_SIZE
from block_budget import block_topk
from future_oracle import campaign, launch
from quality_compare import generate_one


def local_votes(q, k, block_size):
    """Post-RoPE [1,H,N,D]. Sum GQA siblings; average over visible local queries."""
    _, heads, n, dim = q.shape
    kv_heads = k.shape[1]
    group = heads // kv_heads
    keys = k[0].float().transpose(1, 2)
    positions = torch.arange(n, device=q.device)
    votes = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        width = end - start
        local = positions[start:end]
        query = q[0, :, start:end].reshape(kv_heads, group*width, dim).float()
        logits = torch.bmm(query, keys[:, :, :end]).view(heads, width, end)
        logits.mul_(dim ** -.5)
        local_logits = logits[:, :, start:end]
        local_logits.masked_fill_(local[None, :] > local[:, None], -torch.inf)
        # Earlier blocks participate in the denominator but cast no votes in this block.
        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
        mass = (local_logits - log_z).exp().reshape(kv_heads, group, width, width).sum((1, 2))
        votes.append(mass / (end - local))
    return torch.cat(votes, dim=-1).unsqueeze(0)


def measure(model, options, tokenizer, sample, unused_control):
    from PhaseKV import hetero_cache as hc

    n = len(sample['input_ids'])
    candidates, budget = n-32, round(.70*n)
    if candidates <= 32 or (options['obs_window'], options['sink_tokens'], options['prefill_keep_ratio']) != (32, 0, .7):
        raise ValueError('Unexpected input for the frozen B-local protocol')
    layers = len(model.model.layers)
    votes, times, overlap = [], [], []
    original_topk = hc.topk_per_head

    def collect(attention, args, kwargs):
        hidden = kwargs['hidden_states']
        if hidden.shape[1] != candidates:
            return
        if attention.layer_idx != len(votes):
            raise RuntimeError('Unexpected dense prefill layer order')
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        def project(projection, heads):
            return projection(hidden).view(1, candidates, heads, attention.head_dim).transpose(1, 2)
        q = project(attention.q_proj, attention.num_heads)
        k = project(attention.k_proj, attention.num_key_value_heads)
        q, k = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
        votes.append(hc.pool_scores(local_votes(q, k, BLOCK_SIZE), 7, 'maxpool'))
        end.record()
        times.append((begin, end))

    def select(tail_scores, count):
        if count != budget-32 or len(votes) != layers:
            raise RuntimeError('Unexpected candidate budget or missing local votes')
        scores = votes[len(overlap)]
        if scores.shape != tail_scores.shape or not torch.isfinite(scores).all():
            raise RuntimeError('Invalid local scores')
        local_keep = block_topk(scores, original_topk(scores, count), BLOCK_SIZE)
        b_keep = block_topk(tail_scores, original_topk(tail_scores, count), BLOCK_SIZE)
        # Independently verify identical quotas in every head and candidate block.
        edges = torch.arange(BLOCK_SIZE, candidates, BLOCK_SIZE, device=scores.device)
        edges = edges.expand(*local_keep.shape[:-1], -1).contiguous()
        def cumulative(keep):
            return torch.searchsorted(keep.sort().values, edges)
        if not torch.equal(cumulative(local_keep), cumulative(b_keep)):
            raise RuntimeError('B-local changed a block quota')
        shared = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, b_keep, True)
        overlap.append((32 + shared.gather(-1, local_keep).sum(-1)).cpu())
        return local_keep

    with ExitStack() as stack:
        for layer in model.model.layers:
            hook = layer.self_attn.original.register_forward_pre_hook(collect, with_kwargs=True)
            stack.callback(hook.remove)
        stack.enter_context(patch.object(hc, 'topk_per_head', select))
        row = generate_one(model, options, sample, None, tokenizer, 'prompt70')
    if len(overlap) != layers or row['retained_per_head'] != budget:
        raise RuntimeError('Incomplete B-local selection')
    row.update(arm='block_local70', quota_verified_layer_heads=layers*model.config.num_key_value_heads,
               mask_overlap_with_B=float(torch.stack(overlap).float().mean()/budget),
               local_vote_gpu_ms=sum(a.elapsed_time(b) for a,b in times))
    return row


if __name__ == '__main__':
    launch(partial(campaign, candidate='block_local70', run_candidate=measure,
                   archived=('prompt70','prompt85','prompt100','block70'), reference='local_reference.json.gz',
                   policy=dict(block_size=BLOCK_SIZE, selector='Causal full-prefix softmax; within-block '
                               'votes including self, divided by visible local query count; original width-7 '
                               'pool over candidates and B quotas/tail/tie policy.',
                               primary='Paired indexed accuracy difference, B-local minus archived B.')))
