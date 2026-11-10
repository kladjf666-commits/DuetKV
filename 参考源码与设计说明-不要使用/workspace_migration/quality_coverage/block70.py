"""Run only the block-budget arm on the frozen PhaseKV quality benchmark."""
from functools import partial
from unittest.mock import patch

import torch

from block_budget import block_topk
from future_oracle import campaign, launch
from quality_compare import generate_one

BLOCK_SIZE = 128
POLICY = dict(block_size=BLOCK_SIZE, selector='Original tail-32 votes and width-7 max pooling. '
              'Reserve final 32, divide remaining candidates into consecutive 128-token blocks. '
              'Round cumulative proportional quotas to nearest (half up). Within equal scores '
              'prefer original global top-k members, then earlier positions.')


def measure(model, options, tokenizer, sample, unused_control):
    from PhaseKV import hetero_cache as hc

    n = len(sample['input_ids'])
    budget = round(.7*n)
    if (options['prefill_keep_ratio'], options['obs_window'], options['sink_tokens']) != (.7, 32, 0):
        raise ValueError('Unexpected prefill configuration')
    global_topk = hc.topk_per_head
    counts = []
    edges = [i*n//10 for i in range(11)]

    def allocate(scores, k):
        if scores.shape[-1] != n-32 or k != budget-32:
            raise ValueError('Unexpected candidate positions or budget')
        original = global_topk(scores, k)
        selected = block_topk(scores, original, BLOCK_SIZE)
        # Record both masks from the same pooled scores, before any decoding.
        masks = torch.stack((original[0], selected[0])).sort(dim=-1).values
        bounds = torch.tensor(edges, device=scores.device)
        cut = torch.searchsorted(masks, bounds.expand(*masks.shape[:-1], -1).contiguous())
        cut += (bounds - (n-32)).clamp_min(0)
        counts.append(cut.diff().cpu())
        return selected

    with patch.object(hc, 'topk_per_head', allocate):
        row = generate_one(model, options, sample, None, tokenizer, 'prompt70')
    if len(counts) != len(model.model.layers):
        raise RuntimeError('Expected one budget allocation per layer')
    totals = torch.stack(counts, dim=1)
    if not (totals.sum(-1) == budget).all():
        raise RuntimeError('Decile counts do not sum to the frozen prompt budget')
    row.update(arm='block70', position_coverage=dict(edges=edges, axes=['layer','kv_head','decile'],
               prompt70=totals[0].tolist(), block70=totals[1].tolist()))
    return row


if __name__ == '__main__':
    launch(partial(campaign, candidate='block70', run_candidate=measure,
                   reference='block_reference.json.gz', policy=POLICY))
