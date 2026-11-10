"""Proportional block quotas applied to the already pooled prompt scores."""
import torch


def block_topk(scores, original, block_size):
    """Keep the same per-head budget; preserve original winners on score ties."""
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError('block_size must be a positive integer')
    length, budget = scores.shape[-1], original.shape[-1]
    if length <= block_size or budget in (0, length):
        return original
    preferred = torch.zeros_like(scores, dtype=torch.uint8)
    preferred.scatter_(-1, original, 1)
    kept = []
    previous = 0
    for start in range(0, length, block_size):
        end = min(start + block_size, length)
        # Round cumulative proportional quotas to nearest integer, including a short last block.
        cumulative = (2 * budget * end + length) // (2 * length)
        count = cumulative - previous
        previous = cumulative
        if count:
            block = scores[..., start:end]
            priority = preferred[..., start:end].argsort(dim=-1, descending=True, stable=True)
            ranking = block.gather(-1, priority).argsort(dim=-1, descending=True, stable=True)
            kept.append(priority.gather(-1, ranking[..., :count]) + start)
    return torch.cat(kept, dim=-1)
