"""Score prompt keys using the actual packed teacher's decode queries and softmax denominators."""
import torch


def prompt_key_fp32(packed):
    from PhaseKV.kernels.packing import unpack_low_bits
    if packed.axis != 2 or packed.bits != 4 or packed.orig_shape[0] != 1:
        raise ValueError('Expected B=1, 4-bit, per-channel prompt keys')
    n = packed.axis_len
    codes = unpack_low_bits(packed.packed[0], packed.bits, n)
    groups = torch.arange(n, device=codes.device) // packed.group_size
    scale = packed.scale[0].float().index_select(-1, groups)
    mins = packed.mins[0].float().index_select(-1, groups)
    return (codes.float() * scale + mins).transpose(1, 2).contiguous()


def future_scores(keys, queries, log_z):
    # keys [kv_head, position, d], queries [step, query_head, d].
    heads, length, dim = keys.shape
    steps, qheads, _ = queries.shape
    scores = keys.new_zeros(heads, length)
    for start in range(0, steps, 32):
        q = queries[start:start + 32].float()
        z = log_z[start:start + 32]
        count = q.shape[0]
        q = q.reshape(count, heads, -1, dim).transpose(0, 1).reshape(heads, -1, dim)
        logits = torch.bmm(q, keys.transpose(1, 2)) * (dim ** -.5)
        z = z.reshape(count, heads, -1).transpose(0, 1).reshape(heads, -1, 1)
        scores += (logits - z).exp().sum(dim=1)
    mass = scores.sum(dim=-1)
    if not (torch.isfinite(mass).all() and (mass > 0).all()
            and (mass <= 1.002 * steps * (qheads // heads)).all()):
        raise RuntimeError('Invalid prompt probability mass')
    return scores


class FutureTrace:
    def __init__(self, layers, length, max_steps):
        self.layers = layers
        self.length = length
        self.max_steps = max_steps
        self.buffers = []
        self.calls = 0

    def observe(self, q, tiers):
        layer = self.calls % self.layers
        step = self.calls // self.layers
        if step >= self.max_steps or tiers.prompt.length != self.length:
            raise RuntimeError('Unexpected teacher cache or decode step')
        if step == 0:
            queries = q.new_empty(self.max_steps, *q.shape)
            log_z = torch.empty(self.max_steps, q.shape[0], dtype=torch.float32, device=q.device)
            self.buffers.append((tiers.prompt.key, queries, log_z))
        key, queries, log_z = self.buffers[layer]
        if key is not tiers.prompt.key:
            raise RuntimeError('Teacher layer order changed')
        ws = tiers._decode_workspace
        n = ws.partial_count
        maxima, den = ws.maxima[:, :n], ws.denominators[:, :n]
        # Use every live partial (prompt + hot + cold), before the workspace is reused.
        log_z[step] = torch.logsumexp(maxima + den.log(), dim=-1)
        queries[step].copy_(q)
        self.calls += 1

    def finish(self, steps):
        if self.calls != steps * self.layers or not steps:
            raise RuntimeError('Expected one observed query per decode step and layer')
        result = []
        for key, q, z in self.buffers:
            result.append(future_scores(prompt_key_fp32(key), q[:steps], z[:steps]))
        self.buffers.clear()
        return result


def choose(scores, original, tail):
    # Stable sorting makes original position the tie-breaker.
    budget, length = original.shape[-1], scores.shape[-1]
    if original.shape[:2] != (1, scores.shape[0]) or not 0 < tail <= budget <= length:
        raise ValueError('Invalid mask budget or head shape')
    selected = scores[:, :length-tail].argsort(descending=True, stable=True)[:, :budget-tail]
    recent = torch.arange(length-tail, length, device=scores.device).expand(scores.shape[0], -1)
    keep = torch.cat([selected, recent], dim=-1).sort().values
    total = scores.sum(-1)
    before = scores.gather(1, original[0]).sum(-1) / total
    after = scores.gather(1, keep).sum(-1) / total
    if not (after + 1e-5 >= before).all():
        raise RuntimeError('Future mask lost more teacher mass than standard mask')
    return keep.unsqueeze(0), torch.stack([before, after])
