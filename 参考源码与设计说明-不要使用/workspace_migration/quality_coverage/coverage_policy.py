"""Equal-budget, question-aware selection for the frozen LongGenBench experiment."""
import re

import torch
from PhaseKV import hetero_cache


def question_units(tokenizer, ids, task):
    """Read only the visible prompt. Return disjoint content-token sets."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if encoded.ids != ids or text.count('Following Question:') != 1:
        raise ValueError('Cannot map this prompt to its original token positions')
    start = text.index('Following Question:') + len('Following Question:')
    questions = list(re.compile(r'^Question_(\d+):', re.M).finditer(text, start))
    numbers = [int(m[1]) for m in questions]
    if not numbers or numbers != list(range(numbers[0], numbers[0]+len(numbers))):
        raise ValueError('Nonconsecutive question numbers')
    offsets = encoded.offsets
    units = []
    used = set()

    def add(number, kind, label, a, b):
        # Boundary tokens can include leading spaces. Only the part inside
        # the body counts; option markers and question numbers are outside it.
        pieces = [(i, text[max(a,x):min(b,y)]) for i,(x,y) in enumerate(offsets) if x < b and y > a]
        positions = [i for i,part in pieces if any(c.isalnum() for c in part)]
        if not positions:  # symbol-only answer, e.g. a mathematical operator
            positions = [i for i,part in pieces if part.strip()]
        if not positions or used.intersection(positions):
            raise ValueError(f'Empty or overlapping content for question {number}, {label}')
        used.update(positions)
        n = len(positions)
        minimum = min(n, max(6 if kind == 'stem' else 3, (2*n+2)//3))
        units.append(dict(question=number, kind=kind, label=label, positions=positions, minimum=minimum))

    markers = re.compile(r'\(([A-E])\)(?=\s|$)')
    for i,(q,number) in enumerate(zip(questions,numbers)):
        end = questions[i+1].start() if i+1 < len(questions) else len(text)
        opts = [] if task == 'gsm8k' else list(markers.finditer(text,q.end(),end))
        labels = ''.join(m[1] for m in opts)
        if task == 'csqa' and labels != 'ABCDE' or task == 'mmlu' and labels not in ('ABCD',''):
            raise ValueError(f'Unexpected choices at question {number}: {labels}')
        add(number, 'stem', '', q.end(), opts[0].start() if opts else end)
        for j,m in enumerate(opts):
            add(number, 'option', m[1], m.end(), opts[j+1].start() if j+1 < len(opts) else end)
    return units


class CoveragePlan:
    def __init__(self, units, length, device):
        self.length = length
        width = max(len(u['positions']) for u in units)
        padded = [u['positions'] + [-1]*(width-len(u['positions'])) for u in units]
        idx = torch.tensor(padded, dtype=torch.long, device=device)
        self.valid = idx >= 0
        self.positions = idx.clamp_min(0)
        self.ranks = torch.arange(width, device=device)
        self.minimum = torch.tensor([u['minimum'] for u in units], device=device)
        self.stems = torch.tensor([u['kind']=='stem' for u in units], device=device)
        self.layers = []

    def apply(self, keep, attention, *, repair):
        length = self.length
        batch, heads, budget = keep.shape
        if batch != 1 or budget != round(.7*length) or length <= 32:
            raise ValueError('Expected the fixed B=1, 70% prompt protocol')
        old = torch.zeros((heads,length), dtype=torch.bool, device=keep.device)
        old.scatter_(1, keep[0], True)
        if not old[:,-32:].all() or not (old.sum(1)==budget).all():
            raise RuntimeError('Standard selector violated budget or recent-window invariants')
        pos, valid = self.positions, self.valid
        counts = (old[:,pos] & valid).sum(-1)
        before = counts < self.minimum
        new = old
        if repair and before.any():
            votes = hetero_cache.observation_votes(attention, heads, 32)[0]
            pooled = hetero_cache.pool_scores(votes[None,:,:-32],7,'maxpool')[0]
            scores = torch.cat((pooled,votes[:,-32:]), dim=1)
            # Within each span: protected tail first, then old selections,
            # then missing positions. Break ties by pooled vote and position.
            tier = valid.long() + old[:,pos].long() + (pos >= length-32).long()
            tier = tier.masked_fill(~valid, -1)
            by_vote = scores[:,pos].argsort(dim=-1, descending=True, stable=True)
            by_tier = tier.gather(-1,by_vote).argsort(dim=-1,descending=True,stable=True)
            order = by_vote.gather(-1,by_tier)
            required_ids = pos.expand(heads,-1,-1).gather(-1,order)
            required = (self.ranks < self.minimum[:,None]).expand(heads,-1,-1)
            must = torch.zeros_like(old,dtype=torch.int32)
            must.scatter_add_(1, required_ids.reshape(heads,-1), required.reshape(heads,-1).int())
            must[:,-32:] = 1
            if (must.sum(1)>budget).any():
                raise RuntimeError('Span minima exceed the original per-head budget')
            priorities = scores.masked_fill(~old, -torch.inf).masked_fill(must.bool(), torch.inf)
            # Keep heads already satisfying the minima bit-for-bit unchanged.
            updated = priorities.topk(budget,dim=1).indices.sort(dim=1).values
            keep = torch.where(before.any(1,keepdim=True), updated, keep[0])[None]
            new = torch.zeros_like(old).scatter_(1,keep[0],True)
        after = (new[:,pos] & valid).sum(-1) < self.minimum
        if (repair and after.any()) or not new[:,-32:].all() or not (new.sum(1)==budget).all():
            raise RuntimeError('Coverage, recent-window or exact budget check failed')
        self.layers.append(dict(heads=heads,budget=budget, checked_stems=heads*int(self.stems.sum()),
                    checked_options=heads*int((~self.stems).sum()), before_stem=int(before[:,self.stems].sum()),
                    before_option=int(before[:,~self.stems].sum()), after_stem=int(after[:,self.stems].sum()),
                    after_option=int(after[:,~self.stems].sum()), swapped=int((new & ~old).sum())))
        return keep

    def totals(self):
        return {key:sum(row[key] for row in self.layers) for key in
                ('checked_stems','checked_options','before_stem','before_option','after_stem','after_option','swapped')}
