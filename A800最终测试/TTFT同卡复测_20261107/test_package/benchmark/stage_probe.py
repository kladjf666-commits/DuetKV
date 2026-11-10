"""Diagnostic-only CPU spans and CUDA-stream spans. Never used in formal runs."""
from contextlib import contextmanager
import importlib
import time


class StageProbe:
    def __init__(self, runtime, length, observation, event_factory=None, synchronize=None):
        if event_factory is None:
            import torch
            event_factory = lambda: torch.cuda.Event(enable_timing=True)
            synchronize = torch.cuda.synchronize
        self.runtime = runtime
        self.tail_start = length - min(observation, length)
        self.event_factory, self.synchronize = event_factory, synchronize
        self.rows, self.stack, self.restores = [], [], []

    @contextmanager
    def span(self, name):
        start, end = self.event_factory(), self.event_factory()
        parent = self.stack[-1] if self.stack else None
        self.stack.append(name)
        start.record()
        tick = time.perf_counter()
        try:
            yield
        finally:
            wall_ms = (time.perf_counter() - tick) * 1000
            end.record()
            self.stack.pop()
            self.rows.append(dict(name=name, parent=parent, cpu_call_ms=wall_ms, events=(start, end)))

    def patch(self, obj, name, replacement, *, original=None):
        previous = getattr(obj, name) if original is None else original
        self.restores.append((obj, name, previous))
        setattr(obj, name, replacement)

    def __enter__(self):
        import baseline_runtime as baseline
        memory = importlib.import_module('PhaseKV.memory_prefill')
        prompt = importlib.import_module('PhaseKV.prompt_tier')
        from PhaseKV.phasekv_cache import PhaseKVCache
        original_forward = self.runtime.forward
        original_select = prompt.select_prompt_indices
        original_pack = PhaseKVCache.from_dense_prefill

        def forward(model, ids, cache, start, method, **kwargs):
            name = 'observation_forward' if start == self.tail_start else 'prefix_forward'
            with self.span(name):
                return original_forward(model, ids, cache, start, method, **kwargs)

        def select(*args, **kwargs):
            with self.span('selection'):
                return original_select(*args, **kwargs)

        def pack(cls, *args, **kwargs):
            with self.span('packing'):
                return original_pack(*args, **kwargs)

        def flatten_wrapper(original):
            def wrapped(*args, **kwargs):
                with self.span('flatten'):
                    return original(*args, **kwargs)
            return wrapped

        try:
            self.patch(self.runtime, 'forward', forward)
            self.patch(prompt, 'select_prompt_indices', select)
            self.patch(memory, 'select_prompt_indices', select)
            self.patch(baseline, 'flatten_requests', flatten_wrapper(baseline.flatten_requests))
            self.patch(memory, 'flatten_owned_cache', flatten_wrapper(memory.flatten_owned_cache))
            self.patch(PhaseKVCache, 'from_dense_prefill', classmethod(pack),
                       original=PhaseKVCache.__dict__['from_dense_prefill'])
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)
        self.restores.clear()

    def result(self):
        self.synchronize()
        rows = [{k:v for k,v in row.items() if k != 'events'} |
                {'cuda_stream_ms': row['events'][0].elapsed_time(row['events'][1])}
                for row in self.rows]
        groups = {}
        for name in ('prefix_forward', 'observation_forward', 'selection', 'flatten', 'packing', 'first_token'):
            selected = [row for row in rows if row['name'] == name]
            groups[name] = dict(calls=len(selected), **{
                metric:sum(row[metric] for row in selected) for metric in ('cpu_call_ms', 'cuda_stream_ms')})
        # Baseline selection follows observation; optimized selection is nested in it.
        outside = [row for row in rows if row['name'] == 'selection' and row['parent'] != 'observation_forward']
        groups['observation_and_selection'] = {
            metric:groups['observation_forward'][metric] + sum(row[metric] for row in outside)
            for metric in ('cpu_call_ms', 'cuda_stream_ms')}
        return dict(stages=groups, spans=rows,
            note='Diagnostic only. CUDA stream spans may include host launch gaps; they are not summed kernel times. '
                 'CPU call time is not synchronized GPU duration. Selection is already included in observation_and_selection; '
                 'do not add it again. No per-layer synchronize was inserted.')
