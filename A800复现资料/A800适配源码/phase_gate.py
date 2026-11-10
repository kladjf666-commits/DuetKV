"""Check the actual SM80 dispatch and retained lengths outside timed measurements."""
import torch


class CountLaunches:
    def __init__(self, kernel):
        self.kernel, self.count = kernel, 0

    def __getitem__(self, grid):
        self.count += 1
        return self.kernel[grid]


def check_phase(engine, row):
    import PhaseKV.kernels.segmented_attention as attention
    original = attention._a800_segmented_attention
    counted = CountLaunches(original)
    attention._a800_segmented_attention = counted
    results = {}
    try:
        for quality in (False, True):
            ids = torch.tensor([row[:2112]], device='cuda')
            expected = round(2112 * .95) if quality else 2112
            before = counted.count
            logits = engine.start(ids, 3, quality, {})
            lengths = [layer.prompt.length for layer in engine.cache.layers]
            assert all(n == expected for n in lengths), lengths
            assert engine.cache.true_seq_length == 2112 and engine.offset == 2112
            assert torch.isfinite(logits).all()
            next_logits = engine.step(logits[:, -1].argmax(-1, keepdim=True))
            assert engine.cache.true_seq_length == 2113 and engine.offset == 2113
            assert torch.isfinite(next_logits).all()
            launches = counted.count - before
            assert launches == len(engine.model.model.layers), launches
            results['quality95' if quality else 'speed100'] = dict(
                prompt_tokens_by_layer=lengths, absolute_next_position=engine.offset,
                a800_mma_launches=launches)
            engine.release()
    finally:
        attention._a800_segmented_attention = original
    return results
