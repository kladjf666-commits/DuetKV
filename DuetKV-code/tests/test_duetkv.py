from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy, gc, sys, unittest, weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'benchmark'), str(ROOT)]
import torch
import common as c
from DuetKV.duetkv_cache import DuetKVCache
import DuetKV.duetkv_cache as cache_module
from DuetKV.memory_prefill import memory_prefill, capture_indices, flatten_owned_cache
from DuetKV.prompt_tier import select_prompt_indices
from DuetKV.kernels.packing import unpack_dequantize


def packing_options():
    opts = c.options(c.PROFILE)
    return {k: opts[k] for k in ('prompt_bits', 'cold_bits', 'group_size',
            'recent_window', 'lazy_g', 'decode_keep_ratio', 'optimizations')}


def assert_packed_equal(test, left, right):
    test.assertEqual(left.true_seq_length, right.true_seq_length)
    for a, b in zip(left.layers, right.layers):
        for name in ('key', 'value'):
            for field in ('packed', 'scale', 'mins'):
                torch.testing.assert_close(getattr(getattr(a.prompt, name), field),
                                           getattr(getattr(b.prompt, name), field), rtol=0, atol=0)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.manual_seed(411)

    def test_consuming_pack_matches_default_and_releases_each_layer(self):
        dense = SimpleNamespace(key_cache=[torch.randn(1, 2, 97, 32).half() for _ in range(3)],
                                value_cache=[torch.randn(1, 2, 97, 32).half() for _ in range(3)])
        keep = [torch.arange(87).view(1, 1, -1).expand(1, 2, -1) for _ in range(3)]
        expected = DuetKVCache.from_dense_prefill(dense, prompt_len=97, keep_indices=keep, **packing_options())
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in dense.key_cache))
        consumed = copy.deepcopy(dense)
        refs = [(weakref.ref(k), weakref.ref(v)) for k, v in zip(consumed.key_cache, consumed.value_cache)]
        original = cache_module._pack_kv; calls = []
        def check(*args, **kwargs):
            gc.collect()
            for index in range(len(calls)):
                self.assertIsNone(refs[index][0]()); self.assertIsNone(refs[index][1]())
            calls.append(1)
            return original(*args, **kwargs)
        with patch.object(cache_module, '_pack_kv', side_effect=check):
            actual = DuetKVCache.from_dense_prefill(consumed, prompt_len=97, keep_indices=keep,
                                                   consume_dense_cache=True, **packing_options())
        self.assertEqual(consumed.key_cache, [None] * 3)
        self.assertEqual(consumed.value_cache, [None] * 3)
        self.assertTrue(all(k() is None and v() is None for k, v in refs))
        assert_packed_equal(self, expected, actual)

    def test_consume_requires_mutable_lists(self):
        dense = SimpleNamespace(key_cache=(torch.zeros(1, 1, 2, 32).half(),),
                                value_cache=(torch.zeros(1, 1, 2, 32).half(),))
        with self.assertRaises(TypeError):
            DuetKVCache.from_dense_prefill(dense, prompt_len=2, consume_dense_cache=True, **packing_options())

    def test_layer_capture_does_not_retain_attention_and_restores_on_error(self):
        originals = [SimpleNamespace(forward=lambda: None) for _ in range(3)]
        saved = [a.forward for a in originals]; refs = []
        cfg = SimpleNamespace(num_key_value_heads=2, num_attention_heads=8)
        def eager(attention, **kw):
            self.assertTrue(kw['output_attentions'])
            self.assertTrue(all(ref() is None for ref in refs))
            weights = torch.rand(2, 8, 32, 97).half(); refs.append(weakref.ref(weights))
            return None, weights, None
        with capture_indices(originals, cfg, 97, c.options(c.PROFILE), eager_forward=eager) as chosen:
            for attention in originals:
                self.assertIsNone(attention.forward(output_attentions=False)[1])
            self.assertTrue(all(x.shape == (2, 2, 87) for x in chosen))
        self.assertEqual([a.forward for a in originals], saved)
        self.assertTrue(all(ref() is None for ref in refs))
        with self.assertRaisesRegex(RuntimeError, 'test interruption'):
            with capture_indices(originals, cfg, 97, c.options(c.PROFILE), eager_forward=eager):
                originals[0].forward(); raise RuntimeError('test interruption')
        self.assertEqual([a.forward for a in originals], saved)
