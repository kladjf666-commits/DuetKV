from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy, gc, importlib.util, json, sys, tempfile, unittest, weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'benchmark'), str(ROOT)]
import torch
import common as c
import baseline_runtime as baseline
from PhaseKV.phasekv_cache import PhaseKVCache
import PhaseKV.phasekv_cache as cache_module
from PhaseKV.memory_prefill import memory_prefill, capture_indices, flatten_owned_cache
from PhaseKV.prompt_tier import select_prompt_indices
from PhaseKV.kernels.packing import unpack_dequantize


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
        expected = PhaseKVCache.from_dense_prefill(dense, prompt_len=97, keep_indices=keep, **packing_options())
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
            actual = PhaseKVCache.from_dense_prefill(consumed, prompt_len=97, keep_indices=keep,
                                                   consume_dense_cache=True, **packing_options())
        self.assertEqual(consumed.key_cache, [None] * 3)
        self.assertEqual(consumed.value_cache, [None] * 3)
        self.assertTrue(all(k() is None and v() is None for k, v in refs))
        assert_packed_equal(self, expected, actual)

    def test_consume_requires_mutable_lists(self):
        dense = SimpleNamespace(key_cache=(torch.zeros(1, 1, 2, 32).half(),),
                                value_cache=(torch.zeros(1, 1, 2, 32).half(),))
        with self.assertRaises(TypeError):
            PhaseKVCache.from_dense_prefill(dense, prompt_len=2, consume_dense_cache=True, **packing_options())

    def test_in_place_flatten_has_one_owner_and_preserves_request_order(self):
        for batch in (1, 2, 4, 8):
            dense = SimpleNamespace(key_cache=[torch.arange(batch * 2 * 97 * 32).reshape(batch, 2, 97, 32).half()],
                                    value_cache=[torch.zeros(batch, 2, 97, 32).half()])
            selected = [torch.arange(87).view(1, 1, -1).expand(batch, 2, -1)]
            expected, indices = baseline.flatten_requests(dense, selected)
            key_list = dense.key_cache
            flatten_owned_cache(dense, selected)
            self.assertIs(key_list, dense.key_cache)
            torch.testing.assert_close(dense.key_cache[0], expected.key_cache[0], atol=0, rtol=0)
            torch.testing.assert_close(selected[0], indices[0], atol=0, rtol=0)

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

    def test_real_llama_prefill_exact_logits_packing_and_cooling(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=1, max_position_embeddings=2048,
                          vocab_size=128, attention_dropout=0., _attn_implementation='eager')
        model = LlamaForCausalLM(cfg).half().eval()
        class Wrapper(torch.nn.Module):
            def __init__(self, original): super().__init__(); self.original = original
            def forward(self, *args, **kw): return self.original(*args, **kw)
        for layer in model.model.layers: layer.self_attn = Wrapper(layer.self_attn)
        def forward(model, ids, cache, start, method, **kwargs):
            positions = torch.arange(start, start + ids.shape[1])
            result = model.model(input_ids=ids, past_key_values=cache, position_ids=positions[None],
                                 cache_position=positions, use_cache=True, return_dict=True, **kwargs)
            return model.lm_head(result.last_hidden_state[:, -1:, :]).float(), result.past_key_values, result.attentions
        runtime = SimpleNamespace(forward=forward)
        saved = [layer.self_attn.original.forward for layer in model.model.layers]
        with torch.inference_mode():
            for batch, length in ((1, 17), (1, 97), (4, 612), (8, 97)):
                ids = torch.randint(0, 128, (batch, length))
                logits_a, cache_a = baseline.speed_prefill(model, ids, c.options(c.PROFILE), runtime, {})
                logits_b, cache_b = memory_prefill(model, ids, c.options(c.PROFILE), runtime, {})
                torch.testing.assert_close(logits_a, logits_b, rtol=0, atol=0)
                assert_packed_equal(self, cache_a, cache_b)
                self.assertEqual([layer.self_attn.original.forward for layer in model.model.layers], saved)
                # Consume identical generated KV through two cooling cycles and a hot-ring wrap.
                for _ in range(163):
                    for layer in range(3):
                        k = torch.randn(1, batch, 1, 32).half(); v = torch.randn_like(k)
                        cache_a.append_hot(layer, k, v); cache_b.append_hot(layer, k, v)
                    cache_a.finish_decode_step(); cache_b.finish_decode_step()
                self.assertEqual(cache_a.true_seq_length, length + 163)
                for a, b in zip(cache_a.layers, cache_b.layers):
                    self.assertEqual([s.length for s in b.cold], [13, 13])
                    for x, y in zip(a.cold, b.cold):
                        torch.testing.assert_close(unpack_dequantize(x.key), unpack_dequantize(y.key), rtol=0, atol=0)
                    for x, y in zip(a.hot_views(), b.hot_views()): torch.testing.assert_close(x, y, rtol=0, atol=0)
            ids = torch.randint(0, 128, (1, 97))
            a, ca = baseline.speed_prefill(model, ids, c.options(c.PROFILE), runtime, {})
            b, cb = memory_prefill(model, ids, c.options(c.PROFILE), runtime, {}, prefix_chunk_size=None)
            torch.testing.assert_close(a, b, rtol=0, atol=0); assert_packed_equal(self, ca, cb)
        def fail(*a, **kw): raise RuntimeError('tail failure')
        with self.assertRaises(RuntimeError):
            memory_prefill(model, ids[:, :17], c.options(c.PROFILE), SimpleNamespace(forward=fail), {})
        self.assertEqual([layer.self_attn.original.forward for layer in model.model.layers], saved)

    def test_atomic_checkpoint_rejects_overwrite_and_failed_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT.resolve()))
            path = Path(folder) / 'r1.json'; row = dict(status='ok', experiment='x')
            c.write(path, row, new=True)
            with self.assertRaises(FileExistsError): c.write(path, dict(row, status='error'), new=True)
            self.assertEqual(c.read(path), row)
            self.assertEqual(c.validate_saved(row, 'x'), row)
            with self.assertRaises(ValueError): c.validate_saved(row, 'y')
            with self.assertRaises(RuntimeError): c.validate_saved(dict(row, status='oom'), 'x')

    def test_inputs_and_small_quick_plan(self):
        from runner import inputs
        quality, bank = inputs()
        self.assertEqual(len(quality), 2); self.assertEqual(len(bank), 8)
        self.assertEqual([len(r['input_ids']) for r in quality], [8074, 32599])
        self.assertEqual([r['max_new_tokens'] for r in quality], [128, 120])
        self.assertEqual(c.CASES, [(32768, 256, 8), (8192, 256, 1), (32768, 2048, 1)])

    def test_resume_only_uncommitted_rounds_and_report_output_mismatch(self):
        from report import report
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as folder:
            out = Path(folder); self.assertTrue(out.resolve().is_relative_to(ROOT.resolve()))
            plan = dict(experiment='x', quality=[])
            c.write(out / 'plan.json', plan)
            row = dict(status='ok', experiment='x', batch=8, output_tokens_per_row=256,
                       aggregate_tokens_per_s=50, per_sequence_tokens_per_s=6.25, ttft_s=40,
                       decode_s=40, final_kv_bytes=1000, prefill_peak_bytes=2000,
                       decode_peak_bytes=1500, peak_allocated_bytes=2000, output_sha256='a')
            name = c.case_name(c.CASES[0]); first = out / f'speed/{name}/baseline/r1.json'
            c.write(first, row, new=True); before = first.read_bytes()
            self.assertEqual(c.pending_rounds(out, c.CASES[0], 'x'),
                             {'baseline': [2, 3], 'optimized': [1, 2, 3]})
            c.write(out / f'speed/{name}/optimized/r1.json', dict(row, output_sha256='b'), new=True)
            summary = report(out)
            self.assertFalse(summary['complete']); self.assertFalse(summary['output_matches'][name + '/r1'])
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(summary['speed'][name]['baseline']['formal_runs'], 1)

    def test_prepare_is_idempotent_and_rejects_changed_plan(self):
        import runner
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as folder:
            temp = Path(folder); self.assertTrue(temp.resolve().is_relative_to(ROOT.resolve()))
            model = temp / 'model'; model.mkdir(); weight = model / 'weight.bin'; weight.write_bytes(b'weight')
            c.write(model / 'model.safetensors.index.json', {'weight_map': {'w': 'weight.bin'}})
            weights = {'weight.bin': [weight.stat().st_size, weight.stat().st_mtime_ns]}
            cfg = dict(bench_manifest=c.BASE, bench=str(temp / 'bench'), model=str(model),
                       weights=weights, old_out=str(temp / 'old'))
            c.write(temp / 'config.json', cfg)
            c.write(temp / 'old/results/phasekv/protocol.json', dict(weights=weights, model=str(model)))
            args = SimpleNamespace(config=temp / 'config.json', out=temp / 'out')
            original_read = c.read
            def fake_read(path):
                if Path(path) == c.ROOT / 'data/model_metadata.json': return {}
                return original_read(path)
            with patch.object(c, 'verify', return_value='manifest'), patch.object(c, 'read', side_effect=fake_read):
                a = runner.prepare(args); b = runner.prepare(args); self.assertEqual(a, b)
                c.write(args.out / 'plan.json', dict(a, rounds=99))
                with self.assertRaises(ValueError): runner.prepare(args)

    def test_report_uses_per_round_e2e_and_maximum_peak(self):
        from report import summarize
        rows = [dict(status='ok', batch=8, output_tokens_per_row=256, aggregate_tokens_per_s=10*i,
                     per_sequence_tokens_per_s=10*i/8, ttft_s=i, decode_s=10/i,
                     final_kv_bytes=i*2**30, prefill_peak_bytes=i*2**30,
                     decode_peak_bytes=(4-i)*2**30, peak_allocated_bytes=max(i,4-i)*2**30) for i in (1,2,3)]
        result = summarize(rows)
        self.assertEqual(result['tps'], 20); self.assertEqual(result['final_kv_gib'], 2)
        self.assertEqual(result['peak_gib'], 3)
        self.assertAlmostEqual(result['e2e_tps'], 2048/7)
        rows[1]['status'] = 'oom'; self.assertEqual(summarize(rows)['status'], 'failed')


if __name__ == '__main__': unittest.main()
