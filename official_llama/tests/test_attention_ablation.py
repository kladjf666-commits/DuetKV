"""CPU contracts for the additive ablation; no simulated GPU performance claims."""
import ast
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import attention_ablation as runner
import ablation_generate
import ablation_runtime
from ablation_analysis import paired_report


class AttentionAblationTests(unittest.TestCase):
    def test_frozen_sources_and_dataset_coverage(self):
        from prepare import verify
        self.assertGreater(len(verify()['files']), 900)
        rows, _ = runner.samples()
        planned = list(runner.expected(rows, runner.SUITES, runner.METHODS))
        self.assertEqual(len(planned), 4612)
        self.assertEqual(Counter(m for m, s, r in planned), dict.fromkeys(runner.METHODS, 1153))
        self.assertEqual(sum(s['suite'] == 'performance' for _, s, _ in planned), 324)
        for method in runner.METHODS:
            ids = {(s['id'], r) for m, s, r in planned if m == method}
            self.assertEqual(len(ids), 1153)

    def test_unknown_method_fails(self):
        with self.assertRaises(ValueError):
            ablation_runtime.cache_method('phasekv_typo')

    def test_common_packed_math_and_merge_preserved(self):
        def funcs(path):
            return {n.name: n for n in ast.parse(path.read_text('utf-8')).body if isinstance(n, ast.FunctionDef)}
        base = funcs(ROOT/'runtime/PhaseKV/kernels/segmented_attention.py')
        new = funcs(ROOT/'ablation_kernels.py')
        self.assertEqual(ast.dump(base['_merge_segments']), ast.dump(new['_merge_segments']))
        # The packed branch is identical, including dequant arithmetic precision.
        def packed_branch(fn):
            return next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'part < NPACKED').body
        self.assertEqual([ast.dump(x) for x in packed_branch(base['_segmented_attention'])],
                         [ast.dump(x) for x in packed_branch(new['_segmented_attention'])])

    def test_generation_positions_and_commits_all_four(self):
        import torch
        fake = types.ModuleType('ablation_runtime')
        states, positions = [], []
        class Cache:
            n = 0
            commits = 0
            def finish_decode_step(self):
                self.commits += 1
        def forward(model, ids, cache, start, method):
            if not hasattr(cache, 'n'):
                cache = Cache()
                states.append(cache)
            positions.append(start)
            cache.n += ids.shape[1]
            logits = torch.zeros((1, 1, 8))
            logits[0, 0, 2] = 1
            return logits, cache, None
        fake.reset_sample = lambda *a: ({}, 'ablation')
        fake.forward = forward
        fake.phase_prefill = lambda model, ids, opts: forward(model, ids, None, 0, 'phasekv')[:2]
        fake.cache_report = lambda c: {'physical_bytes': c.n*16, 'auxiliary_bytes': 0}
        fake.cache_trace = lambda c, m: {'physical_tokens': c.n}
        original_tensor = torch.tensor
        def cpu_tensor(*a, **kw):
            kw.pop('device', None)
            return original_tensor(*a, **kw)
        sample = {'id': 'test', 'input_ids': [1, 3, 4], 'max_new_tokens': 7,
                  'stop_ids': [2], 'natural_eos': False}
        model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=1000))
        with patch.dict(sys.modules, {'ablation_runtime': fake}), patch('torch.tensor', cpu_tensor), \
             patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), \
             patch('torch.cuda.max_memory_allocated', return_value=0), \
             patch('torch.cuda.max_memory_reserved', return_value=0):
            for method in runner.METHODS:
                positions.clear()
                result = ablation_generate.generate(model, method, {}, sample, diagnostic=True)
                self.assertEqual(result['output_tokens'], 7)
                self.assertEqual(positions, [0, 3, 4, 5, 6, 7, 8])
                self.assertEqual(sum(c['tokens'] for c in result['decode_chunks']), 6)
                self.assertEqual(states[-1].commits, 6 if method.startswith('phasekv_') else 0)
            sample['natural_eos'] = True
            result = ablation_generate.generate(model, 'phasekv_fa2', {}, sample)
            self.assertEqual(result['output_tokens'], 1)
            self.assertEqual(result['decode_s'], 0)
            self.assertEqual(states[-1].commits, 0)

    def test_dense_scratch_and_transient_memory_separate(self):
        import torch
        tensor = torch.zeros((1, 2, 5, 16), dtype=torch.float16)
        scratch = torch.zeros((2, 32), dtype=torch.float32)
        cache = types.SimpleNamespace(key_cache=[tensor], value_cache=[tensor],
                                      _ablation_scratch={(0, 0): (scratch,)}, _materialized_peak=4096)
        report = ablation_runtime.cache_report(cache)
        self.assertEqual(report['physical_bytes'], tensor.numel()*2)  # aliased storage once
        self.assertEqual(report['auxiliary_bytes'], scratch.numel()*4)
        self.assertEqual(report['total_owned_bytes'], tensor.numel()*2+scratch.numel()*4)
        self.assertEqual(report['materialized_kv_bytes_max_layer'], 4096)

    def test_prefill_delegates_including_one_token_observation(self):
        import torch
        from ablation_attention import AblationAttention
        class Original(torch.nn.Module):
            layer_idx = 0
            def forward(self, **kw):
                return kw
        original = Original().eval()
        hidden = torch.zeros(1, 1, 16)
        # Dense PhaseKV prefill must delegate even with one query and past KV.
        wrapper = AblationAttention(original, 'phasekv_fa2')
        result = wrapper(hidden, past_key_value=object(), output_attentions=True)
        self.assertIs(result['hidden_states'], hidden)
        self.assertTrue(result['output_attentions'])
        # Production decode must never silently fall back to CPU computation.
        wrapper.method = 'full_triton'
        cache = types.SimpleNamespace(get_seq_length=lambda idx: 3)
        with self.assertRaises(ValueError):
            wrapper(hidden, past_key_value=cache, use_cache=True)

    def test_pairs_require_all_repeats_and_matching_lengths(self):
        rows = []
        for method, speed in zip(runner.METHODS, (10., 8., 5., 12.)):
            for repeat in range(3):
                rows.append({'method': method, 'id': 'p', 'suite': 'performance', 'task': 'shape',
                             'repeat': repeat, 'status': 'ok', 'input_sha256': 'same',
                             'output_tokens': 256, 'decode_tokens_per_s': speed})
        with tempfile.TemporaryDirectory() as directory:
            def report():
                paired_report(directory, rows)
                return json.loads((Path(directory)/'ablation_pairs.json').read_text('utf-8'))['contrasts']
            comparisons = report()
            self.assertAlmostEqual(comparisons[0]['performance_pairs'][0]['speedup'], .8)
            self.assertAlmostEqual(comparisons[1]['performance_pairs'][0]['speedup'], 2.4)
            rows[0]['status'] = 'oom'
            comparisons = report()
            self.assertEqual(comparisons[0]['performance_pairs'], [])
            self.assertEqual(comparisons[0]['skipped_incomplete_or_failed_pairs'], ['p'])
            self.assertEqual(len(comparisons[1]['performance_pairs']), 1)
            rows[0]['status'] = 'ok'
            rows[0]['output_tokens'] = 255
            with self.assertRaises(RuntimeError):
                report()

    def test_failed_results_and_pending_are_retained(self):
        sample = {'id': 's', 'suite': 'longbench', 'task': 'qasper', 'methods': ['full', 'phasekv'],
                  'length_budget': 8192, 'input_sha256': 'same', 'cluster_id': 'q/1'}
        with tempfile.TemporaryDirectory() as directory:
            path = runner.result_path(directory, 'phasekv_fa2', sample, 0)
            runner.atomic(path, {**sample, 'method': 'phasekv_fa2', 'repeat': 0, 'status': 'oom'})
            self.assertFalse(runner.pending(path))
            self.assertTrue(runner.pending(path, retry_failed=True))
            status = runner.summarize(directory, [sample], ['longbench'], runner.METHODS)
            self.assertEqual(status['oom'], 1)
            self.assertEqual(status['pending'], 3)
            summary = json.loads((Path(directory)/'summary.json').read_text('utf-8'))
            self.assertFalse(summary['completed'])

    def test_worker_rejects_failed_stale_gate_and_changed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            job = {'out': directory, 'protocol_hash': 'test-protocol'}
            source = 'ablation_runtime.py'
            code_hash = runner.digest((ROOT/source).read_bytes())
            manifest = {'protocol_hash': 'test-protocol', 'protocol': {'code': {source: code_hash}}}
            runner.atomic(out/'run_manifest.json', manifest)
            for gate in ({'status': 'failed', 'protocol_hash': 'test-protocol'},
                         {'status': 'passed', 'protocol_hash': 'different'}):
                runner.atomic(out/'gpu_gate.json', gate)
                with self.assertRaises(RuntimeError):
                    runner.verify_worker_protocol(job)
            runner.atomic(out/'gpu_gate.json', {'status': 'passed', 'protocol_hash': 'test-protocol'})
            runner.verify_worker_protocol(job)
            manifest['protocol']['code'][source] = 'wrong-hash'
            runner.atomic(out/'run_manifest.json', manifest)
            with self.assertRaises(RuntimeError):
                runner.verify_worker_protocol(job)


if __name__ == '__main__':
    unittest.main()
