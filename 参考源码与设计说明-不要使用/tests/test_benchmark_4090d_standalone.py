"""Controller/data/report regressions; no CUDA, torch, or missing eval package needed.

Run: python -m unittest discover -s tests -p test_benchmark_4090d_standalone.py -v
"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_4090d as bench


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def tolist(self):
        return self.values


class Tokenizer:
    chat_template = "test-template"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=True):
        text = "<BOS><USER>" + messages[0]["content"] + "<ASSISTANT>"
        return self.encode(text) if tokenize else text


class TestPlanning(unittest.TestCase):
    def test_all_chunks_cover_input_once_and_last_has_observation_window(self):
        for length in (1, 31, 32, 33, 511, 512, 513, 1024, 32768):
            chunks = bench.chunk_ranges(length, 512, 32)
            self.assertEqual([n for start, end in chunks for n in range(start, end)], list(range(length)))
            self.assertEqual(chunks[-1][1] - chunks[-1][0], min(length, 32))
            self.assertTrue(all(end - start <= 512 for start, end in chunks))

    def test_capacity_recovers_from_oom_and_finds_target_boundary(self):
        attempted = []

        def probe(n):
            attempted.append(n)
            return {"status": "oom"} if n > 6000 else {"status": "ok", "peak_allocated_bytes": n * 10}

        result = bench.capacity_search(32000, 50000, probe)
        self.assertIn(8192, attempted)
        self.assertLessEqual(result["target_safe_input"], 5000)
        self.assertLess(5000 - result["target_safe_input"], 256)
        self.assertEqual(len(attempted), len(set(attempted)))

    def test_capacity_stops_at_model_limit_without_fake_padding(self):
        attempted = []
        def probe(n):
            attempted.append(n)
            return {"status": "ok", "peak_allocated_bytes": n}
        result = bench.capacity_search(3968, 100000, probe)
        self.assertEqual(result["target_safe_input"], 3968)
        self.assertEqual(result["reason"], "checkpoint_or_requested_limit")
        self.assertLessEqual(max(attempted), 3968)

    def test_capacity_total_failure_is_not_reported_as_success(self):
        result = bench.capacity_search(4096, 100, lambda n: {"status": "oom"})
        self.assertEqual(result["max_completed_input"], 0)
        self.assertEqual(result["target_safe_input"], 0)

    def test_sampling_is_reproducible_and_supports_full_dataset(self):
        indices = bench.sample_indices(200, 32, 1234, "qasper")
        self.assertEqual(indices, bench.sample_indices(200, 32, 1234, "qasper"))
        self.assertNotEqual(indices, list(range(32)))
        self.assertEqual(len(set(indices)), 32)
        self.assertEqual(bench.sample_indices(17, 0, 1, "qasper"), list(range(17)))

    def test_dry_run_works_without_gpu_or_legacy_eval(self):
        result = subprocess.run([sys.executable, str(Path(bench.__file__)), "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(len(plan["models"]), 2)
        self.assertEqual(plan["models"][1]["context_limit"], 4096)


class TestData(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.fake_torch = types.SimpleNamespace(tensor=lambda data, **kw: FakeTensor(data), long="long")

    def test_needle_survives_at_each_position_and_exact_context_size(self):
        with patch.dict(sys.modules, {"torch": self.fake_torch}):
            locations = []
            for depth in (.1, .5, .9):
                ids, answer = bench.synthetic_input(self.tokenizer, 1024, 1234, depth)
                text = self.tokenizer.decode(ids.tolist()[0])
                self.assertEqual(ids.shape[1], 1024)
                self.assertEqual(text.count(answer), 1)
                self.assertEqual(text.count("<BOS>"), 1)
                locations.append(text.index(answer))
            self.assertTrue(locations[0] < locations[1] < locations[2])

    def test_quality_truncation_keeps_question_and_respects_total_budget(self):
        sample = {"context": "start-" + "x" * 2000 + "-end", "input": "What happened?"}
        with patch.dict(sys.modules, {"torch": self.fake_torch}):
            ids, meta = bench.quality_input(self.tokenizer, "{context}\nQuestion: {input}", sample, 256)
        text = self.tokenizer.decode(ids.tolist()[0])
        self.assertLessEqual(ids.shape[1], 256)
        self.assertIn("start-", text)
        self.assertIn("-end", text)
        self.assertIn("What happened?", text)
        self.assertTrue(meta["truncated"])
        self.assertEqual(text.count("<BOS>"), 1)

    def test_data_loader_extracts_only_named_jsonl_members(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            folder = Path(directory)
            with zipfile.ZipFile(folder / "data.zip", "w") as archive:
                archive.writestr("data/qasper.jsonl", json.dumps({"context": "article", "input": "Q", "answers": ["A"]}) + "\n")
                archive.writestr("../unexpected.txt", "must not be extracted")
            (folder / "dataset2prompt.json").write_text(json.dumps({"qasper": "{context} {input}"}))
            with patch.object(bench, "download"):
                result = bench.prepare_data(folder, ["qasper"])
            self.assertEqual(result["tasks"]["qasper"]["count"], 1)
            self.assertFalse((folder.parent / "unexpected.txt").exists())
            self.assertEqual(len(result["tasks"]["qasper"]["sha256"]), 64)

    def test_qa_metric_accepts_equivalent_words_not_only_identical_tokens(self):
        self.assertEqual(bench.qa_f1("The River!", "river"), 1)
        self.assertAlmostEqual(bench.qa_f1("red blue", "blue green"), .5)
        self.assertEqual(bench.qa_f1("", "answer"), 0)


class TestValidationIsolation(unittest.TestCase):
    def test_materialized_cache_preserves_dtype_and_cannot_mutate_source(self):
        class Tensor:
            def __init__(self, dtype, values):
                self.dtype, self.values = dtype, values

            def clone(self):
                return Tensor(self.dtype, list(self.values))

        class Cache:
            def __init__(self):
                self.key_cache, self.value_cache = [], []

            def update(self, key, value, index):
                self.key_cache.append(key)
                self.value_cache.append(value)

        for dtype in ('float16', 'bfloat16', 'float32'):
            with self.subTest(dtype=dtype):
                source = Cache()
                source.update(Tensor(dtype, [1]), Tensor(dtype, [2]), 0)
                with patch.dict(sys.modules, {'transformers.cache_utils': types.SimpleNamespace(DynamicCache=Cache)}):
                    copy = bench.materialized_cache(source)
                self.assertEqual(copy.key_cache[0].dtype, dtype)
                self.assertEqual(copy.value_cache[0].dtype, dtype)
                copy.key_cache[0].values[0] = 99
                copy.value_cache[0].values[0] = 99
                self.assertEqual(source.key_cache[0].values, [1])
                self.assertEqual(source.value_cache[0].values, [2])

    def test_packed_instrumentation_restores_kernels_and_hooks_on_failure(self):
        removed = []

        class Attention:
            original = types.SimpleNamespace(original=object())

            def register_forward_pre_hook(self, hook, with_kwargs):
                return types.SimpleNamespace(remove=lambda: removed.append('before'))

            def register_forward_hook(self, hook):
                return types.SimpleNamespace(remove=lambda: removed.append('after'))

        class Model:
            config = types.SimpleNamespace(model_type='qwen2')
            model = types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=Attention())])

            def __call__(self, **kwargs):
                self.test.assertIsNot(kivi.kivi_decode, original_kivi)
                self.test.assertIsNot(phase.fused_phasekv_decode, original_phase)
                raise RuntimeError('simulated kernel failure')

        names = ['torch', 'KIVI', 'KIVI.kernels', 'PhaseKV', 'PhaseKV.kernels',
                 'PhaseKV.kernels.attention_triton', 'transformers.models.qwen2.modeling_qwen2',
                 'transformers.models.llama.modeling_llama']
        modules = {name: types.ModuleType(name) for name in names}
        kivi, phase = modules['KIVI.kernels'], modules['PhaseKV.kernels.attention_triton']
        original_kivi, original_phase = object(), object()
        kivi.kivi_decode, phase.fused_phasekv_decode = original_kivi, original_phase
        modules['KIVI'].kernels = kivi
        modules['PhaseKV'].kernels = modules['PhaseKV.kernels']
        modules['PhaseKV.kernels'].attention_triton = phase
        for name in names[-2:]:
            modules[name].apply_rotary_pos_emb = object()
        model = Model()
        model.test = self
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, 'simulated kernel failure'):
                bench.validated_packed_forward(model, types.SimpleNamespace(layers=[object()]), {}, None)
        self.assertIs(kivi.kivi_decode, original_kivi)
        self.assertIs(phase.fused_phasekv_decode, original_phase)
        self.assertEqual(removed, ['after', 'before'])


class TestPersistenceAndReports(unittest.TestCase):
    def test_validation_failure_saves_stage_and_keeps_prior_passes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "validation.json"
            checks = []
            bench.validation_step(checks, {"check": "fp32"}, lambda: {"max_abs_error": 0.0}, path)
            def mismatch():
                raise AssertionError("mismatched element")
            with self.assertRaisesRegex(RuntimeError, '"method": "kivi"'):
                bench.validation_step(checks, {"check": "decode", "method": "kivi", "step": 17}, mismatch, path)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure"]["step"], 17)
            self.assertEqual(report["failure"]["error_type"], "AssertionError")
            self.assertEqual(len(report["checks"]), 1)
            self.assertEqual(report["checks"][0]["status"], "passed")

    def test_validation_runtime_error_cannot_be_recorded_as_pass(self):
        checks = []
        def failure():
            raise ValueError("non-finite")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            bench.validation_step(checks, {"check": "finiteness"}, failure)
        self.assertEqual(checks, [])

    def test_resume_discards_only_torn_final_record(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "records.jsonl"
            path.write_bytes(b'{"n": 1}\n{"n":')
            self.assertEqual(bench.read_records(path, repair=True), [{"n": 1}])
            bench.append_record(path, {"n": 2})
            self.assertEqual(bench.read_records(path), [{"n": 1}, {"n": 2}])
            path.write_bytes(b'broken\n{"n": 2}\n')
            with self.assertRaises(ValueError):
                bench.read_records(path)

    def test_live_report_reader_does_not_truncate_in_progress_append(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "records.jsonl"
            data = b'{"n": 1}\n{"n":'
            path.write_bytes(data)
            self.assertEqual(bench.read_records(path), [{"n": 1}])
            self.assertEqual(path.read_bytes(), data)

    def test_resume_rejects_mixed_protocol(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "records.jsonl"
            bench.append_record(path, {"protocol_hash": "old", "case_id": "1"})
            with self.assertRaisesRegex(ValueError, "protocol"):
                bench.Recorder(path, "new", "qwen", {"method": "full"})

    def make_report(self, out, mismatch=False):
        (out / "records").mkdir()
        bench.atomic_json(out / "manifest.json", {"protocol": "test", "settings": {"methods": ["full", "phasekv"]}})
        for method in ("full", "phasekv"):
            for index in (0, 1):
                record = {"model": "qwen", "method": method, "phase": "quality", "task": "qasper",
                    "case_id": f"quality/qasper/1024/{index}", "length_budget": 1024,
                    "status": "oom" if method == "phasekv" and index == 1 else "ok",
                    "input_hash": method if mismatch else str(index), "score": 50 if method == "full" else 40,
                    "final_kv_bytes": 2**20, "decode_compression_steps": 1}
                bench.append_record(out / "records" / f"{method}.jsonl", record)

    def test_report_pairs_same_samples_and_does_not_score_oom_as_zero(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            out = Path(directory)
            self.make_report(out)
            bench.summarize(out)
            report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["excluded_unpaired_cases"], 1)
            rows = {r["method"]: r for r in report["paired_quality"]}
            self.assertEqual(rows["phasekv"]["score"], 40)
            self.assertEqual(rows["phasekv"]["n_paired"], 1)
            self.assertEqual(rows["phasekv"]["delta_vs_full"], -10)
            self.assertEqual(rows["phasekv"]["paired_delta_95pct_bootstrap"], [-10, -10])

    def test_report_rejects_different_model_inputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            out = Path(directory)
            self.make_report(out, mismatch=True)
            with self.assertRaisesRegex(ValueError, "input hash"):
                bench.summarize(out)


if __name__ == "__main__":
    unittest.main()
