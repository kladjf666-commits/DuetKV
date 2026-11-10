"""CPU-only regressions for paired comparisons, selection and resumable jobs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sweep_phasekv as s


def fixture():
    rows = []
    for task in ("qasper", "gov_report"):
        for length in (1536, 3072):
            for i in range(2):
                rows.append(dict(case_id=f"quality/{task}/{length}/{i}", phase="quality", task=task,
                    length_budget=length, sample_index=i, status="ok", score=40. + i,
                    input_hash=f"{task}/{length}/{i}", answers=["answer"]))
    rows.append(dict(case_id="performance/short/1024/1024/1", phase="performance", sweep="short", status="ok",
        input_hash="perf", answers=None, input_tokens=1024, output_tokens=1024, requested_output_tokens=1024,
        decode_tokens_per_s=50., final_kv_bytes=100000, prefill_kv_bytes=50000, total_s=21., prefill_and_compress_s=1., peak_allocated_bytes=16*2**30))
    return rows


class TestSweep(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU cache lifecycle requires torch")
    def test_actual_cooling_retention_and_absolute_positions_across_windows(self):
        s.b.bootstrap_algorithms()
        import torch
        from transformers.cache_utils import DynamicCache
        from PhaseKV.phasekv_cache import PhaseKVCache
        torch.manual_seed(7)
        for window in (32, 64, 128, 256):
            for bits, keep in ((2, .5), (4, .75), (8, 1.)):
                cfg = s.spec(window=window, cold=bits, keep=keep)
                dense = DynamicCache()
                dense.update(torch.randn(1, 2, 12, 32, dtype=torch.float16),
                             torch.randn(1, 2, 12, 32, dtype=torch.float16), 0)
                cache = PhaseKVCache.from_dense_prefill(dense, prompt_len=12, recent_window=window,
                                                        cold_bits=bits, decode_keep_ratio=keep, lazy_g=16)
                frozen = cache.layers[0].prompt.key.packed.clone()
                steps = s.b.phasekv_validation_steps(cfg)
                for _ in range(steps):
                    k = torch.randn(1, 2, 1, 32, dtype=torch.float16)
                    cache.append_hot(0, k, k)
                    cache.finish_decode_step()
                layer = cache.layers[0]
                self.assertEqual(layer.hot_len, window)
                self.assertGreaterEqual(len(layer.cold), 2)
                self.assertTrue(all(x.length == round(16*keep) for x in layer.cold))
                self.assertEqual(cache.true_seq_length, 12 + steps)
                self.assertTrue(torch.equal(layer.prompt.key.packed, frozen))

    def test_holdout_documents_never_overlap_and_native_limits_are_respected(self):
        for alias in ("qwen", "llama"):
            args = s.arguments(["--model", alias])
            base = s.b.parse_args(["--models", alias])
            screen, final = s.settings(args, base), s.settings(args, base, True)
            for task in screen["tasks"]:
                a = s.b.sample_indices(200, screen["samples"], args.seed, task)
                z = s.b.sample_indices(200, final["samples"], args.seed, task, final["sample_offset"])
                self.assertFalse(set(a) & set(z))
            limit = 32768 if alias == "qwen" else 4096
            self.assertTrue(all(n+m <= limit for n, m, _ in final["performance_shapes"]))
            self.assertTrue(all(n+s.b.TASK_LIMITS[t] <= limit for n in final["quality_lengths"] for t in final["tasks"]))

    def test_incomplete_or_failed_candidate_cannot_win(self):
        baseline = fixture()
        keys = {r["case_id"] for r in baseline}
        for rows in (baseline[:-1], [dict(r, status="oom") for r in baseline]):
            row = s.metrics(rows, baseline, keys, s.spec(), 1)
            self.assertFalse(row["valid"])
            self.assertFalse(row["feasible"])
            with self.assertRaises(RuntimeError):
                s.select([row], 1)

    def test_quality_guard_checks_both_groups_and_pairs_actual_inputs(self):
        baseline = fixture()
        rows = copy.deepcopy(baseline)
        for r in rows:
            if r.get("task") == "gov_report":
                r["score"] -= 2
        keys = {r["case_id"] for r in baseline}
        result = s.metrics(rows, baseline, keys, s.spec(), 1, intervals=True)
        self.assertFalse(result["feasible"])
        self.assertEqual(result["qa_delta_ci95"], [0., 0.])
        self.assertEqual(result["summary_delta_ci95"], [-2., -2.])
        rows[0]["input_hash"] = "wrong truncation"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            s.metrics(rows, baseline, keys, s.spec(), 1)

    def test_fixed_generation_must_finish_requested_tokens(self):
        baseline = fixture()
        rows = copy.deepcopy(baseline)
        rows[-1]["output_tokens"] -= 1
        with self.assertRaisesRegex(ValueError, "fixed output"):
            s.metrics(rows, baseline, {r["case_id"] for r in baseline}, s.spec(), 1)

    def test_pareto_retains_real_tradeoffs_and_excludes_dominated_config(self):
        rows = [dict(name=name, spec=s.spec(ratio=ratio), valid=True, feasible=True,
                     loss=loss, speed=speed, kv=kv) for name, ratio, loss, speed, kv in
                [("quality", .7, .1, 1., .4), ("speed", .5, .5, 2., .2),
                 ("small", .3, .8, 1.5, .1), ("dominated", 1., .9, 1., .5)]]
        self.assertEqual({r["name"] for r in s.pareto(rows)}, {"quality", "speed", "small"})
        self.assertEqual({r["name"] for r in s.select(rows, 3)}, {"quality", "speed", "small"})

    def test_grid_includes_current_config_and_all_requested_dimensions(self):
        grid = s.prefill_grid()
        self.assertIn(s.spec(.3, 2, 32), grid)
        self.assertEqual({x["prompt_bits"] for x in grid}, {2, 4, 8})
        self.assertEqual({x["prefill_keep_ratio"] for x in grid}, {.3, .5, .7, 1.})
        decode = s.decode_grid([s.spec()])
        self.assertEqual({x["decode_keep_ratio"] for x in decode}, {.5, .75, 1.})
        for x in grid + decode:
            s.b.validate_spec(x, 512)

    def test_recommendation_requires_complete_verification(self):
        baseline = fixture()
        row = s.metrics(baseline, baseline, {r["case_id"] for r in baseline}, s.spec(), 1)
        with tempfile.TemporaryDirectory(dir=s.b.ROOT, prefix="_sweep_test_") as tmp:
            out = Path(tmp)
            s.report(out, {"final": [row]}, 1)
            self.assertEqual(json.loads((out / "recommendations.json").read_text()), {})
            self.assertFalse((out / "recommended_quality.json").exists())
            s.b.atomic_json(out / "final/stage_complete.json", [row["name"]])
            s.report(out, {"final": [row]}, 1)
            self.assertEqual(json.loads((out / "recommended_quality.json").read_text())["methods"], [s.spec()])

    def test_stage_resume_does_not_rerun_and_detects_modified_results(self):
        baseline = fixture()
        with tempfile.TemporaryDirectory(dir=s.b.ROOT, prefix="_sweep_test_") as tmp:
            out = Path(tmp)
            for t in ("qasper", "gov_report"):
                p = out / "data" / (t + ".jsonl")
                p.parent.mkdir(exist_ok=True)
                p.write_text('{}\n{}\n')
            a = dict(data_dir=tmp, tasks=["qasper", "gov_report"], samples=2, sample_offset=0,
                     seed=1234, quality_lengths=[1536, 3072], performance_shapes=[(1024, 1024, "short")], repeats=1)
            profile = dict(name="fixture", architecture={}, methods=[])
            manifest = dict(hash="protocol", checkpoint={"fixture": {"resolved_path": "unused"}})
            def fake_execute(command, log):
                if "__validate__" in command:
                    s.b.atomic_json(command[-1], {"status": "passed"})
                else:
                    job = json.loads(Path(command[-1]).read_text())
                    Path(job["records_path"]).write_text("".join(json.dumps(r) + "\n" for r in baseline))
            with patch.object(s, "execute", side_effect=fake_execute) as execute:
                first = s.run_stage(out, "final", [s.spec()], a, profile, manifest, {}, 1)
                self.assertEqual(execute.call_count, 4)
            with patch.object(s, "execute", side_effect=AssertionError("must not rerun")):
                again = s.run_stage(out, "final", [s.spec()], a, profile, manifest, {}, 1)
                self.assertEqual(first, again)
                with (out / "final/Full/records.jsonl").open("a") as f:
                    f.write("\n")
                with self.assertRaisesRegex(ValueError, "modified"):
                    s.run_stage(out, "final", [s.spec()], a, profile, manifest, {}, 1)


if __name__ == "__main__":
    unittest.main()
