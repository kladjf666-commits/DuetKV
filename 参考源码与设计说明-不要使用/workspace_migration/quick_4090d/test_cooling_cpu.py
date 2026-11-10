"""Checks for the timing and pairing math, without loading CUDA."""
import unittest
from unittest.mock import patch

import cooling_cpu as probe


class HostTimingTests(unittest.TestCase):
    def test_separate_clocks_and_return_value(self):
        timer = probe.HostTimer()
        timer.slot = 4
        with patch.object(probe.time, 'perf_counter_ns', side_effect=(1000, 6000)), \
                patch.object(probe.time, 'thread_time_ns', side_effect=(500, 3500)):
            wrapped = timer.wrap(lambda value, *, extra: value + extra, 1)
            self.assertEqual(wrapped(7, extra=4), 11)
        self.assertEqual(timer.elapsed[4], [0, 5000, 0])
        self.assertEqual(timer.cpu[4], [0, 3000, 0])
        self.assertEqual(timer.counts[4], [0, 1, 0])

    def test_missing_calls_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'Incomplete host timing'):
            probe.HostTimer().summarize(32)

    @staticmethod
    def synthetic_runs():
        rows = []
        for pair, (baseline, delta) in enumerate(zip((10, 20, 30, 40), (2, 1, 3, -2)), 1):
            for arm, value in (('plain', baseline), ('host_timed', baseline + delta)):
                timer = probe.HostTimer()
                for slot in range(31):
                    timer.counts[slot] = [2, 1, 2]
                    # Deliberately different per-generation levels; compare within each run.
                    timer.elapsed[slot] = [pair * 1000000, 100000 if slot != 15 else pair * 2000000, 10000]
                    timer.cpu[slot] = [pair * 500000, 50000 if slot != 15 else pair * 1000000, 5000]
                rows.append(dict(pair=pair, arm=arm, order=['plain', 'host_timed'],
                                 wall_ms={name: value for name in probe.WINDOWS}, **timer.summarize(2)))
        return rows

    def test_pair_differences_precede_median(self):
        result = probe.summarize(self.synthetic_runs())
        self.assertEqual(result['overhead_median']['stable']['delta_ms'], 1.5)
        self.assertEqual(result['perturbation_check'], 'exceeds_budget_review_before_interpreting')
        cooling = result['contrasts']['cooling144_minus_before']['host_elapsed_ms']['cache_finish']
        self.assertEqual(cooling['per_pair_ms'], [1.9, 3.9, 5.9, 7.9])
        self.assertAlmostEqual(cooling['median_ms'], 4.9)

    def test_small_changes_pass_but_single_step_bias_is_detected(self):
        rows = self.synthetic_runs()
        for row in rows:
            row['wall_ms'] = {name: 25.0 if row['arm'] == 'plain' else 25.1 for name in probe.WINDOWS}
        self.assertEqual(probe.summarize(rows)['perturbation_check'], 'within_budget')
        for row in rows:
            if row['arm'] == 'host_timed':
                row['wall_ms']['cooling144'] = 28.0
        result = probe.summarize(rows)
        self.assertFalse(result['overhead_checks']['cooling144']['within_budget'])
        self.assertEqual(result['perturbation_check'], 'exceeds_budget_review_before_interpreting')


if __name__ == '__main__':
    unittest.main()
