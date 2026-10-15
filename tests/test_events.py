import unittest

import numpy as np

from qubo.events import events, file_event_counts
from qubo.tuning import tune_qubo_params_from_cache


def threshold_solver(scores, lmbda=0.0, threshold=0.5):
    del lmbda
    return (np.asarray(scores) >= threshold).astype(int)


class EventCountingTests(unittest.TestCase):
    def test_events_merge_short_gaps(self):
        mask = [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]
        self.assertEqual(events(mask), [(1, 3), (5, 6), (10, 11)])
        self.assertEqual(events(mask, merge_gap=2), [(1, 6), (10, 11)])
        self.assertEqual(events([0, 0]), [])

    def test_detection_latency_and_false_alarms(self):
        y_true = np.zeros(60, dtype=int)
        y_true[20:30] = 1
        y_pred = np.zeros(60, dtype=int)
        y_pred[24:26] = 1   # detects the seizure 4 epochs after onset
        y_pred[50:52] = 1   # false alarm
        n_true, detected, latencies, false_alarms = file_event_counts(y_true, y_pred)
        self.assertEqual((n_true, detected, false_alarms), (1, 1, 1))
        self.assertEqual(latencies, [4.0])


class EventObjectiveTests(unittest.TestCase):
    def test_event_objective_trades_sensitivity_for_false_alarms(self):
        # One seizure file: threshold 0.3 detects it, 0.7 misses it.
        seizure = {
            "scores": np.r_[np.full(10, 0.1), np.full(5, 0.5), np.full(10, 0.1)],
            "y_val": np.r_[np.zeros(10), np.ones(5), np.zeros(10)].astype(int),
        }
        # One hour without seizures: threshold 0.3 raises 30 separate alarms.
        quiet_scores = np.full(3600, 0.1)
        quiet_scores[::120] = 0.4
        quiet = {"scores": quiet_scores, "y_val": np.zeros(3600, dtype=int)}
        cache = {"a_seizure": seizure, "a_quiet": quiet}
        subjects = {"a_seizure": "a", "a_quiet": "a"}

        def best_threshold(alpha):
            return tune_qubo_params_from_cache(
                cache, threshold_solver, [0.0], [0.3, 0.7], alpha=alpha,
                objective="event", file_to_subject=subjects,
            )[1]

        # Detecting the seizure is worth 1.0; 30 alarms per hour cost 30 * alpha.
        self.assertEqual(best_threshold(0.01), 0.3)
        self.assertEqual(best_threshold(0.1), 0.7)


if __name__ == "__main__":
    unittest.main()
