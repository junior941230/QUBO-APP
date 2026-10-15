import unittest

import numpy as np

from qubo.tuning import tune_qubo_params_from_cache


def threshold_solver(scores, lmbda=0.0, threshold=0.5):
    del lmbda
    return (np.asarray(scores) >= threshold).astype(int)


class QuboTuningObjectiveTests(unittest.TestCase):
    def setUp(self):
        # Patient "a" has six seizure files that only threshold 0.3 detects.
        # Patients "b" and "c" have one file each where 0.3 floods false alarms
        # (F1 2/7) and only 0.7 is right (F1 1).
        low = {"scores": np.array([0.4, 0.1, 0.1]), "y_val": np.array([1, 0, 0])}
        high = {
            "scores": np.array([0.8, 0.5, 0.5, 0.5, 0.5, 0.5]),
            "y_val": np.array([1, 0, 0, 0, 0, 0]),
        }
        self.cache = {f"a{i}": low for i in range(6)}
        self.cache.update({"b1": high, "c1": high})
        self.subjects = {name: name[0] for name in self.cache}

    def tune(self, objective):
        return tune_qubo_params_from_cache(
            self.cache, threshold_solver, [0.0], [0.3, 0.7],
            objective=objective, file_to_subject=self.subjects,
        )

    def test_file_mean_lets_patients_with_more_files_dominate(self):
        _, threshold, score = self.tune("file_mean")
        self.assertEqual(threshold, 0.3)
        self.assertAlmostEqual(score, (6 * 1.0 + 2 * 2 / 7) / 8)

    def test_patient_macro_weights_each_patient_equally(self):
        _, threshold, score = self.tune("patient_macro")
        self.assertEqual(threshold, 0.7)
        self.assertAlmostEqual(score, (0.0 + 1.0 + 1.0) / 3)

    def test_patient_macro_requires_subject_mapping(self):
        with self.assertRaises(ValueError):
            tune_qubo_params_from_cache(
                self.cache, threshold_solver, [0.0], [0.3], objective="patient_macro",
            )

    def test_unknown_objective_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tune("median")


if __name__ == "__main__":
    unittest.main()
