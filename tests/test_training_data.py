import unittest

import numpy as np

from core.options import ExperimentOptions
from models.training_data import prepare_training_data


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.files = ["p1_a.edf", "p2_a.edf"]
        self.features = {
            "p1_a.edf": np.arange(8).reshape(4, 2),
            "p2_a.edf": np.arange(8, 20).reshape(6, 2),
        }
        self.labels = {
            "p1_a.edf": np.array([0, 0, 0, 1]),
            "p2_a.edf": np.array([0, 0, 0, 0, 0, 1]),
        }
        self.subjects = {"p1_a.edf": "p1", "p2_a.edf": "p2"}

    def test_negative_downsampling_is_deterministic_and_keeps_positives(self):
        options = ExperimentOptions(
            negative_downsample_enabled=True, negative_keep_fraction=0.25
        )
        first = prepare_training_data(
            self.files, self.features, self.labels, options, 42, self.subjects
        )
        second = prepare_training_data(
            self.files, self.features, self.labels, options, 42, self.subjects
        )
        np.testing.assert_array_equal(first.features, second.features)
        self.assertEqual(int(first.labels.sum()), 2)
        self.assertEqual(len(first.labels), 5)

    def test_patient_balanced_weights_give_equal_subject_totals(self):
        options = ExperimentOptions(patient_balanced_weights=True)
        prepared = prepare_training_data(
            self.files, self.features, self.labels, options, 42, self.subjects
        )
        self.assertAlmostEqual(float(prepared.sample_weight[:4].sum()), 5.0)
        self.assertAlmostEqual(float(prepared.sample_weight[4:].sum()), 5.0)


if __name__ == "__main__":
    unittest.main()
