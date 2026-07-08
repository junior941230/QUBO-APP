import unittest
from unittest.mock import patch

import numpy as np

from qubo.validation_cache import build_validation_score_cache_kfold


class ValidationCacheTests(unittest.TestCase):
    def test_kfold_trains_once_per_fold_without_validation_overlap(self):
        files = [f"f{i}.edf" for i in range(6)]
        features = {
            name: np.array([[float(idx)]])
            for idx, name in enumerate(files)
        }
        labels = {
            name: np.array([idx % 2])
            for idx, name in enumerate(files)
        }
        calls = []

        def fake_predict_scores(_baseline, x_train, _y_train, x_test):
            train_ids = set(x_train[:, 0].astype(int).tolist())
            val_ids = set(x_test[:, 0].astype(int).tolist())
            calls.append((train_ids, val_ids))
            return x_test[:, 0] / 10.0

        with patch("qubo.validation_cache.predict_scores", side_effect=fake_predict_scores):
            cache = build_validation_score_cache_kfold(
                files, features, labels, "svm", n_splits=3
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(set(cache), set(files))
        for train_ids, val_ids in calls:
            self.assertTrue(val_ids)
            self.assertTrue(train_ids.isdisjoint(val_ids))
        for idx, name in enumerate(files):
            np.testing.assert_array_equal(cache[name]["y_val"], labels[name])
            np.testing.assert_allclose(cache[name]["scores"], np.array([idx / 10.0]))


if __name__ == "__main__":
    unittest.main()
