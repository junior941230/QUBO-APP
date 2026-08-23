import unittest

import numpy as np

from models.selection import tune_baseline_threshold_from_cache


class BaselineSelectionTests(unittest.TestCase):
    def test_threshold_uses_seizure_f1_and_nonseizure_fp_penalty(self):
        cache = {
            "seizure.edf": {
                "scores": np.array([0.1, 0.2]),
                "y_val": np.array([0, 1]),
            },
            "normal.edf": {
                "scores": np.array([0.15, 0.05]),
                "y_val": np.array([0, 0]),
            },
        }
        result = tune_baseline_threshold_from_cache(cache, [0.1, 0.2], alpha=0.2)
        self.assertEqual(result["threshold"], 0.2)
        self.assertEqual(result["seizure_macro_f1"], 1.0)
        self.assertEqual(result["nonseizure_fp_rate"], 0.0)

    def test_empty_cache_is_rejected(self):
        with self.assertRaises(ValueError):
            tune_baseline_threshold_from_cache({}, [0.5])


if __name__ == "__main__":
    unittest.main()
