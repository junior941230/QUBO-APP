import unittest

import numpy as np

from features import (
    BANDS,
    add_causal_context,
    extract_band_power,
    robust_normalize_features,
)


class FeatureTests(unittest.TestCase):
    def test_relative_band_power_sums_to_one_per_channel(self):
        rng = np.random.default_rng(42)
        epoch = rng.normal(size=(3, 256))
        result = extract_band_power(epoch, relative_power=True)
        channel_band = result.reshape(len(BANDS), 3).T
        np.testing.assert_allclose(channel_band.sum(axis=1), np.ones(3))

    def test_log_power_is_finite(self):
        result = extract_band_power(np.zeros((2, 256)), log_power=True)
        self.assertTrue(np.isfinite(result).all())

    def test_robust_normalization_has_unit_iqr(self):
        values = np.array([[0.0], [1.0], [2.0], [100.0]])
        normalized = robust_normalize_features(values)
        q25, q75 = np.percentile(normalized[:, 0], [25, 75])
        self.assertAlmostEqual(float(q75 - q25), 1.0)

    def test_temporal_context_is_causal(self):
        values = np.array([[1.0], [3.0], [9.0]])
        context = add_causal_context(values, 2)
        np.testing.assert_allclose(context[:, 1], np.array([1.0, 2.0, 6.0]))
        changed_future = add_causal_context(np.array([[1.0], [3.0], [99.0]]), 2)
        np.testing.assert_allclose(context[:2], changed_future[:2])


if __name__ == "__main__":
    unittest.main()
