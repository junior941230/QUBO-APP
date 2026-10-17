import unittest

import numpy as np

from features import (
    BANDS,
    add_causal_context,
    causal_robust_normalize_features,
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

    def test_causal_robust_normalization_never_uses_future_epochs(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(200, 3))
        for window in (0, 50):
            reference = causal_robust_normalize_features(values, window, 20)
            for cut in (20, 21, 99, 150):
                changed = values.copy()
                changed[cut:] = rng.normal(loc=1e6, size=changed[cut:].shape)
                result = causal_robust_normalize_features(changed, window, 20)
                # Epoch t depends on epochs <= t only, so everything before cut is unchanged.
                np.testing.assert_array_equal(result[:cut], reference[:cut])

    def test_causal_robust_normalization_uses_only_earlier_history(self):
        values = np.arange(10.0)[:, None] ** 2
        result = causal_robust_normalize_features(values, window_epochs=0, warmup_epochs=4)
        np.testing.assert_array_equal(result[:4], 0.0)
        for index in range(4, 10):
            history = values[:index, 0]
            q25, median, q75 = np.percentile(history, [25, 50, 75])
            self.assertAlmostEqual(result[index, 0], (values[index, 0] - median) / (q75 - q25))

    def test_causal_robust_sliding_window_drops_old_epochs(self):
        values = np.concatenate([np.full(50, 1000.0), np.arange(50.0)])[:, None]
        result = causal_robust_normalize_features(values, window_epochs=10, warmup_epochs=5)
        history = values[89:99, 0]
        q25, median, q75 = np.percentile(history, [25, 50, 75])
        self.assertAlmostEqual(result[99, 0], (values[99, 0] - median) / (q75 - q25))

    def test_causal_robust_constant_history_keeps_unit_scale(self):
        result = causal_robust_normalize_features(np.ones((5, 2)), warmup_epochs=1)
        np.testing.assert_array_equal(result, np.zeros((5, 2)))

    def test_temporal_context_is_causal(self):
        values = np.array([[1.0], [3.0], [9.0]])
        context = add_causal_context(values, 2)
        np.testing.assert_allclose(context[:, 1], np.array([1.0, 2.0, 6.0]))
        changed_future = add_causal_context(np.array([[1.0], [3.0], [99.0]]), 2)
        np.testing.assert_allclose(context[:2], changed_future[:2])


if __name__ == "__main__":
    unittest.main()
