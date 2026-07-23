import unittest
from unittest.mock import Mock, patch

import numpy as np

from models import classical


class GpuPreferenceTests(unittest.TestCase):
    def test_svm_uses_cuml_when_cuda_and_cuml_are_available(self):
        fake_svc = object()
        fake_scaler = object()

        with (
            patch.object(classical, "CUM_AVAILABLE", True),
            patch.object(classical, "_cuda_available", return_value=True),
            patch.object(classical, "CuMLSVC", fake_svc),
            patch.object(classical, "CuMLStandardScaler", fake_scaler),
        ):
            svc, scaler, device = classical._svm_backend()

        self.assertIs(svc, fake_svc)
        self.assertIs(scaler, fake_scaler)
        self.assertEqual(device, "cuda (cuML)")

    def test_svm_falls_back_to_sklearn_without_cuml(self):
        with (
            patch.object(classical, "CUM_AVAILABLE", False),
            patch.object(classical, "_cuda_available", return_value=True),
        ):
            svc, scaler, device = classical._svm_backend()

        self.assertIs(svc, classical.SklearnSVC)
        self.assertIs(scaler, classical.SklearnStandardScaler)
        self.assertEqual(device, "cpu (scikit-learn)")

    def test_svm_retries_on_cpu_when_cuml_fails(self):
        expected = np.array([0.8])

        with (
            patch.object(
                classical,
                "_svm_backend",
                return_value=(object(), object(), "cuda (cuML)"),
            ),
            patch.object(
                classical,
                "_predict_svm_with_backend",
                side_effect=[RuntimeError("CUDA unavailable"), expected],
            ) as predict_backend,
        ):
            scores = classical.predict_svm(
                np.array([[0.0], [1.0]]),
                np.array([0, 1]),
                np.array([[0.5]]),
            )

        self.assertEqual(predict_backend.call_count, 2)
        self.assertIs(
            predict_backend.call_args_list[1].args[0], classical.SklearnSVC
        )
        np.testing.assert_allclose(scores, expected)

    def test_xgboost_prefers_cuda(self):
        classifier = Mock()
        classifier.predict_proba.return_value = np.array([[0.25, 0.75]])
        factory = Mock(return_value=classifier)

        with (
            patch.object(classical, "_cuda_available", return_value=True),
            patch.object(classical, "XGBClassifier", factory),
        ):
            scores = classical.predict_xgboost(
                np.array([[0.0], [1.0]]),
                np.array([0, 1]),
                np.array([[0.5]]),
            )

        self.assertEqual(factory.call_args.kwargs["device"], "cuda")
        self.assertEqual(factory.call_args.kwargs["tree_method"], "hist")
        np.testing.assert_allclose(scores, np.array([0.75]))

    def test_xgboost_uses_cpu_when_cuda_is_unavailable(self):
        classifier = Mock()
        classifier.predict_proba.return_value = np.array([[0.6, 0.4]])
        factory = Mock(return_value=classifier)

        with (
            patch.object(classical, "_cuda_available", return_value=False),
            patch.object(classical, "XGBClassifier", factory),
        ):
            classical.predict_xgboost(
                np.array([[0.0], [1.0]]),
                np.array([0, 1]),
                np.array([[0.5]]),
            )

        self.assertEqual(factory.call_args.kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
