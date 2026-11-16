import unittest
from unittest.mock import patch

import numpy as np

from core.checkpoint import make_run_id
from core.options import ExperimentRequest, TransformerOptions, build_experiment_request

try:
    import torch

    from models.transformer import (
        SeizureTransformer,
        _predict_transformer_sequence,
        _train_transformer_on_files,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _model(causal, window=4, num_layers=2, conv_kernel=3, input_dim=3):
    torch.manual_seed(0)
    model = SeizureTransformer(
        input_dim=input_dim, d_model=8, n_heads=2, num_layers=num_layers,
        ff_dim=16, dropout=0.0, window=window, conv_kernel=conv_kernel,
        causal=causal,
    )
    return model.eval()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TransformerModelTests(unittest.TestCase):
    def test_causal_outputs_ignore_future_epochs(self):
        model = _model(causal=True)
        x = torch.randn(1, 40, 3)
        changed = x.clone()
        changed[:, 25:] = torch.randn(1, 15, 3) * 10
        with torch.no_grad():
            before, after = model(x), model(changed)
        torch.testing.assert_close(before[:, :25], after[:, :25])
        self.assertFalse(torch.allclose(before[:, 25:], after[:, 25:]))

    def test_receptive_field_matches_declared_context(self):
        for causal in (True, False):
            with self.subTest(causal=causal):
                model = _model(causal=causal)
                before, after = model.context
                x = torch.randn(1, 60, 3)
                changed = x.clone()
                changed[:, 30] += 10.0
                with torch.no_grad():
                    diff = (model(x) - model(changed)).abs().squeeze(0)
                affected = torch.nonzero(diff > 1e-6).squeeze(-1).tolist()
                self.assertTrue(affected)
                # Output t depends on inputs in [t - before, t + after].
                self.assertGreaterEqual(min(affected), 30 - after)
                self.assertLessEqual(max(affected), 30 + before)

    def test_chunked_inference_equals_full_sequence(self):
        for causal in (True, False):
            with self.subTest(causal=causal):
                model = _model(causal=causal)
                seq = np.random.default_rng(1).normal(size=(53, 3)).astype(np.float32)
                mean, std = np.zeros((1, 3)), np.ones((1, 3))
                chunked = _predict_transformer_sequence(
                    model, seq, mean, std, "cpu", chunk=7
                )
                with torch.no_grad():
                    full = torch.sigmoid(
                        model(torch.from_numpy(seq).unsqueeze(0))
                    ).squeeze(0).numpy()
                np.testing.assert_allclose(chunked, full, atol=1e-6)

    def test_padding_does_not_change_real_epochs(self):
        for causal in (True, False):
            with self.subTest(causal=causal):
                model = _model(causal=causal)
                short, long = torch.randn(9, 3), torch.randn(20, 3)
                batch = torch.zeros(2, 20, 3)
                batch[0, :9], batch[1] = short, long
                valid = torch.zeros(2, 20)
                valid[0, :9], valid[1] = 1.0, 1.0
                with torch.no_grad():
                    padded = model(batch, valid=valid)
                    alone = model(short.unsqueeze(0)).squeeze(0)
                self.assertFalse(torch.isnan(padded).any())
                torch.testing.assert_close(padded[0, :9], alone)
                torch.testing.assert_close(
                    padded[1], model(long.unsqueeze(0)).squeeze(0)
                )

    def test_training_is_deterministic_and_learns_a_separable_signal(self):
        rng = np.random.default_rng(0)
        files = [f"f{i}.edf" for i in range(4)]
        features, labels = {}, {}
        for name in files:
            y = np.zeros(90, dtype=int)
            start = rng.integers(10, 60)
            y[start:start + 20] = 1
            features[name] = (
                rng.normal(size=(90, 3)) + 3.0 * y[:, None]
            ).astype(np.float32)
            labels[name] = y
        params = dict(
            d_model=8, n_heads=2, num_layers=1, ff_dim=16, dropout=0.0,
            window_seconds=8, conv_kernel=3, segment_seconds=30, epochs=15,
            lr=1e-2, batch_size=4, device="cpu",
        )
        runs = []
        for _ in range(2):
            model, mean, std, device = _train_transformer_on_files(
                files[:3], features, labels, random_seed=7, **params
            )
            runs.append(_predict_transformer_sequence(
                model, features[files[3]], mean, std, device
            ))
        np.testing.assert_array_equal(runs[0], runs[1])
        probs, y_test = runs[0], labels[files[3]]
        self.assertEqual(probs.shape, y_test.shape)
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))
        self.assertGreater(probs[y_test == 1].mean(), probs[y_test == 0].mean() + 0.3)


class TransformerOptionsTests(unittest.TestCase):
    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            TransformerOptions(d_model=30, n_heads=4)
        with self.assertRaises(ValueError):
            TransformerOptions(causal=False, conv_kernel=4)
        with self.assertRaises(ValueError):
            TransformerOptions(window_seconds=0)

    def test_builder_routes_flat_transformer_fields(self):
        request = build_experiment_request(
            selected_subjects=("p1", "p2", "p3"),
            baseline="transformer",
            transformer_window_seconds=60,
            transformer_causal=False,
            transformer_conv_kernel=3,
        )
        self.assertEqual(request.transformer.window_seconds, 60)
        self.assertFalse(request.transformer.causal)
        self.assertEqual(request.sequence_params()["window_seconds"], 60)

    def test_transformer_settings_enter_run_identity_only_when_selected(self):
        xgb = ExperimentRequest(selected_subjects=("p1", "p2", "p3"), baseline="xgboost")
        self.assertNotIn("transformer", xgb.semantic_config())
        tweaked = ExperimentRequest(
            selected_subjects=("p1", "p2", "p3"), baseline="xgboost",
            transformer=TransformerOptions(window_seconds=30),
        )
        self.assertEqual(
            make_run_id(xgb.semantic_config()), make_run_id(tweaked.semantic_config())
        )
        first = ExperimentRequest(selected_subjects=("p1", "p2", "p3"), baseline="transformer")
        second = ExperimentRequest(
            selected_subjects=("p1", "p2", "p3"), baseline="transformer",
            transformer=TransformerOptions(window_seconds=30),
        )
        self.assertNotEqual(
            make_run_id(first.semantic_config()), make_run_id(second.semantic_config())
        )

    def test_sample_weighting_options_are_rejected(self):
        with self.assertRaises(ValueError):
            build_experiment_request(
                selected_subjects=("p1", "p2", "p3"),
                baseline="transformer",
                negative_downsample_enabled=True,
                negative_keep_fraction=0.25,
            )


class TransformerValidationCacheTests(unittest.TestCase):
    def test_inner_folds_never_train_on_their_validation_patients(self):
        from qubo.validation_cache import build_validation_score_cache

        files = [f"p{i}_{part}.edf" for i in range(4) for part in "ab"]
        file_to_subject = {name: name.split("_")[0] for name in files}
        features = {name: np.zeros((2, 1)) for name in files}
        labels = {name: np.array([0, 1]) for name in files}
        trained_on = []

        def fake_train(train_files, *_args, **kwargs):
            trained_on.append((set(train_files), kwargs))
            return object(), 0.0, 1.0, "cpu"

        with patch(
            "models.transformer._train_transformer_on_files", side_effect=fake_train
        ), patch(
            "models.transformer._predict_transformer_sequence",
            return_value=np.array([0.2, 0.8]),
        ):
            cache = build_validation_score_cache(
                files, features, labels, "transformer", tune_mode="group_nfold",
                n_splits=2, sequence_params={"window_seconds": 30},
                file_to_subject=file_to_subject,
            )

        self.assertEqual(set(cache), set(files))
        self.assertEqual(len(trained_on), 2)
        for train_files, kwargs in trained_on:
            self.assertEqual(kwargs["window_seconds"], 30)
            val_subjects = {file_to_subject[n] for n in files} - {
                file_to_subject[n] for n in train_files
            }
            self.assertTrue(val_subjects)


if __name__ == "__main__":
    unittest.main()
