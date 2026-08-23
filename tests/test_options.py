import unittest

from core.checkpoint import make_run_id
from core.options import (
    ExperimentOptions,
    ExperimentRequest,
    build_experiment_request,
)


class ExperimentOptionsTests(unittest.TestCase):
    def test_optional_improvements_are_disabled_by_default(self):
        options = ExperimentOptions()
        self.assertFalse(options.tune_baseline_threshold)
        self.assertFalse(options.xgb_class_weight_enabled)
        self.assertFalse(options.xgb_max_delta_step_enabled)
        self.assertFalse(options.patient_balanced_weights)
        self.assertFalse(options.negative_downsample_enabled)
        self.assertFalse(options.log_power)
        self.assertFalse(options.relative_power)
        self.assertFalse(options.robust_normalize)
        self.assertEqual(options.temporal_context_seconds, 0)

    def test_builder_parses_grids_and_routes_lstm_aliases(self):
        request = build_experiment_request(
            selected_subjects=("p1", "p2", "p3"),
            lambda_values="0.5, 1.0",
            qubo_threshold_values="0.3,0.6",
            baseline_threshold_grid="0.01, 0.1",
            tune_baseline_threshold=True,
            lstm_hidden=128,
        )
        self.assertEqual(request.lambda_values, (0.5, 1.0))
        self.assertEqual(request.qubo_threshold_values, (0.3, 0.6))
        self.assertEqual(request.options.baseline_threshold_grid, (0.01, 0.1))
        self.assertEqual(request.lstm.hidden_dim, 128)

    def test_runtime_flags_do_not_change_run_identity(self):
        first = ExperimentRequest(
            selected_subjects=("p1", "p2", "p3"), save_pkl=True,
            resume_enabled=True, reuse_validation_cache=True,
        )
        second = ExperimentRequest(
            selected_subjects=("p1", "p2", "p3"), save_pkl=False,
            resume_enabled=False, reuse_validation_cache=False,
        )
        self.assertEqual(
            make_run_id(first.semantic_config()),
            make_run_id(second.semantic_config()),
        )

    def test_subject_order_does_not_change_run_identity(self):
        first = ExperimentRequest(selected_subjects=("p3", "p1", "p2"))
        second = ExperimentRequest(selected_subjects=("p1", "p2", "p3"))
        self.assertEqual(
            make_run_id(first.semantic_config()),
            make_run_id(second.semantic_config()),
        )

    def test_semantic_option_changes_run_identity(self):
        first = ExperimentRequest(selected_subjects=("p1", "p2", "p3"))
        second = ExperimentRequest(
            selected_subjects=("p1", "p2", "p3"),
            options=ExperimentOptions(log_power=True),
        )
        self.assertNotEqual(
            make_run_id(first.semantic_config()),
            make_run_id(second.semantic_config()),
        )

    def test_inactive_option_values_do_not_change_run_identity(self):
        first = ExperimentRequest(selected_subjects=("p1", "p2", "p3"))
        second = ExperimentRequest(
            selected_subjects=("p1", "p2", "p3"),
            options=ExperimentOptions(
                baseline_threshold_grid=(0.25,),
                xgb_scale_pos_weight=500,
                xgb_max_delta_step=1,
                negative_keep_fraction=0.1,
            ),
        )
        self.assertEqual(
            make_run_id(first.semantic_config()),
            make_run_id(second.semantic_config()),
        )

    def test_invalid_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            ExperimentOptions(negative_keep_fraction=0)


if __name__ == "__main__":
    unittest.main()
