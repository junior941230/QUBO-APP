import unittest
from unittest.mock import patch

try:
    import numpy as np
    from pipeline_runner.experiment import run_experiment
except (ImportError, ModuleNotFoundError, NameError):
    np = None
    run_experiment = None


@unittest.skipUnless(run_experiment is not None, "project runtime dependencies unavailable")
class ExperimentIsolationTests(unittest.TestCase):
    def test_precomputed_caches_exclude_each_outer_test_file(self):
        files = ["a.edf", "b.edf", "c.edf", "d.edf"]
        paths = [f"/fake/{name}" for name in files]
        features = {name: np.array([[0.0], [1.0]]) for name in files}
        labels = {name: np.array([0, 1]) for name in files}
        cache_candidates = []

        def fake_cache_builder(candidate_files, *_args, **_kwargs):
            cache_candidates.append(tuple(candidate_files))
            return {
                name: {
                    "scores": np.array([0.1, 0.9]),
                    "y_val": np.array([0, 1]),
                }
                for name in candidate_files
            }

        def threshold_solver(scores, lmbda=0.0, threshold=0.5):
            del lmbda
            return np.asarray(scores) >= threshold

        patches = (
            patch(
                "pipeline_runner.experiment.collect_files_and_seizures",
                return_value=(paths, {}, []),
            ),
            patch(
                "pipeline_runner.experiment.processAllFiles",
                return_value=(features, labels),
            ),
            patch(
                "pipeline_runner.experiment.validate_edf_channels",
                return_value=(paths, {}),
            ),
            patch(
                "pipeline_runner.experiment.build_validation_score_cache",
                side_effect=fake_cache_builder,
            ),
            patch(
                "pipeline_runner.experiment.predict_scores",
                return_value=np.array([0.1, 0.9]),
            ),
            patch(
                "pipeline_runner.experiment.get_qubo_solver",
                return_value=threshold_solver,
            ),
            patch("pipeline_runner.experiment.save_checkpoint"),
            patch("pipeline_runner.experiment.build_summary_plot", return_value=None),
            patch("pipeline_runner.experiment.build_detail_plot", return_value=None),
        )

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            _, result_df, _, _, _, _ = run_experiment(
                ["chb01"], "svm", "solve_chain_qubo_exact", "lofo", 2,
                0, 1, "0.0", "0.5", True, False, False,
                32, 1, 1, 1e-3, 1, 0.0, False,
                progress=lambda *_args, **_kwargs: None,
            )

        expected_train_sets = {
            tuple(name for name in files if name != test_file)
            for test_file in files
        }
        self.assertEqual(set(cache_candidates), expected_train_sets)
        self.assertEqual(len(cache_candidates), len(files))
        self.assertEqual(len(result_df), len(files))


if __name__ == "__main__":
    unittest.main()
