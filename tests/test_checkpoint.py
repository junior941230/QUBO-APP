import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from core.checkpoint import load_checkpoint, make_run_id, save_checkpoint


class CheckpointIdentityTests(unittest.TestCase):
    def test_schema_version_changes_run_id(self):
        config = {
            "subjects": ["chb01"],
            "baseline": "svm",
            "solver_name": "solve_chain_qubo_exact",
            "tune_mode": "group_nfold",
            "tune_n_splits": 5,
            "max_files_per_subject": 0,
            "lambda_list": [0.5],
            "threshold_list": [0.5],
            "reuse_global_cache": True,
        }

        old_id = make_run_id({**config, "run_schema_version": 1})
        new_id = make_run_id({**config, "run_schema_version": 2})

        self.assertNotEqual(old_id, new_id)

    def test_lstm_params_change_run_id(self):
        config = {
            "run_schema_version": 3,
            "subjects": ["chb01"],
            "baseline": "lstm",
            "solver_name": "solve_chain_qubo_exact",
            "tune_mode": "group_nfold",
            "tune_n_splits": 5,
            "max_files_per_subject": 0,
            "lambda_list": [0.5],
            "threshold_list": [0.5],
            "reuse_global_cache": True,
            "lstm_params": {
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "epochs": 5,
                "lr": 0.001,
                "batch_size": 16,
            },
        }

        changed_config = {
            **config,
            "lstm_params": {**config["lstm_params"], "hidden_dim": 512, "epochs": 100},
        }

        self.assertNotEqual(make_run_id(config), make_run_id(changed_config))

    def test_load_checkpoint_rejects_mismatched_config(self):
        config = {
            "run_schema_version": 3,
            "subjects": ["chb01"],
            "baseline": "lstm",
            "lstm_params": {"hidden_dim": 32, "epochs": 5},
        }
        changed_config = {
            **config,
            "lstm_params": {"hidden_dim": 512, "epochs": 100},
        }

        with TemporaryDirectory() as tmp_dir, patch("core.checkpoint.CHECKPOINT_DIR", Path(tmp_dir)):
            run_id = "test-run"
            save_checkpoint(run_id, [], {}, [], config)

            self.assertIsNone(load_checkpoint(run_id, expected_config=changed_config))
            self.assertIsNotNone(load_checkpoint(run_id, expected_config=config))

    def test_random_seed_changes_run_id(self):
        config = {
            "run_schema_version": 4,
            "subjects": ["chb01"],
            "baseline": "lstm",
            "solver_name": "solve_qubo_seizure",
            "random_seed": 42,
            "lstm_params": {"hidden_dim": 32, "epochs": 5},
        }

        self.assertNotEqual(
            make_run_id(config), make_run_id({**config, "random_seed": 7})
        )


if __name__ == "__main__":
    unittest.main()
