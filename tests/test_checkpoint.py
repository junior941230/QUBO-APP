import unittest

from core.checkpoint import make_run_id


class CheckpointIdentityTests(unittest.TestCase):
    def test_schema_version_changes_run_id(self):
        config = {
            "subjects": ["chb01"],
            "baseline": "svm",
            "solver_name": "solve_chain_qubo_exact",
            "tune_mode": "nfold",
            "tune_n_splits": 5,
            "max_files_per_subject": 0,
            "lambda_list": [0.5],
            "threshold_list": [0.5],
            "reuse_global_cache": True,
        }

        old_id = make_run_id({**config, "run_schema_version": 1})
        new_id = make_run_id({**config, "run_schema_version": 2})

        self.assertNotEqual(old_id, new_id)


if __name__ == "__main__":
    unittest.main()
