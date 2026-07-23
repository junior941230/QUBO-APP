import unittest

from pipeline_runner.experiment import _is_retryable_skip


class RetryableSkipTests(unittest.TestCase):
    def test_subject_cache_failure_is_retryable(self):
        self.assertTrue(
            _is_retryable_skip(
                "chb10_31.edf: subject-level cache build failed (CUDA unavailable)"
            )
        )

    def test_file_level_failure_remains_completed(self):
        self.assertFalse(
            _is_retryable_skip("chb10_31.edf: inference failed (invalid data)")
        )


if __name__ == "__main__":
    unittest.main()
