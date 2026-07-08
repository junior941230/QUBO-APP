import unittest

from core.splits import leave_one_file_out_train_sets


class LeaveOneFileOutTests(unittest.TestCase):
    def test_outer_test_file_never_appears_in_its_training_set(self):
        files = ["a.edf", "b.edf", "c.edf", "d.edf"]

        train_sets = leave_one_file_out_train_sets(files)

        self.assertEqual(set(train_sets), set(files))
        for test_file, train_files in train_sets.items():
            self.assertNotIn(test_file, train_files)
            self.assertEqual(len(train_files), len(files) - 1)

    def test_duplicate_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            leave_one_file_out_train_sets(["a.edf", "a.edf"])


if __name__ == "__main__":
    unittest.main()
