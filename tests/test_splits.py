import unittest

import numpy as np

from core.splits import (
    leave_one_subject_out_splits,
    patient_independent_validation_splits,
)


class PatientIndependentSplitTests(unittest.TestCase):
    def setUp(self):
        self.files = [
            "p1_a.edf", "p1_b.edf",
            "p2_a.edf", "p2_b.edf",
            "p3_a.edf", "p3_b.edf",
            "p4_a.edf", "p4_b.edf",
        ]
        self.file_to_subject = {
            name: name.split("_")[0] for name in self.files
        }

    def test_outer_loso_never_crosses_subjects(self):
        splits = leave_one_subject_out_splits(
            self.files, self.file_to_subject
        )

        self.assertEqual(set(splits), {"p1", "p2", "p3", "p4"})
        for test_subject, split in splits.items():
            train_subjects = {
                self.file_to_subject[name] for name in split["train_files"]
            }
            test_subjects = {
                self.file_to_subject[name] for name in split["test_files"]
            }
            self.assertEqual(test_subjects, {test_subject})
            self.assertNotIn(test_subject, train_subjects)
            self.assertEqual(
                set(split["train_files"]) | set(split["test_files"]),
                set(self.files),
            )

    def test_grouped_inner_folds_never_cross_subjects(self):
        labels = {
            name: np.array([0, int(index % 2 == 0)])
            for index, name in enumerate(self.files)
        }
        folds = patient_independent_validation_splits(
            self.files,
            self.file_to_subject,
            "group_nfold",
            n_splits=2,
            labels=labels,
            random_seed=7,
        )

        self.assertEqual(len(folds), 2)
        seen_validation_files = set()
        for train_files, val_files in folds:
            train_subjects = {self.file_to_subject[name] for name in train_files}
            val_subjects = {self.file_to_subject[name] for name in val_files}
            self.assertTrue(train_subjects.isdisjoint(val_subjects))
            seen_validation_files.update(val_files)
        self.assertEqual(seen_validation_files, set(self.files))

    def test_inner_loso_holds_out_all_files_for_one_subject(self):
        folds = patient_independent_validation_splits(
            self.files, self.file_to_subject, "loso"
        )

        self.assertEqual(len(folds), 4)
        for train_files, val_files in folds:
            train_subjects = {self.file_to_subject[name] for name in train_files}
            val_subjects = {self.file_to_subject[name] for name in val_files}
            self.assertEqual(len(val_subjects), 1)
            self.assertTrue(train_subjects.isdisjoint(val_subjects))
            self.assertEqual(len(val_files), 2)

    def test_subject_identity_is_required_for_every_file(self):
        with self.assertRaisesRegex(ValueError, "Missing subject identity"):
            leave_one_subject_out_splits(
                self.files, {self.files[0]: "p1"}
            )

    def test_nested_validation_requires_two_training_subjects(self):
        with self.assertRaisesRegex(ValueError, "at least 2 training subjects"):
            patient_independent_validation_splits(
                self.files[:2], self.file_to_subject, "loso"
            )


if __name__ == "__main__":
    unittest.main()
