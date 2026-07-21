from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def _group_files_by_subject(file_names, file_to_subject):
    """Validate and group EDF keys without inferring identity from filenames."""
    names = list(file_names)
    if len(names) != len(set(names)):
        raise ValueError("File names must be unique")

    missing = [name for name in names if name not in file_to_subject]
    if missing:
        raise ValueError(
            "Missing subject identity for: " + ", ".join(sorted(missing))
        )

    grouped = defaultdict(list)
    for name in names:
        subject = str(file_to_subject[name]).strip()
        if not subject:
            raise ValueError(f"Empty subject identity for {name}")
        grouped[subject].append(name)
    return dict(grouped)


def leave_one_subject_out_splits(file_names, file_to_subject):
    """Return outer folds whose train and test patients are disjoint."""
    names = list(file_names)
    grouped = _group_files_by_subject(names, file_to_subject)
    if len(grouped) < 2:
        raise ValueError("Patient-independent evaluation needs at least 2 subjects")

    return {
        subject: {
            "train_files": [
                name for name in names if str(file_to_subject[name]).strip() != subject
            ],
            "test_files": list(test_files),
        }
        for subject, test_files in sorted(grouped.items())
    }


def patient_independent_validation_splits(
    candidate_files,
    file_to_subject,
    mode,
    n_splits=5,
    labels=None,
    random_seed=42,
):
    """Build inner folds by subject, never by EDF recording.

    ``loso`` holds out one training subject at a time. ``group_nfold`` splits
    the available subjects into deterministic folds and then expands each fold
    back to EDF files. Legacy ``lofo``/``nfold`` values are accepted as aliases,
    but retain the patient-grouped semantics.
    """
    files = list(candidate_files)
    grouped = _group_files_by_subject(files, file_to_subject)
    subjects = np.asarray(sorted(grouped), dtype=object)
    if len(subjects) < 2:
        raise ValueError(
            "Patient-independent validation needs at least 2 training subjects"
        )

    normalized_mode = {"lofo": "loso", "nfold": "group_nfold"}.get(mode, mode)
    if normalized_mode == "loso":
        subject_folds = [([s for s in subjects if s != val], [val]) for val in subjects]
    elif normalized_mode == "group_nfold":
        effective = min(max(2, int(n_splits)), len(subjects))
        subject_has_seizure = None
        if labels is not None:
            subject_has_seizure = np.asarray([
                int(any(np.asarray(labels[name]).sum() > 0 for name in grouped[subject]))
                for subject in subjects
            ])

        class_counts = (
            np.bincount(subject_has_seizure, minlength=2)
            if subject_has_seizure is not None
            else np.asarray([0, 0])
        )
        if np.min(class_counts) >= effective:
            splitter = StratifiedKFold(
                n_splits=effective,
                shuffle=True,
                random_state=int(random_seed),
            )
            split_iter = splitter.split(subjects, subject_has_seizure)
        else:
            splitter = KFold(
                n_splits=effective,
                shuffle=True,
                random_state=int(random_seed),
            )
            split_iter = splitter.split(subjects)
        subject_folds = [
            (subjects[train_idx].tolist(), subjects[val_idx].tolist())
            for train_idx, val_idx in split_iter
        ]
    else:
        raise ValueError(f"Unknown patient-independent tuning mode: {mode}")

    folds = []
    for train_subjects, val_subjects in subject_folds:
        train_subjects = set(train_subjects)
        val_subjects = set(val_subjects)
        if train_subjects & val_subjects:
            raise AssertionError("Subject leakage detected while building validation folds")
        folds.append((
            [
                name for name in files
                if str(file_to_subject[name]).strip() in train_subjects
            ],
            [
                name for name in files
                if str(file_to_subject[name]).strip() in val_subjects
            ],
        ))
    return folds
