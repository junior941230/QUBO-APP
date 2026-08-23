from dataclasses import dataclass
import hashlib
import os

import numpy as np


@dataclass(frozen=True)
class PreparedTrainingData:
    features: np.ndarray
    labels: np.ndarray
    sample_weight: np.ndarray | None


def _stable_seed(random_seed, file_name):
    digest = hashlib.sha256(f"{random_seed}:{file_name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _selected_indices(labels, keep_fraction, random_seed, file_name):
    labels = np.asarray(labels)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    keep_count = int(np.ceil(len(negative) * keep_fraction))
    if keep_count >= len(negative):
        return np.arange(len(labels))
    rng = np.random.default_rng(_stable_seed(random_seed, file_name))
    kept_negative = rng.choice(negative, size=keep_count, replace=False)
    return np.sort(np.concatenate([positive, kept_negative]))


def prepare_training_data(
    train_files,
    features_by_file,
    labels_by_file,
    options,
    random_seed,
    file_to_subject=None,
):
    feature_parts = []
    label_parts = []
    patient_parts = []
    for file_name in train_files:
        labels = np.asarray(labels_by_file[file_name])
        indices = np.arange(len(labels))
        if options.negative_downsample_enabled:
            indices = _selected_indices(
                labels,
                options.negative_keep_fraction,
                random_seed,
                file_name,
            )
        feature_parts.append(np.asarray(features_by_file[file_name])[indices])
        label_parts.append(labels[indices])
        patient = (
            file_to_subject[file_name]
            if file_to_subject is not None
            else os.path.basename(file_name).split("_")[0]
        )
        patient_parts.append(np.full(len(indices), patient, dtype=object))

    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    sample_weight = None
    if options.patient_balanced_weights:
        patients = np.concatenate(patient_parts)
        subjects, counts = np.unique(patients, return_counts=True)
        target_total = len(patients) / len(subjects)
        per_subject = dict(zip(subjects, target_total / counts))
        sample_weight = np.asarray([per_subject[patient] for patient in patients])
    return PreparedTrainingData(features, labels, sample_weight)
