"""Leak-free inner-validation score caches."""

import numpy as np

from config import RANDOM_SEED
from core.logging_utils import log_step
from core.options import ExperimentOptions
from core.splits import patient_independent_validation_splits
from models.registry import predict_scores
from models.training_data import prepare_training_data


def _cache_classical_folds(
    splits,
    features,
    labels,
    baseline,
    log_prefix,
    options=None,
    random_seed=RANDOM_SEED,
    file_to_subject=None,
):
    options = options or ExperimentOptions()
    cache = {}
    log_step(f"[{log_prefix}] folds={len(splits)}")
    for fold_index, (train_files, val_files) in enumerate(splits, start=1):
        training = prepare_training_data(
            train_files,
            features,
            labels,
            options,
            random_seed,
            file_to_subject=file_to_subject,
        )
        if len(np.unique(training.labels)) < 2:
            continue
        log_step(
            f"[{log_prefix}] fold {fold_index}/{len(splits)} "
            f"train_files={len(train_files)} val_files={len(val_files)}"
        )
        val_features = np.concatenate([features[name] for name in val_files])
        scores_all = np.asarray(predict_scores(
            baseline,
            training.features,
            training.labels,
            val_features,
            options=options,
            sample_weight=training.sample_weight,
            random_seed=random_seed,
        ))
        offset = 0
        for val_file in val_files:
            val_length = len(features[val_file])
            scores = scores_all[offset:offset + val_length]
            y_val = np.asarray(labels[val_file]).astype(int)
            if len(scores) != len(y_val):
                raise ValueError(
                    f"Score length mismatch for {val_file}: "
                    f"{len(scores)} scores vs {len(y_val)} labels"
                )
            cache[val_file] = {"scores": scores, "y_val": y_val}
            offset += val_length
    log_step(f"[{log_prefix}] done, cached_files={len(cache)}")
    return cache


def build_validation_score_cache_loso(
    candidate_files,
    features,
    labels,
    baseline,
    file_to_subject,
    random_seed=RANDOM_SEED,
    options=None,
):
    splits = patient_independent_validation_splits(
        candidate_files,
        file_to_subject,
        "loso",
        labels=labels,
        random_seed=random_seed,
    )
    return _cache_classical_folds(
        splits,
        features,
        labels,
        baseline,
        "Cache-LOSO",
        options,
        random_seed,
        file_to_subject,
    )


def build_validation_score_cache_lofo(*args, **kwargs):
    """Compatibility alias; validation is grouped by subject, not file."""
    return build_validation_score_cache_loso(*args, **kwargs)


def build_validation_score_cache_kfold(
    candidate_files,
    features,
    labels,
    baseline,
    n_splits=5,
    random_seed=RANDOM_SEED,
    file_to_subject=None,
    options=None,
):
    if file_to_subject is None:
        raise ValueError("file_to_subject is required for patient-independent validation")
    splits = patient_independent_validation_splits(
        candidate_files,
        file_to_subject,
        "group_nfold",
        n_splits=n_splits,
        labels=labels,
        random_seed=random_seed,
    )
    return _cache_classical_folds(
        splits,
        features,
        labels,
        baseline,
        "Cache-GroupKFold",
        options,
        random_seed,
        file_to_subject,
    )


def build_validation_score_cache_lstm(
    candidate_files,
    features,
    labels,
    tune_mode,
    n_splits,
    lstm_params,
    file_to_subject,
    random_seed=RANDOM_SEED,
):
    from models.lstm import _predict_lstm_sequence, _train_lstm_on_files

    splits = patient_independent_validation_splits(
        candidate_files,
        file_to_subject,
        tune_mode,
        n_splits=n_splits,
        labels=labels,
        random_seed=random_seed,
    )
    cache = {}
    log_step(f"[Cache-LSTM-Grouped] folds={len(splits)}")
    for fold_index, (train_files, val_files) in enumerate(splits, start=1):
        y_train = np.concatenate([labels[name] for name in train_files]).astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        log_step(
            f"[Cache-LSTM-Grouped] fold {fold_index}/{len(splits)} "
            f"train_files={len(train_files)} val_files={len(val_files)}"
        )
        model, mean, std, device = _train_lstm_on_files(
            train_files,
            features,
            labels,
            random_seed=random_seed,
            **(lstm_params or {}),
        )
        for val_file in val_files:
            cache[val_file] = {
                "scores": _predict_lstm_sequence(
                    model, features[val_file], mean, std, device
                ),
                "y_val": np.asarray(labels[val_file]).astype(int),
            }
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    log_step(f"[Cache-LSTM-Grouped] done, cached_files={len(cache)}")
    return cache


def build_validation_score_cache(
    candidate_files,
    features,
    labels,
    baseline,
    tune_mode,
    n_splits=5,
    lstm_params=None,
    file_to_subject=None,
    random_seed=RANDOM_SEED,
    options=None,
):
    if file_to_subject is None:
        raise ValueError("file_to_subject is required for patient-independent validation")
    normalized_mode = {"lofo": "loso", "nfold": "group_nfold"}.get(
        tune_mode, tune_mode
    )
    if baseline == "lstm":
        return build_validation_score_cache_lstm(
            candidate_files,
            features,
            labels,
            normalized_mode,
            n_splits,
            lstm_params,
            file_to_subject,
            random_seed,
        )
    if normalized_mode == "loso":
        return build_validation_score_cache_loso(
            candidate_files,
            features,
            labels,
            baseline,
            file_to_subject,
            random_seed,
            options,
        )
    if normalized_mode == "group_nfold":
        return build_validation_score_cache_kfold(
            candidate_files,
            features,
            labels,
            baseline,
            n_splits,
            random_seed,
            file_to_subject,
            options,
        )
    raise ValueError(f"Unknown patient-independent tuning mode: {tune_mode}")
