import numpy as np

from config import RANDOM_SEED
from core.logging_utils import log_step
from core.splits import patient_independent_validation_splits
from models.registry import predict_scores
from models.lstm import _predict_lstm_sequence, _train_lstm_on_files

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _cache_classical_folds(splits, features, labels, baseline, log_prefix):
    cache = {}
    log_step(f"[{log_prefix}] folds={len(splits)}")
    for fold_idx, (inner_train_files, val_files) in enumerate(splits, start=1):
        x_train = np.concatenate([features[name] for name in inner_train_files])
        y_train = np.concatenate([labels[name] for name in inner_train_files]).astype(int)
        if len(np.unique(y_train)) < 2:
            continue

        log_step(
            f"[{log_prefix}] fold {fold_idx}/{len(splits)} "
            f"train_files={len(inner_train_files)} val_files={len(val_files)}"
        )
        x_val_all = np.concatenate([features[name] for name in val_files])
        scores_all = np.asarray(predict_scores(baseline, x_train, y_train, x_val_all))

        offset = 0
        for val_file in val_files:
            val_len = len(features[val_file])
            scores = scores_all[offset:offset + val_len]
            y_val = np.asarray(labels[val_file]).astype(int)
            if scores.shape[0] != y_val.shape[0]:
                raise ValueError(
                    f"Score length mismatch for {val_file}: "
                    f"{scores.shape[0]} scores vs {y_val.shape[0]} labels"
                )
            cache[val_file] = {"scores": scores, "y_val": y_val}
            offset += val_len

    log_step(f"[{log_prefix}] done, cached_files={len(cache)}")
    return cache


def build_validation_score_cache_loso(
    candidate_files,
    features,
    labels,
    baseline,
    file_to_subject,
    random_seed=RANDOM_SEED,
):
    splits = patient_independent_validation_splits(
        candidate_files,
        file_to_subject,
        "loso",
        labels=labels,
        random_seed=random_seed,
    )
    return _cache_classical_folds(
        splits, features, labels, baseline, "Cache-LOSO"
    )


def build_validation_score_cache_lofo(
    candidate_files,
    features,
    labels,
    baseline,
    file_to_subject,
    random_seed=RANDOM_SEED,
):
    """Compatibility alias; validation is grouped by subject, not file."""
    return build_validation_score_cache_loso(
        candidate_files,
        features,
        labels,
        baseline,
        file_to_subject,
        random_seed=random_seed,
    )


def build_validation_score_cache_kfold(
    candidate_files,
    features,
    labels,
    baseline,
    n_splits=5,
    random_seed=RANDOM_SEED,
    file_to_subject=None,
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
        splits, features, labels, baseline, "Cache-GroupKFold"
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
    """Train once per patient-independent fold, then infer all fold files."""
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
    for fold_idx, (inner_train, val_files) in enumerate(splits, start=1):
        y_train = np.concatenate([labels[name] for name in inner_train]).astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        log_step(
            f"[Cache-LSTM-Grouped] fold {fold_idx}/{len(splits)} "
            f"train_files={len(inner_train)} val_files={len(val_files)}"
        )
        model, mean, std, device = _train_lstm_on_files(
            inner_train,
            features,
            labels,
            random_seed=random_seed,
            **(lstm_params or {}),
        )
        for val_file in val_files:
            scores = _predict_lstm_sequence(
                model, features[val_file], mean, std, device
            )
            cache[val_file] = {
                "scores": scores,
                "y_val": np.asarray(labels[val_file]).astype(int),
            }

        del model
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

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
):
    if file_to_subject is None:
        raise ValueError("file_to_subject is required for patient-independent validation")
    if baseline == "lstm":
        return build_validation_score_cache_lstm(
            candidate_files,
            features,
            labels,
            tune_mode,
            n_splits,
            lstm_params,
            file_to_subject,
            random_seed=random_seed,
        )

    normalized_mode = {"lofo": "loso", "nfold": "group_nfold"}.get(
        tune_mode, tune_mode
    )
    if normalized_mode == "loso":
        return build_validation_score_cache_loso(
            candidate_files,
            features,
            labels,
            baseline,
            file_to_subject,
            random_seed=random_seed,
        )
    if normalized_mode == "group_nfold":
        return build_validation_score_cache_kfold(
            candidate_files,
            features,
            labels,
            baseline,
            n_splits=n_splits,
            random_seed=random_seed,
            file_to_subject=file_to_subject,
        )
    raise ValueError(f"Unknown patient-independent tuning mode: {tune_mode}")
