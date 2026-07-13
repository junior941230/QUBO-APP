from config import RANDOM_SEED
from core.logging_utils import log_step
from sklearn.model_selection import KFold, StratifiedKFold
from models.registry import predict_scores
from models.lstm import _predict_lstm_sequence, _train_lstm_on_files
import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def build_validation_score_cache_lofo(candidate_files, features, labels, baseline):
    cache = {}
    log_step(f"[Cache-LOFO] start, files={len(candidate_files)}")
    for val_file in candidate_files:
        inner_train_files = [name for name in candidate_files if name != val_file]
        if not inner_train_files:
            continue
        x_train = np.concatenate([features[f] for f in inner_train_files])
        y_train = np.concatenate([labels[f] for f in inner_train_files]).astype(int)
        if len(np.unique(y_train)) < 2:
            continue
        x_val = features[val_file]
        y_val = np.asarray(labels[val_file]).astype(int)
        scores = np.asarray(predict_scores(baseline, x_train, y_train, x_val))
        cache[val_file] = {"scores": scores, "y_val": y_val}
    log_step(f"[Cache-LOFO] done, cached_files={len(cache)}")
    return cache


def build_validation_score_cache_kfold(
    candidate_files, features, labels, baseline, n_splits=5, random_seed=RANDOM_SEED,
):
    cache = {}
    arr = np.array(candidate_files)
    effective_splits = max(2, min(int(n_splits), len(candidate_files)))
    log_step(
        f"[Cache-NFold] start, files={len(candidate_files)}, "
        f"effective={effective_splits}"
    )
    file_has_seizure = np.array([1 if np.sum(labels[f]) > 0 else 0 for f in candidate_files])
    min_class_count = min(np.sum(file_has_seizure), np.sum(1 - file_has_seizure))

    if min_class_count >= effective_splits:
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(arr, file_has_seizure)
    else:
        splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(arr)

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        inner_train_files = arr[train_idx].tolist()
        val_files = arr[val_idx].tolist()
        x_train = np.concatenate([features[f] for f in inner_train_files])
        y_train = np.concatenate([labels[f] for f in inner_train_files]).astype(int)
        if len(np.unique(y_train)) < 2:
            continue

        x_val_all = np.concatenate([features[f] for f in val_files])
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

    log_step(f"[Cache-NFold] done, cached_files={len(cache)}")
    return cache

def build_validation_score_cache_lstm(candidate_files, features, labels,
                                      tune_mode, n_splits, lstm_params,
                                      random_seed=RANDOM_SEED):
    """LSTM version: train once per fold, infer on all val files."""
    cache = {}
    arr = np.array(candidate_files)

    if tune_mode == "lofo":
        splits = [([f for f in candidate_files if f != v], [v]) for v in candidate_files]
    else:
        effective = max(2, min(int(n_splits), len(candidate_files)))
        file_has_sz = np.array([1 if np.sum(labels[f]) > 0 else 0 for f in candidate_files])
        min_cls = min(np.sum(file_has_sz), np.sum(1 - file_has_sz))
        if min_cls >= effective:
            splitter = StratifiedKFold(n_splits=effective, shuffle=True, random_state=random_seed)
            split_iter = splitter.split(arr, file_has_sz)
        else:
            splitter = KFold(n_splits=effective, shuffle=True, random_state=random_seed)
            split_iter = splitter.split(arr)
        splits = [(arr[tr].tolist(), arr[va].tolist()) for tr, va in split_iter]

    log_step(f"[Cache-LSTM] folds={len(splits)}")
    for fi, (inner_train, val_files) in enumerate(splits, 1):
        y_tr_all = np.concatenate([labels[f] for f in inner_train]).astype(int)
        if len(np.unique(y_tr_all)) < 2:
            continue
        log_step(f"[Cache-LSTM] fold {fi}/{len(splits)} train={len(inner_train)} val={len(val_files)}")
        model, mean, std, device = _train_lstm_on_files(
            inner_train, features, labels,
            random_seed=random_seed, **(lstm_params or {}),
        )
        for vf in val_files:
            scores = _predict_lstm_sequence(model, features[vf], mean, std, device)
            cache[vf] = {"scores": scores, "y_val": np.asarray(labels[vf]).astype(int)}

        # free GPU
        del model
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_step(f"[Cache-LSTM] done, cached={len(cache)}")
    return cache

def build_validation_score_cache(
    candidate_files,
    features,
    labels,
    baseline,
    tune_mode,
    n_splits=5,
    lstm_params=None,
    random_seed=RANDOM_SEED,
):
    if baseline == "lstm":
        return build_validation_score_cache_lstm(
            candidate_files, features, labels, tune_mode, n_splits, lstm_params,
            random_seed=random_seed,
        )
    if tune_mode == "lofo":
        return build_validation_score_cache_lofo(candidate_files, features, labels, baseline)
    if tune_mode == "nfold":
        return build_validation_score_cache_kfold(
            candidate_files, features, labels, baseline, n_splits, random_seed=random_seed,
        )
    raise ValueError(f"Unknown tuning mode: {tune_mode}")
