"""Does a cross-channel QUBO beat 1-D smoothing? Offline nested LOSO study.

Per outer held-out patient, two XGBoost models are fit with the same inner
patient-grouped folds as the main pipeline:

* global: all 90 band-power features -> s[t] (the usual baseline score),
* channel: one channel-agnostic model on a single channel's 5 band powers plus
  the cross-channel median of each band -> p[c, t] for each of 18 channels.

Every post-processor is then tuned on the same inner-validation scores with
the main pipeline's objective (seizure-file F1 - 0.2 x non-seizure FP rate)
and applied to the held-out patient:

* chain_qubo / hmm_viterbi / moving_average on s (1-D references),
* fused_hmm / fused_ma: HMM Viterbi / moving average on a logistic-regression
  fusion of s and channel-score summaries (same extra information as the
  cross-channel QUBO, without the coupling),
* cross_qubo: qubo.cross_channel (global chain + channel chains + spatial and
  channel-to-global coupling), exact by min cut.

Each method is written as a result PKL readable by tools/event_metrics.py and
tools/bootstrap_compare.py.

Usage:
  python tools/cross_channel_study.py scores [--outer chb01 ...]  # GPU, sequential
  python tools/cross_channel_study.py evaluate [--n-jobs 24]    # CPU, parallel
"""
import argparse
import itertools
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED  # noqa: E402
from core.channels import CANONICAL_CHB_CHANNELS, validate_edf_channels  # noqa: E402
from core.io import collect_files_and_seizures, discover_subjects  # noqa: E402
from core.logging_utils import log_step  # noqa: E402
from core.options import ExperimentOptions  # noqa: E402
from core.splits import (  # noqa: E402
    leave_one_subject_out_splits,
    patient_independent_validation_splits,
)
from features import BANDS  # noqa: E402
from models.classical import predict_xgboost  # noqa: E402
from models.training_data import _selected_indices, prepare_training_data  # noqa: E402
from qubo.cross_channel import solve_cross_channel_qubo  # noqa: E402

WORK_DIR = Path("checkpoints/xchan")
OUT_DIR = Path("results/xchan-20261018")
N_CHANNELS, N_BANDS = len(CANONICAL_CHB_CHANNELS), len(BANDS)
OPTIONS = ExperimentOptions(
    tune_baseline_threshold=True, xgb_class_weight_enabled=True,
    xgb_scale_pos_weight=25.0, robust_normalize=True,
)
# The channel model keeps 5% of negatives; scale_pos_weight 1.25 gives the
# same effective positive weight as spw25 on all data.
CHANNEL_KEEP = 0.05
CHANNEL_OPTIONS = ExperimentOptions(xgb_class_weight_enabled=True, xgb_scale_pos_weight=1.25)
ALPHA = 0.2
THETAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
BASELINE_THRESHOLDS = OPTIONS.baseline_threshold_grid


# ---------------------------------------------------------------- scores stage

def channel_features(features):
    """(T, 90) band-major features -> (T * 18, 10) per-channel rows.

    Row t * 18 + c holds channel c's 5 band powers and the cross-channel
    median of each band at epoch t.
    """
    bands = np.asarray(features, dtype=np.float32).reshape(len(features), N_BANDS, N_CHANNELS)
    own = bands.transpose(0, 2, 1)                          # (T, C, B)
    median = np.median(bands, axis=2)[:, None, :]           # (T, 1, B)
    return np.concatenate([own, np.broadcast_to(median, own.shape)], axis=2).reshape(-1, 2 * N_BANDS)


def _fit_predict(train_files, test_files, features, labels, file_to_subject, seed):
    training = prepare_training_data(
        train_files, features, labels, OPTIONS, seed, file_to_subject=file_to_subject
    )
    test_x = np.concatenate([features[name] for name in test_files])
    global_scores = predict_xgboost(
        training.features, training.labels, test_x, options=OPTIONS, random_seed=seed
    )
    parts_x, parts_y = [], []
    for name in train_files:
        keep = _selected_indices(labels[name], CHANNEL_KEEP, seed, name)
        parts_x.append(channel_features(features[name][keep]))
        parts_y.append(np.repeat(np.asarray(labels[name])[keep], N_CHANNELS))
    channel_scores = predict_xgboost(
        np.concatenate(parts_x), np.concatenate(parts_y),
        channel_features(test_x), options=CHANNEL_OPTIONS, random_seed=seed,
    ).reshape(-1, N_CHANNELS).T                              # (C, T_total)
    out, offset = {}, 0
    for name in test_files:
        length = len(features[name])
        out[name] = {
            "scores": global_scores[offset:offset + length].astype(np.float32),
            "channel_scores": channel_scores[:, offset:offset + length].astype(np.float32),
            "y": np.asarray(labels[name]).astype(np.int8),
        }
        offset += length
    return out


def load_features(subjects, n_jobs):
    path = WORK_DIR / "features.pkl"
    if path.exists():
        with open(path, "rb") as fp:
            return pickle.load(fp)
    from pipeline import process_all_files
    file_paths, seizure_times, _ = collect_files_and_seizures(subjects, 0)
    file_paths, failures = validate_edf_channels(file_paths)
    log_step(f"[xchan] preprocessing {len(file_paths)} EDFs ({len(failures)} excluded)")
    features, labels = process_all_files(file_paths, seizure_times, n_jobs=n_jobs, options=OPTIONS)
    features = {name: value.astype(np.float32) for name, value in features.items()}
    file_to_subject = {
        os.path.basename(p): os.path.basename(os.path.dirname(p)) for p in file_paths
    }
    data = {"features": features, "labels": labels, "file_to_subject": file_to_subject}
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(data, fp, protocol=4)
    return data


def scores_stage(args):
    subjects = discover_subjects() if args.subjects == ["all"] else args.subjects
    data = load_features(subjects, args.n_jobs)
    features, labels, f2s = data["features"], data["labels"], data["file_to_subject"]
    outer = leave_one_subject_out_splits(sorted(features), f2s)
    for subject, split in outer.items():
        if args.outer and subject not in args.outer:
            continue
        path = WORK_DIR / f"scores_{subject}.pkl"
        if path.exists():
            continue
        start = time.perf_counter()
        train_files = split["train_files"]
        validation = {}
        folds = patient_independent_validation_splits(
            train_files, f2s, "group_nfold", n_splits=5, labels=labels, random_seed=args.seed
        )
        for train, val in folds:
            if len(np.unique(np.concatenate([labels[n] for n in train]))) < 2:
                continue
            validation.update(_fit_predict(train, val, features, labels, f2s, args.seed))
        test = _fit_predict(train_files, split["test_files"], features, labels, f2s, args.seed)
        with open(path, "wb") as fp:
            pickle.dump({"validation": validation, "test": test}, fp, protocol=4)
        log_step(f"[xchan] {subject} scores done in {time.perf_counter() - start:.0f}s")


# -------------------------------------------------------------- evaluate stage

def _logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def fusion_inputs(entry):
    channel = _logit(entry["channel_scores"])
    top3 = np.sort(channel, axis=0)[-3:].mean(axis=0)
    return np.column_stack([_logit(entry["scores"]), channel.max(axis=0), channel.mean(axis=0), top3])


def _objective(cache, predict):
    f1s, fps = [], []
    for entry in cache.values():
        y, pred = entry["y"], predict(entry)
        if y.sum() > 0:
            tp = int((pred & (y == 1)).sum())
            denom = int(pred.sum()) + int(y.sum())
            f1s.append(2 * tp / denom if denom else 0.0)
        else:
            fps.append(float(pred.mean()))
    return (np.mean(f1s) if f1s else 0.0) - ALPHA * (np.mean(fps) if fps else 0.0)


def _tune(cache, make_predict, grid):
    best = max(grid, key=lambda params: _objective(cache, make_predict(params)))
    return best


def _one_d(solver, key="scores"):
    def make(params):
        lam, theta = params
        return lambda entry: solver(entry[key], lam, theta).astype(bool)
    return make


def _cross(params):
    lam_t, theta, lam_s, mu, theta_c = params
    return lambda entry: solve_cross_channel_qubo(
        entry["scores"], entry["channel_scores"], lam_t, theta, lam_s, mu, theta_c
    ).astype(bool)


def evaluate_subject(subject):
    from sklearn.linear_model import LogisticRegression
    from qubo.solvers import hmm_viterbi, smooth_moving_average, solve_chain_qubo_exact

    with open(WORK_DIR / f"scores_{subject}.pkl", "rb") as fp:
        data = pickle.load(fp)
    val, test = data["validation"], data["test"]

    # Baseline threshold, as in the main pipeline (for the y_baseline column).
    base_thr = _tune(val, lambda t: (lambda e: e["scores"] >= t), BASELINE_THRESHOLDS)

    # Fusion of global and channel scores, fit on inner-validation scores.
    fusion = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 25})
    fusion.fit(np.concatenate([fusion_inputs(e) for e in val.values()]),
               np.concatenate([e["y"] for e in val.values()]))
    for cache in (val, test):
        for entry in cache.values():
            entry["fused"] = fusion.predict_proba(fusion_inputs(entry))[:, 1]

    qubo_grid = list(itertools.product((0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0), THETAS))
    hmm_grid = list(itertools.product((1, 2, 3, 5, 8, 12, 16), THETAS))
    ma_grid = list(itertools.product((1, 3, 5, 10, 15, 20, 30, 45, 60), THETAS))
    methods = {
        "chain_qubo": (_one_d(solve_chain_qubo_exact), qubo_grid),
        "hmm_viterbi": (_one_d(hmm_viterbi), hmm_grid),
        "moving_average": (_one_d(smooth_moving_average), ma_grid),
        "fused_hmm": (_one_d(hmm_viterbi, "fused"), hmm_grid),
        "fused_ma": (_one_d(smooth_moving_average, "fused"), ma_grid),
    }
    chosen, predictions = {}, {}
    for name, (make, grid) in methods.items():
        chosen[name] = _tune(val, make, grid)
        predictions[name] = make(chosen[name])

    # Cross-channel QUBO: start from the tuned 1-D chain (lam_t, theta), then
    # tune the coupling terms; mu = 0 recovers the 1-D chain exactly.
    lam_t, theta = chosen["chain_qubo"]
    coupling_grid = [(lam_t, theta, 0.0, 0.0, 0.5)] + list(itertools.product(
        (lam_t,), (theta,), (0.0, 0.1, 0.3), (0.03, 0.1, 0.3, 1.0), (0.4, 0.6, 0.8, 0.9)
    ))
    stage_b = _tune(val, _cross, coupling_grid)
    # Re-tune the global threshold with the chosen coupling.
    retune = [(lam_t, t, *stage_b[2:]) for t in THETAS]
    chosen["cross_qubo"] = _tune(val, _cross, retune)
    predictions["cross_qubo"] = _cross(chosen["cross_qubo"])

    out = {}
    for name, predict in predictions.items():
        out[name] = {
            file: {
                "subject": subject, "y_true": entry["y"].astype(int),
                "scores": entry["scores"],
                "y_baseline": (entry["scores"] >= base_thr).astype(int),
                "y_qubo": predict(entry).astype(int),
            }
            for file, entry in test.items()
        }
    return subject, chosen, out


def _patient_metrics(subject, details):
    from sklearn.metrics import average_precision_score, f1_score
    y = np.concatenate([d["y_true"] for d in details.values()])
    s = np.concatenate([d["scores"] for d in details.values()])
    seizure = [d for d in details.values() if d["y_true"].any()]
    normal = [d for d in details.values() if not d["y_true"].any()]

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "subject": subject,
        "patient_average_precision": average_precision_score(y, s) if y.any() else 0.0,
        "patient_seizure_macro_baseline_f1": mean(
            [f1_score(d["y_true"], d["y_baseline"], zero_division=0) for d in seizure]),
        "patient_seizure_macro_qubo_f1": mean(
            [f1_score(d["y_true"], d["y_qubo"], zero_division=0) for d in seizure]),
        "patient_nonseizure_baseline_fp_rate": mean([d["y_baseline"].mean() for d in normal]),
        "patient_nonseizure_qubo_fp_rate": mean([d["y_qubo"].mean() for d in normal]),
    }


def evaluate_stage(args):
    from joblib import Parallel, delayed
    subjects = sorted(p.stem[len("scores_"):] for p in WORK_DIR.glob("scores_*.pkl"))
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(evaluate_subject)(subject) for subject in subjects
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chosen_by_subject = {subject: chosen for subject, chosen, _ in results}
    for method in results[0][2]:
        detail_cache, patient_metrics = {}, []
        for subject, _, out in results:
            detail_cache.update(out[method])
            patient_metrics.append(_patient_metrics(subject, out[method]))
        meta = {
            "run_id": f"xchan_{method}", "solver_name": method,
            "patient_metrics": patient_metrics,
            "chosen_params": {s: c[method] for s, c in chosen_by_subject.items()},
        }
        with open(OUT_DIR / f"{method}.pkl", "wb") as fp:
            pickle.dump({"meta": meta, "detail_cache": detail_cache}, fp, protocol=4)
    with open(OUT_DIR / "chosen_params.txt", "w") as fp:
        for subject, chosen in sorted(chosen_by_subject.items()):
            fp.write(f"{subject} {chosen}\n")
    log_step(f"[xchan] wrote {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["scores", "evaluate"])
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--outer", nargs="+", help="Only score these held-out subjects.")
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (scores_stage if args.stage == "scores" else evaluate_stage)(args)


if __name__ == "__main__":
    main()
