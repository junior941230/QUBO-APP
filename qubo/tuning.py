from collections import defaultdict

from config import RANDOM_SEED, TUNE_ALPHA
from core.logging_utils import log_step
from sklearn.metrics import f1_score
from qubo.events import file_event_counts, recording_hours
from qubo.solvers import safe_solver_call
import numpy as np

QUBO_TUNE_OBJECTIVES = ("file_mean", "patient_macro", "event")


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _objective_terms(seizure_f1s, nonseizure_fps, objective):
    """Return (mean F1, mean FP) from {subject: [values]} under an objective.

    file_mean pools every validation file, so patients with many files dominate.
    patient_macro averages within each patient first, then across patients.
    """
    if objective == "file_mean":
        return (
            _mean([v for values in seizure_f1s.values() for v in values]),
            _mean([v for values in nonseizure_fps.values() for v in values]),
        )
    return (
        _mean([_mean(values) for values in seizure_f1s.values()]),
        _mean([_mean(values) for values in nonseizure_fps.values()]),
    )


def _event_score(counts, alpha):
    """Patient-macro seizure sensitivity minus alpha * false alarms per hour.

    counts maps subject -> [true events, detected events, false alarms, hours].
    Patients without seizure events in validation only contribute alarms.
    """
    sensitivity = _mean([d / n for n, d, _, _ in counts.values() if n > 0])
    alarms = _mean([fa / h for _, _, fa, h in counts.values() if h > 0])
    return sensitivity - alpha * alarms


def tune_qubo_params_from_cache(
    score_cache,
    solver,
    lambda_list,
    threshold_list,
    alpha=TUNE_ALPHA,
    random_seed=RANDOM_SEED,
    objective="file_mean",
    file_to_subject=None,
):
    if not lambda_list or not threshold_list:
        raise ValueError("lambda_list and threshold_list must not be empty")
    if not score_cache:
        raise ValueError("score_cache is empty, cannot tune")
    if objective not in QUBO_TUNE_OBJECTIVES:
        raise ValueError(f"Unknown QUBO tune objective: {objective}")
    if objective != "file_mean" and file_to_subject is None:
        raise ValueError(f"{objective} objective requires file_to_subject")
    subject_of = (
        (lambda name: None) if objective == "file_mean"
        else (lambda name: file_to_subject[name])
    )

    best_score = -1e9
    best_lambda = float(lambda_list[0])
    best_threshold = float(threshold_list[0])

    for lmbda in lambda_list:
        for threshold in threshold_list:
            seizure_f1s = defaultdict(list)
            nonseizure_fps = defaultdict(list)
            event_counts = defaultdict(lambda: [0, 0, 0, 0.0])
            for name, data in score_cache.items():
                try:
                    y_qubo_val = safe_solver_call(
                        solver, data["scores"], lmbda, threshold, seed=random_seed,
                    )
                except Exception as exc:
                    log_step(f"[QUBO-Tune] solver failed λ={lmbda},θ={threshold}: {exc}")
                    continue
                y_val = data["y_val"]
                if objective == "event":
                    n_true, n_det, _, false_alarms = file_event_counts(y_val, y_qubo_val)
                    counts = event_counts[subject_of(name)]
                    counts[0] += n_true
                    counts[1] += n_det
                    counts[2] += false_alarms
                    counts[3] += recording_hours(y_val)
                    continue
                if np.sum(y_val) > 0:
                    seizure_f1s[subject_of(name)].append(
                        f1_score(y_val, y_qubo_val, zero_division=0)
                    )
                else:
                    nonseizure_fps[subject_of(name)].append(float(np.mean(y_qubo_val)))

            if objective == "event":
                combined = _event_score(event_counts, alpha)
            else:
                mean_f1, mean_fp = _objective_terms(seizure_f1s, nonseizure_fps, objective)
                combined = mean_f1 - alpha * mean_fp

            if combined > best_score:
                best_score = combined
                best_lambda = float(lmbda)
                best_threshold = float(threshold)

    log_step(f"[QUBO-Tune] best λ={best_lambda}, θ={best_threshold}, score={best_score:.4f}")
    return best_lambda, best_threshold, float(best_score)
