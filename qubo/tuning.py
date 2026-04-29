from config import TUNE_ALPHA
from core.logging_utils import log_step
from sklearn.metrics import f1_score
from qubo.solvers import safe_solver_call
import numpy as np

def tune_qubo_params_from_cache(score_cache, solver, lambda_list, threshold_list, alpha=TUNE_ALPHA):
    if not lambda_list or not threshold_list:
        raise ValueError("lambda_list and threshold_list must not be empty")
    if not score_cache:
        raise ValueError("score_cache is empty, cannot tune")

    best_score = -1e9
    best_lambda = float(lambda_list[0])
    best_threshold = float(threshold_list[0])

    for lmbda in lambda_list:
        for threshold in threshold_list:
            seizure_f1s = []
            nonseizure_fps = []
            for data in score_cache.values():
                try:
                    y_qubo_val = safe_solver_call(solver, data["scores"], lmbda, threshold)
                except Exception as exc:
                    log_step(f"[QUBO-Tune] solver failed λ={lmbda},θ={threshold}: {exc}")
                    continue
                y_val = data["y_val"]
                if np.sum(y_val) > 0:
                    seizure_f1s.append(f1_score(y_val, y_qubo_val, zero_division=0))
                else:
                    nonseizure_fps.append(float(np.mean(y_qubo_val)))

            mean_f1 = float(np.mean(seizure_f1s)) if seizure_f1s else 0.0
            mean_fp = float(np.mean(nonseizure_fps)) if nonseizure_fps else 0.0
            combined = mean_f1 - alpha * mean_fp

            if combined > best_score:
                best_score = combined
                best_lambda = float(lmbda)
                best_threshold = float(threshold)

    log_step(f"[QUBO-Tune] best λ={best_lambda}, θ={best_threshold}, score={best_score:.4f}")
    return best_lambda, best_threshold, float(best_score)