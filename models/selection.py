import numpy as np
from sklearn.metrics import f1_score


def tune_baseline_threshold_from_cache(score_cache, thresholds, alpha=0.2):
    """Choose a baseline threshold using inner-validation recordings only."""
    if not score_cache:
        raise ValueError("Cannot tune a baseline threshold from an empty cache")
    best = None
    items = score_cache.values() if hasattr(score_cache, "values") else score_cache
    items = list(items)
    for threshold in thresholds:
        seizure_f1 = []
        nonseizure_fp = []
        for item in items:
            labels = np.asarray(item.get("labels", item.get("y_val"))).astype(int)
            predictions = np.asarray(item["scores"]) >= float(threshold)
            if labels.any():
                seizure_f1.append(f1_score(labels, predictions, zero_division=0))
            else:
                nonseizure_fp.append(float(predictions.mean()))
        macro_f1 = float(np.mean(seizure_f1)) if seizure_f1 else 0.0
        fp_rate = float(np.mean(nonseizure_fp)) if nonseizure_fp else 0.0
        objective = macro_f1 - float(alpha) * fp_rate
        candidate = (objective, -fp_rate, float(threshold), macro_f1)
        if best is None or candidate > best:
            best = candidate
    objective, neg_fp_rate, threshold, macro_f1 = best
    return {
        "threshold": threshold,
        "objective": objective,
        "seizure_macro_f1": macro_f1,
        "nonseizure_fp_rate": -neg_fp_rate,
    }
