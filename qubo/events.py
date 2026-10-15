"""Seizure-event counting shared by QUBO tuning and tools/event_metrics.py."""
import numpy as np

from config import EPOCH_DURATION_SECONDS

# Predicted events closer than this many epochs count as one alarm.
MERGE_GAP = 10


def events(mask, merge_gap=0):
    """Return [(start, end_exclusive)] runs of ones, merging gaps <= merge_gap."""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    merged = [[edges[0], edges[1]]]
    for start, end in zip(edges[2::2], edges[3::2]):
        if start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(int(start), int(end)) for start, end in merged]


def file_event_counts(y_true, y_pred):
    """Return (true events, detected events, latencies in seconds, false alarms)."""
    truth = events(y_true)
    y_pred = np.asarray(y_pred).astype(bool)
    detected, latencies = 0, []
    for start, end in truth:
        hits = np.flatnonzero(y_pred[start:end])
        if hits.size:
            detected += 1
            latencies.append(float(hits[0]) * EPOCH_DURATION_SECONDS)
    false_alarms = sum(
        1 for p_start, p_end in events(y_pred, MERGE_GAP)
        if not any(p_start < t_end and t_start < p_end for t_start, t_end in truth)
    )
    return len(truth), detected, latencies, false_alarms


def recording_hours(y_true):
    return len(y_true) * EPOCH_DURATION_SECONDS / 3600.0
