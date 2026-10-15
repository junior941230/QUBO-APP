"""Seizure-event counting shared by QUBO tuning and tools/event_metrics.py."""
import numpy as np

from config import EPOCH_DURATION_SECONDS

# Predicted events closer than this many epochs count as one alarm.
MERGE_GAP = 10
# Alarms within this many epochs before onset / after offset of a seizure are
# not false alarms (pre-/post-ictal tolerance).
SEIZURE_MARGIN = 30
# A false alarm lasting longer than this many epochs counts once per started
# block, so flagging long stretches cannot pass as a single alarm.
MAX_ALARM_EPOCHS = 60


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
    """Return (true events, detected events, latencies in seconds, false alarms).

    A seizure is detected when any predicted epoch falls inside it. Predicted
    epochs outside every seizure (widened by SEIZURE_MARGIN) are false-alarm
    epochs; they are grouped into alarms (gaps <= MERGE_GAP merged) and each
    alarm counts once per MAX_ALARM_EPOCHS it lasts.
    """
    truth = events(y_true)
    y_pred = np.asarray(y_pred).astype(bool)
    detected, latencies = 0, []
    for start, end in truth:
        hits = np.flatnonzero(y_pred[start:end])
        if hits.size:
            detected += 1
            latencies.append(float(hits[0]) * EPOCH_DURATION_SECONDS)
    return len(truth), detected, latencies, false_alarm_count(y_true, y_pred)


def false_alarm_mask(y_true, y_pred):
    """Predicted epochs that are not within SEIZURE_MARGIN of any seizure."""
    allowed = np.zeros(len(y_pred), dtype=bool)
    for start, end in events(y_true):
        allowed[max(0, start - SEIZURE_MARGIN):end + SEIZURE_MARGIN] = True
    return np.asarray(y_pred).astype(bool) & ~allowed


def false_alarm_count(y_true, y_pred):
    return sum(
        -(-(end - start) // MAX_ALARM_EPOCHS)
        for start, end in events(false_alarm_mask(y_true, y_pred), MERGE_GAP)
    )


def recording_hours(y_true):
    return len(y_true) * EPOCH_DURATION_SECONDS / 3600.0
