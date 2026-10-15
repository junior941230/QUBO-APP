"""Seizure-event metrics from result PKLs (per-epoch predictions in detail_cache).

Epoch-level F1 rewards matching every second of a seizure. Clinically the
questions are whether each seizure is detected, how late, and how many false
alarms fire per hour. For each prediction (tuned baseline and QUBO):

* sensitivity: share of true seizure events with at least one predicted epoch
  inside the event,
* latency: seconds from seizure onset to the first predicted epoch inside it
  (detected events only),
* false alarms per hour: predicted events that overlap no true seizure,
  divided by recording hours. Predicted events closer than MERGE_GAP epochs are
  merged first, so one flickering alarm counts once.

Metrics are computed per patient and then averaged over patients.

Usage: python tools/event_metrics.py results/qubo_run_*.pkl [--label NAME=PKL ...]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EPOCH_DURATION_SECONDS  # noqa: E402

MERGE_GAP = 10  # epochs


def events(mask, merge_gap=0):
    """Return [(start, end_exclusive)] runs of ones, merging gaps <= merge_gap."""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    runs = list(zip(edges[::2], edges[1::2]))
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(run) for run in merged]


def file_metrics(y_true, y_pred):
    truth = events(y_true)
    pred = events(y_pred, MERGE_GAP)
    y_pred = np.asarray(y_pred).astype(bool)
    detected, latencies = 0, []
    for start, end in truth:
        hits = np.flatnonzero(y_pred[start:end])
        if hits.size:
            detected += 1
            latencies.append(hits[0] * EPOCH_DURATION_SECONDS)
    false_alarms = sum(
        1 for p_start, p_end in pred
        if not any(p_start < t_end and t_start < p_end for t_start, t_end in truth)
    )
    return len(truth), detected, latencies, false_alarms


def run_metrics(pkl_path):
    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp)
    rows = []
    for detail in data["detail_cache"].values():
        y_true = detail["y_true"]
        hours = len(y_true) * EPOCH_DURATION_SECONDS / 3600.0
        for kind in ("y_baseline", "y_qubo"):
            n_true, n_det, lat, fa = file_metrics(y_true, detail[kind])
            rows.append({
                "subject": detail["subject"], "pred": kind[2:], "events": n_true,
                "detected": n_det, "latency_sum": sum(lat), "latency_n": len(lat),
                "false_alarms": fa, "hours": hours,
            })
    df = pd.DataFrame(rows).groupby(["pred", "subject"]).sum(numeric_only=True)
    per_patient = pd.DataFrame({
        "sensitivity": df.detected / df.events.where(df.events > 0),
        "latency_s": df.latency_sum / df.latency_n.where(df.latency_n > 0),
        "fa_per_h": df.false_alarms / df.hours,
    })
    summary = per_patient.groupby(level="pred").mean()
    summary["events_detected"] = df.groupby(level="pred").detected.sum().astype(str) + "/" + \
        df.groupby(level="pred").events.sum().astype(str)
    return data["meta"].get("run_id", Path(pkl_path).stem), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="PKL paths, optionally NAME=PATH")
    args = parser.parse_args()
    tables = []
    for item in args.runs:
        name, _, path = item.rpartition("=")
        run_id, summary = run_metrics(path)
        summary.insert(0, "run", name or run_id)
        tables.append(summary.reset_index())
    out = pd.concat(tables, ignore_index=True)
    print(out.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
