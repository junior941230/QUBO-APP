"""Patient-independent experiment orchestration.

The engine accepts one typed request and has no dependency on Gradio or the CLI.
"""

from dataclasses import asdict
from datetime import datetime
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from config import BASELINE_THRESHOLD, RUN_SCHEMA_VERSION
from core.channels import validate_edf_channels
from core.checkpoint import (
    clear_checkpoint,
    load_checkpoint,
    make_run_id,
    save_checkpoint,
)
from core.io import collect_files_and_seizures
from core.logging_utils import log_step
from core.options import ExperimentRequest
from core.results import save_results_pkl
from core.splits import leave_one_subject_out_splits
from models.registry import predict_scores
from models.selection import tune_baseline_threshold_from_cache
from models.training_data import prepare_training_data
from pipeline import process_all_files
from qubo.solvers import get_qubo_solver, safe_solver_call
from qubo.tuning import tune_qubo_params_from_cache
from qubo.validation_cache import build_validation_score_cache
from viz.plots import build_detail_plot, build_summary_plot


def _is_retryable_skip(message):
    return ": subject-level cache build failed (" in message


def _empty_result(message, run_id=""):
    return message, pd.DataFrame(), None, None, "", run_id


def _report(progress, value, description):
    if progress is not None:
        progress(value, desc=description)


def _file_to_subject(file_paths):
    return {
        os.path.basename(path): os.path.basename(os.path.dirname(os.path.normpath(path)))
        for path in file_paths
    }


def _metric_row(
    subject,
    file_name,
    y_true,
    scores,
    y_fixed,
    y_baseline,
    y_qubo,
    baseline_threshold,
    qubo_lambda,
    qubo_threshold,
    qubo_val_score,
    train_subject_count,
):
    baseline_f1 = f1_score(y_true, y_baseline, zero_division=0)
    qubo_f1 = f1_score(y_true, y_qubo, zero_division=0)
    return {
        "subject": subject,
        "file": file_name,
        "has_seizure": bool(y_true.sum()),
        "baseline_fixed_f1": f1_score(y_true, y_fixed, zero_division=0),
        "baseline_f1": baseline_f1,
        "qubo_f1": qubo_f1,
        "improvement": qubo_f1 - baseline_f1,
        "baseline_threshold": float(baseline_threshold),
        "best_lambda": float(qubo_lambda),
        "best_threshold": float(qubo_threshold),
        "val_score": float(qubo_val_score),
        "baseline_average_precision": (
            average_precision_score(y_true, scores) if y_true.any() else 0.0
        ),
        "baseline_precision": precision_score(y_true, y_baseline, zero_division=0),
        "qubo_precision": precision_score(y_true, y_qubo, zero_division=0),
        "baseline_recall": recall_score(y_true, y_baseline, zero_division=0),
        "qubo_recall": recall_score(y_true, y_qubo, zero_division=0),
        "baseline_fp_rate": float(np.mean(y_baseline)),
        "qubo_fp_rate": float(np.mean(y_qubo)),
        "epochs": int(len(y_true)),
        "seizure_epochs": int(y_true.sum()),
        "train_subject_count": int(train_subject_count),
    }


def _save_progress(run_id, rows, detail_cache, skipped, config):
    save_checkpoint(run_id, rows, detail_cache, skipped, config)


def _attach_patient_metrics(result_df, detail_cache):
    """Attach held-out-patient aggregates without changing per-file rows."""
    patient_metrics = []
    for subject, subject_rows in result_df.groupby("subject"):
        details = [detail_cache[name] for name in subject_rows["file"]]
        y_true = np.concatenate([item["y_true"] for item in details])
        scores = np.concatenate([item["scores"] for item in details])
        seizure_rows = subject_rows[subject_rows["has_seizure"]]
        normal_rows = subject_rows[~subject_rows["has_seizure"]]
        metrics = {
            "subject": subject,
            "patient_average_precision": (
                average_precision_score(y_true, scores) if y_true.any() else 0.0
            ),
            "patient_seizure_macro_baseline_f1": (
                float(seizure_rows["baseline_f1"].mean())
                if len(seizure_rows) else 0.0
            ),
            "patient_seizure_macro_qubo_f1": (
                float(seizure_rows["qubo_f1"].mean())
                if len(seizure_rows) else 0.0
            ),
            "patient_nonseizure_baseline_fp_rate": (
                float(normal_rows["baseline_fp_rate"].mean())
                if len(normal_rows) else 0.0
            ),
            "patient_nonseizure_qubo_fp_rate": (
                float(normal_rows["qubo_fp_rate"].mean())
                if len(normal_rows) else 0.0
            ),
        }
        patient_metrics.append(metrics)
        mask = result_df["subject"] == subject
        for key, value in metrics.items():
            if key != "subject":
                result_df.loc[mask, key] = value
    return result_df, patient_metrics


def _build_subject_cache(request, train_files, features, labels, file_to_subject):
    subject_count = len({file_to_subject[name] for name in train_files})
    return build_validation_score_cache(
        train_files,
        features,
        labels,
        request.baseline,
        tune_mode=request.tune_mode,
        n_splits=min(request.tune_n_splits, subject_count),
        lstm_params=request.lstm.as_dict(),
        file_to_subject=file_to_subject,
        random_seed=request.random_seed,
        options=request.options,
    )


def _summary_text(request, run_id, result_df, notes, skipped):
    seizure = result_df[result_df["has_seizure"]]
    nonseizure = result_df[~result_df["has_seizure"]]
    lines = [
        f"Run ID: {run_id}",
        "Evaluation: patient-independent nested leave-one-subject-out",
        f"Finished {len(result_df)} files "
        f"(seizure={len(seizure)}, non-seizure={len(nonseizure)})",
        f"Held-out subjects evaluated: {result_df['subject'].nunique()} "
        f"({', '.join(sorted(result_df['subject'].unique()))})",
        f"Baseline={request.baseline}, Solver={request.solver_name}, "
        f"Patient-grouped tuning={request.tune_mode}",
        "",
        "[Seizure files]",
    ]
    if len(seizure):
        patient_ap = result_df.groupby("subject")["patient_average_precision"].first()
        lines.extend([
            f"  Mean fixed-threshold baseline F1 = {seizure['baseline_fixed_f1'].mean():.4f}",
            f"  Mean selected baseline F1        = {seizure['baseline_f1'].mean():.4f}",
            f"  Mean held-out-patient PR-AUC     = {patient_ap.mean():.4f}",
            f"  Mean QUBO F1                     = {seizure['qubo_f1'].mean():.4f}",
            f"  Mean Δ F1                        = {seizure['improvement'].mean():.4f}",
        ])
    else:
        lines.append("  (none)")
    lines.extend(["", "[Non-seizure files]"])
    if len(nonseizure):
        lines.extend([
            f"  Mean baseline FP rate = {nonseizure['baseline_fp_rate'].mean():.4f}",
            f"  Mean QUBO FP rate     = {nonseizure['qubo_fp_rate'].mean():.4f}",
        ])
    else:
        lines.append("  (none)")
    if notes:
        lines.extend(["", "Notes:", *(f"- {note}" for note in notes)])
    if skipped:
        lines.extend(["", "Skipped:", *(f"- {item}" for item in skipped[:10])])
        if len(skipped) > 10:
            lines.append(f"- ... and {len(skipped) - 10} more")
    return "\n".join(lines)


def run_experiment(request, progress=None):
    """Run a complete nested patient-independent evaluation."""
    if not isinstance(request, ExperimentRequest):
        raise TypeError("run_experiment expects an ExperimentRequest")
    if not request.selected_subjects:
        return _empty_result("Please select at least one subject")
    if len(request.selected_subjects) < 3:
        return _empty_result(
            "Patient-independent nested evaluation needs at least 3 subjects "
            "(one outer test subject and at least two inner train/validation subjects)"
        )

    config = request.semantic_config()
    run_id = make_run_id(config)
    log_step(f"[Run] run_id={run_id}")
    run_start = time.perf_counter()
    rows, detail_cache, skipped, done_files = [], {}, [], set()
    if request.force_restart:
        clear_checkpoint(run_id)
    if request.resume_enabled and not request.force_restart:
        checkpoint = load_checkpoint(run_id, expected_config=config)
        if checkpoint is not None:
            rows = checkpoint.get("rows", [])
            detail_cache = checkpoint.get("detail_cache", {})
            saved_skips = checkpoint.get("skipped", [])
            skipped = [item for item in saved_skips if not _is_retryable_skip(item)]
            done_files = {row["file"] for row in rows} | {
                item.split(":")[0].strip() for item in skipped if ":" in item
            }
            log_step(f"[Run] resumed, already done/skipped={len(done_files)}")

    _report(progress, 0.02, "Collecting files")
    file_paths, seizure_times, notes = collect_files_and_seizures(
        request.selected_subjects, request.max_files_per_subject
    )
    if len(file_paths) < 2:
        return _empty_result("Need at least 2 EDF files", run_id)

    _report(progress, 0.05, "Validating EDF channels")
    file_paths, channel_failures = validate_edf_channels(file_paths)
    for path, reason in channel_failures.items():
        notes.append(
            f"Excluded {os.path.basename(path)}: channel preflight failed ({reason})"
        )
    if len(file_paths) < 2:
        return _empty_result(
            "Need at least 2 channel-compatible EDF files\n\n"
            + "\n".join(f"- {note}" for note in notes),
            run_id,
        )

    _report(progress, 0.10, "Preprocessing EDF files")
    features, labels = process_all_files(
        file_paths,
        seizure_times,
        n_jobs=request.n_jobs,
        options=request.options,
    )
    test_files = [os.path.basename(path) for path in file_paths]
    if len(test_files) != len(set(test_files)):
        return _empty_result("EDF basenames must be unique", run_id)
    file_to_subject = _file_to_subject(file_paths)
    missing = [name for name in test_files if name not in features or name not in labels]
    if missing:
        test_files = [name for name in test_files if name not in missing]
        notes.append(f"Dropped {len(missing)} files with missing features")
    if len({file_to_subject[name] for name in test_files}) < 3:
        return _empty_result(
            "Need at least 3 subjects with usable EDF files after preprocessing", run_id
        )

    solver = get_qubo_solver(request.solver_name)
    outer_splits = leave_one_subject_out_splits(test_files, file_to_subject)
    validation_caches, cache_errors = {}, {}
    if request.reuse_validation_cache:
        targets = [
            subject for subject, split in outer_splits.items()
            if any(name not in done_files for name in split["test_files"])
        ]
        for index, subject in enumerate(targets, start=1):
            _report(
                progress,
                0.10 + 0.05 * index / max(1, len(targets)),
                f"Building grouped cache for held-out {subject}",
            )
            try:
                validation_caches[subject] = _build_subject_cache(
                    request,
                    outer_splits[subject]["train_files"],
                    features,
                    labels,
                    file_to_subject,
                )
            except Exception as exc:
                cache_errors[subject] = str(exc)
                log_step(f"[Run] validation cache failed for {subject}: {exc}")

    for subject_index, (test_subject, split) in enumerate(outer_splits.items(), start=1):
        train_files = split["train_files"]
        pending_files = [
            name for name in split["test_files"] if name not in done_files
        ]
        if not pending_files:
            continue
        _report(
            progress,
            0.15 + 0.8 * subject_index / max(1, len(outer_splits)),
            f"Evaluating held-out subject {test_subject}",
        )
        try:
            if request.reuse_validation_cache:
                if test_subject in cache_errors:
                    raise RuntimeError(cache_errors[test_subject])
                score_cache = validation_caches.get(test_subject, {})
            else:
                score_cache = _build_subject_cache(
                    request, train_files, features, labels, file_to_subject
                )
        except Exception as exc:
            skipped.extend(
                f"{name}: subject-level cache build failed ({exc})"
                for name in pending_files
            )
            _save_progress(run_id, rows, detail_cache, skipped, config)
            continue
        if not score_cache:
            skipped.extend(
                f"{name}: empty subject-level score cache" for name in pending_files
            )
            _save_progress(run_id, rows, detail_cache, skipped, config)
            continue

        try:
            qubo_lambda, qubo_threshold, qubo_val_score = (
                tune_qubo_params_from_cache(
                    score_cache,
                    solver,
                    request.lambda_values,
                    request.qubo_threshold_values,
                    alpha=request.tune_alpha,
                    random_seed=request.random_seed,
                )
            )
            baseline_selection = None
            baseline_threshold = BASELINE_THRESHOLD
            if request.options.tune_baseline_threshold:
                baseline_selection = tune_baseline_threshold_from_cache(
                    score_cache,
                    request.options.baseline_threshold_grid,
                    alpha=request.tune_alpha,
                )
                baseline_threshold = baseline_selection["threshold"]
        except Exception as exc:
            skipped.extend(
                f"{name}: subject-level tuning failed ({exc})"
                for name in pending_files
            )
            _save_progress(run_id, rows, detail_cache, skipped, config)
            continue

        if request.baseline == "lstm":
            train_features = np.empty((0, 0))
            train_labels = np.concatenate([labels[name] for name in train_files])
            train_sample_weight = None
        else:
            training = prepare_training_data(
                train_files,
                features,
                labels,
                request.options,
                request.random_seed,
                file_to_subject=file_to_subject,
            )
            train_features = training.features
            train_labels = training.labels
            train_sample_weight = training.sample_weight
        if len(np.unique(train_labels)) < 2:
            skipped.extend(
                f"{name}: single class in training labels" for name in pending_files
            )
            _save_progress(run_id, rows, detail_cache, skipped, config)
            continue
        train_subjects = sorted({file_to_subject[name] for name in train_files})
        if test_subject in train_subjects:
            raise AssertionError(f"Patient leakage detected for {test_subject}")

        for test_file in pending_files:
            file_start = time.perf_counter()
            y_true = np.asarray(labels[test_file]).astype(int)
            try:
                scores = np.asarray(predict_scores(
                    request.baseline,
                    train_features,
                    train_labels,
                    features[test_file],
                    options=request.options,
                    sample_weight=train_sample_weight,
                    train_files=train_files,
                    test_file=test_file,
                    features=features,
                    labels=labels,
                    lstm_params=request.lstm.as_dict(),
                    random_seed=request.random_seed,
                ))
                y_fixed = (scores >= BASELINE_THRESHOLD).astype(int)
                y_baseline = (scores >= baseline_threshold).astype(int)
                y_qubo = safe_solver_call(
                    solver,
                    scores,
                    qubo_lambda,
                    qubo_threshold,
                    seed=request.random_seed,
                )
            except Exception as exc:
                skipped.append(f"{test_file}: inference failed ({exc})")
                _save_progress(run_id, rows, detail_cache, skipped, config)
                continue

            row = _metric_row(
                test_subject,
                test_file,
                y_true,
                scores,
                y_fixed,
                y_baseline,
                y_qubo,
                baseline_threshold,
                qubo_lambda,
                qubo_threshold,
                qubo_val_score,
                len(train_subjects),
            )
            rows.append(row)
            detail_cache[test_file] = {
                "subject": test_subject,
                "file_name": test_file,
                "has_seizure": row["has_seizure"],
                "y_true": y_true,
                "y_baseline_fixed": y_fixed,
                "y_baseline": y_baseline,
                "y_qubo": y_qubo,
                "scores": scores,
                "baseline_threshold": baseline_threshold,
                "baseline_selection": baseline_selection,
                "best_lambda": qubo_lambda,
                "best_threshold": qubo_threshold,
            }
            _save_progress(run_id, rows, detail_cache, skipped, config)
            log_step(
                f"[File] done {test_file}, baseline_f1={row['baseline_f1']:.4f}, "
                f"qubo_f1={row['qubo_f1']:.4f}, elapsed={time.perf_counter() - file_start:.2f}s"
            )

    if not rows:
        return _empty_result(
            "Run failed\n\n" + ("\n".join(notes + skipped) or "No valid result"),
            run_id,
        )
    result_df = (
        pd.DataFrame(rows)
        .sort_values(["subject", "has_seizure", "improvement"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    result_df, patient_metrics = _attach_patient_metrics(result_df, detail_cache)
    seizure_df = result_df[result_df["has_seizure"]]
    top_file = (
        seizure_df.sort_values("improvement", ascending=False).iloc[0]["file"]
        if len(seizure_df)
        else result_df.iloc[0]["file"]
    )
    summary_figure = build_summary_plot(result_df)
    detail_figure = build_detail_plot(
        detail_cache[top_file], request.baseline, request.solver_name
    )
    _report(progress, 0.96, "Saving results")
    meta = {
        **config,
        "run_schema_version": RUN_SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "n_jobs": request.n_jobs,
        "reuse_validation_cache": request.reuse_validation_cache,
        "baseline_fixed_threshold": BASELINE_THRESHOLD,
        "options": asdict(request.options),
        "patient_metrics": patient_metrics,
        "notes": notes,
        "skipped": skipped,
        "total_elapsed_sec": time.perf_counter() - run_start,
    }
    saved_path = ""
    if request.save_pkl:
        try:
            saved_path = save_results_pkl(result_df, detail_cache, meta)
            clear_checkpoint(run_id)
        except Exception as exc:
            saved_path = f"(save failed: {exc})"
    _report(progress, 1.0, "Done")
    log_step(f"[Run] done, evaluated={len(result_df)}")
    return (
        _summary_text(request, run_id, result_df, notes, skipped),
        result_df,
        summary_figure,
        detail_figure,
        saved_path,
        run_id,
    )
