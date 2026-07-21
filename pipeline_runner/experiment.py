from config import *
from core.logging_utils import log_step, parse_float_list
from core.io import collect_files_and_seizures
from core.channels import validate_edf_channels
from core.splits import leave_one_subject_out_splits
from core.checkpoint import make_run_id, load_checkpoint, save_checkpoint, clear_checkpoint
from core.results import save_results_pkl
from models.registry import predict_scores
from qubo.solvers import get_qubo_solver, safe_solver_call
from qubo.validation_cache import build_validation_score_cache
from qubo.tuning import tune_qubo_params_from_cache
from viz.plots import build_summary_plot, build_detail_plot
from pipeline import processAllFiles  # 你的 EDF 前處理
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
import gradio as gr

def run_experiment(
    selected_subjects,
    baseline,
    solver_name,
    tune_mode,
    tune_n_splits,
    max_files_per_subject,
    n_jobs,
    lambda_text,
    threshold_text,
    reuse_global_cache,
    save_pkl,
    resume_enabled,
    lstm_hidden,
    lstm_layers,
    lstm_epochs,
    lstm_lr,
    lstm_batch,
    lstm_dropout,
    force_restart,
    progress=gr.Progress(),
):
    if not selected_subjects:
        return "Please select at least one subject", pd.DataFrame(), None, None, "", ""

    selected_subjects = list(dict.fromkeys(selected_subjects))
    if len(selected_subjects) < 3:
        return (
            "Patient-independent nested evaluation needs at least 3 subjects "
            "(one outer test subject and at least two inner train/validation subjects)",
            pd.DataFrame(), None, None, "", "",
        )

    n_jobs = int(n_jobs)
    if n_jobs == 0:
        n_jobs = 1
    tune_n_splits = int(tune_n_splits)
    tune_mode = {"lofo": "loso", "nfold": "group_nfold"}.get(
        tune_mode, tune_mode
    )
    max_files_per_subject = int(max_files_per_subject)
    lambda_list = parse_float_list(lambda_text, DEFAULT_LAMBDA_LIST)
    threshold_list = parse_float_list(threshold_text, DEFAULT_THRESHOLD_LIST)
    lstm_params = {
        "hidden_dim": int(lstm_hidden),
        "num_layers": int(lstm_layers),
        "dropout": float(lstm_dropout),
        "epochs": int(lstm_epochs),
        "lr": float(lstm_lr),
        "batch_size": int(lstm_batch),
    }
    config = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "evaluation_protocol": "nested_leave_one_subject_out",
        "subjects": list(selected_subjects),
        "baseline": baseline,
        "solver_name": solver_name,
        "tune_mode": tune_mode,
        "tune_n_splits": tune_n_splits,
        "max_files_per_subject": max_files_per_subject,
        "lambda_list": lambda_list,
        "threshold_list": threshold_list,
        "reuse_global_cache": reuse_global_cache,
        "random_seed": RANDOM_SEED,
        "lstm_params": lstm_params if baseline == "lstm" else None,
    }
    run_id = make_run_id(config)
    log_step(f"[Run] run_id={run_id}")

    run_start = time.perf_counter()

    # --- Resume ---
    rows = []
    detail_cache = {}
    skipped = []
    done_files = set()

    if force_restart:
        clear_checkpoint(run_id)

    if resume_enabled and not force_restart:
        ckpt = load_checkpoint(run_id, expected_config=config)
        if ckpt is not None:
            rows = ckpt.get("rows", [])
            detail_cache = ckpt.get("detail_cache", {})
            skipped = ckpt.get("skipped", [])
            done_files = {r["file"] for r in rows} | {
                s.split(":")[0].strip() for s in skipped if ":" in s
            }
            log_step(f"[Run] resumed, already done/skipped={len(done_files)}")

    # --- Collect files ---
    progress(0.02, desc="Collecting files")
    file_paths, seizure_times, notes = collect_files_and_seizures(
        selected_subjects, max_files_per_subject
    )
    log_step(f"[Run] files={len(file_paths)}")

    if len(file_paths) < 2:
        return ("Need at least 2 EDF files", pd.DataFrame(), None, None, "", run_id)

    # --- Channel preflight ---
    # Read headers only, so incompatible recordings are rejected before the
    # expensive parallel preprocessing stage starts.
    progress(0.05, desc="Validating EDF channels")
    file_paths, channel_failures = validate_edf_channels(file_paths)
    for path, reason in channel_failures.items():
        note = f"Excluded {os.path.basename(path)}: channel preflight failed ({reason})"
        notes.append(note)
        log_step(f"[Channel-Preflight] {note}")
    log_step(
        f"[Channel-Preflight] compatible={len(file_paths)}, "
        f"excluded={len(channel_failures)}"
    )

    if len(file_paths) < 2:
        message = "Need at least 2 channel-compatible EDF files"
        if notes:
            message += "\n\n" + "\n".join(f"- {note}" for note in notes)
        return (message, pd.DataFrame(), None, None, "", run_id)

    # --- Preprocess ---
    progress(0.10, desc="Preprocessing EDF files")
    t0 = time.perf_counter()
    features, labels = processAllFiles(file_paths, seizure_times, nJobs=n_jobs)
    log_step(f"[Run] preprocess done, elapsed={time.perf_counter() - t0:.2f}s")

    test_files = [os.path.basename(path) for path in file_paths]
    if len(test_files) != len(set(test_files)):
        return (
            "EDF basenames must be unique so features can be assigned to subjects",
            pd.DataFrame(), None, None, "", run_id,
        )
    file_to_subject = {
        os.path.basename(path): os.path.basename(os.path.dirname(os.path.normpath(path)))
        for path in file_paths
    }
    missing = [f for f in test_files if f not in features or f not in labels]
    if missing:
        test_files = [f for f in test_files if f not in missing]
        notes.append(f"Dropped {len(missing)} files with missing features")

    effective_subjects = sorted({file_to_subject[name] for name in test_files})
    if len(effective_subjects) < 3:
        return (
            "Need at least 3 subjects with usable EDF files after preprocessing "
            "for nested patient-independent evaluation",
            pd.DataFrame(), None, None, "", run_id,
        )

    solver = get_qubo_solver(solver_name)

    outer_splits = leave_one_subject_out_splits(test_files, file_to_subject)

    # --- Leak-free outer validation caches ---
    # Each cache belongs to one held-out subject. Inner folds are grouped by
    # subject as well, so no patient's EDFs can cross any split boundary.
    outer_validation_caches = {}
    outer_cache_errors = {}
    if reuse_global_cache:
        cache_targets = [
            subject for subject, split in outer_splits.items()
            if any(name not in done_files for name in split["test_files"])
        ]
        for cache_idx, test_subject in enumerate(cache_targets, start=1):
            train_files = outer_splits[test_subject]["train_files"]
            train_subject_count = len({file_to_subject[name] for name in train_files})
            progress(
                0.10 + 0.05 * (cache_idx / max(1, len(cache_targets))),
                desc=f"Building grouped cache for held-out {test_subject}",
            )
            try:
                outer_validation_caches[test_subject] = (
                    build_validation_score_cache(
                        train_files, features, labels, baseline,
                        tune_mode=tune_mode,
                        n_splits=min(tune_n_splits, train_subject_count),
                        lstm_params=lstm_params,
                        file_to_subject=file_to_subject,
                        random_seed=RANDOM_SEED,
                    )
                )
            except Exception as exc:
                outer_cache_errors[test_subject] = str(exc)
                log_step(
                    f"[Run] validation cache failed for held-out subject "
                    f"{test_subject}: {exc}"
                )

    # --- Outer leave-one-subject-out loop ---
    loop_total = max(1, len(outer_splits))
    for subject_idx, (test_subject, split) in enumerate(outer_splits.items(), start=1):
        train_files = split["train_files"]
        subject_test_files = split["test_files"]
        pending_test_files = [
            name for name in subject_test_files if name not in done_files
        ]
        progress(
            0.15 + 0.8 * (subject_idx / loop_total),
            desc=f"Evaluating held-out subject {test_subject}",
        )
        if not pending_test_files:
            log_step(f"[Subject] skip (already done): {test_subject}")
            continue

        try:
            if reuse_global_cache:
                if test_subject in outer_cache_errors:
                    raise RuntimeError(outer_cache_errors[test_subject])
                score_cache = outer_validation_caches.get(test_subject, {})
            else:
                train_subject_count = len({file_to_subject[name] for name in train_files})
                score_cache = build_validation_score_cache(
                    train_files, features, labels, baseline,
                    tune_mode=tune_mode,
                    n_splits=min(tune_n_splits, train_subject_count),
                    lstm_params=lstm_params,
                    file_to_subject=file_to_subject,
                    random_seed=RANDOM_SEED,
                )
        except Exception as exc:
            skipped.extend(
                f"{name}: subject-level cache build failed ({exc})"
                for name in pending_test_files
            )
            save_checkpoint(run_id, rows, detail_cache, skipped, config)
            continue

        if not score_cache:
            skipped.extend(
                f"{name}: empty subject-level score cache"
                for name in pending_test_files
            )
            save_checkpoint(run_id, rows, detail_cache, skipped, config)
            continue

        try:
            best_lambda, best_threshold, best_val_score = tune_qubo_params_from_cache(
                score_cache, solver, lambda_list, threshold_list,
                alpha=TUNE_ALPHA, random_seed=RANDOM_SEED,
            )
        except Exception as exc:
            skipped.extend(
                f"{name}: subject-level QUBO tuning failed ({exc})"
                for name in pending_test_files
            )
            save_checkpoint(run_id, rows, detail_cache, skipped, config)
            continue

        x_train = np.concatenate([features[f] for f in train_files])
        y_train = np.concatenate([labels[f] for f in train_files]).astype(int)

        if len(np.unique(y_train)) < 2:
            skipped.extend(
                f"{name}: single class in training labels"
                for name in pending_test_files
            )
            save_checkpoint(run_id, rows, detail_cache, skipped, config)
            continue

        train_subjects = sorted({file_to_subject[name] for name in train_files})
        if test_subject in train_subjects:
            raise AssertionError(f"Patient leakage detected for {test_subject}")
        log_step(
            f"[Subject] {subject_idx}/{len(outer_splits)} test={test_subject} "
            f"train_subjects={len(train_subjects)} "
            f"test_files={len(subject_test_files)}"
        )

        for test_file in pending_test_files:
            file_start = time.perf_counter()
            x_test = features[test_file]
            y_test = np.asarray(labels[test_file]).astype(int)

            try:
                scores = np.asarray(predict_scores(
                    baseline, x_train, y_train, x_test,
                    train_files=train_files, test_file=test_file,
                    features=features, labels=labels,
                    lstm_params=lstm_params,
                    random_seed=RANDOM_SEED,
                ))
                y_baseline = (scores >= BASELINE_THRESHOLD).astype(int)
                y_qubo = safe_solver_call(
                    solver, scores, best_lambda, best_threshold, seed=RANDOM_SEED,
                )
            except Exception as exc:
                skipped.append(f"{test_file}: inference failed ({exc})")
                save_checkpoint(run_id, rows, detail_cache, skipped, config)
                continue

            has_seizure = bool(y_test.sum() > 0)
            baseline_f1 = f1_score(y_test, y_baseline, zero_division=0)
            qubo_f1 = f1_score(y_test, y_qubo, zero_division=0)

            rows.append({
                "subject": test_subject,
                "file": test_file,
                "has_seizure": has_seizure,
                "baseline_f1": baseline_f1,
                "qubo_f1": qubo_f1,
                "improvement": qubo_f1 - baseline_f1,
                "best_lambda": best_lambda,
                "best_threshold": best_threshold,
                "val_score": best_val_score,
                "baseline_precision": precision_score(y_test, y_baseline, zero_division=0),
                "qubo_precision": precision_score(y_test, y_qubo, zero_division=0),
                "baseline_recall": recall_score(y_test, y_baseline, zero_division=0),
                "qubo_recall": recall_score(y_test, y_qubo, zero_division=0),
                "baseline_fp_rate": float(np.mean(y_baseline)),
                "qubo_fp_rate": float(np.mean(y_qubo)),
                "epochs": int(len(y_test)),
                "seizure_epochs": int(y_test.sum()),
                "train_subject_count": len(train_subjects),
            })

            detail_cache[test_file] = {
                "subject": test_subject,
                "file_name": test_file,
                "has_seizure": has_seizure,
                "y_true": y_test,
                "y_baseline": y_baseline,
                "y_qubo": y_qubo,
                "scores": scores,
                "best_lambda": best_lambda,
                "best_threshold": best_threshold,
            }

            save_checkpoint(run_id, rows, detail_cache, skipped, config)
            log_step(
                f"[File] done {test_file}, held_out_subject={test_subject}, "
                f"baseline_f1={baseline_f1:.4f}, qubo_f1={qubo_f1:.4f}, "
                f"Δ={qubo_f1 - baseline_f1:.4f}, "
                f"elapsed={time.perf_counter() - file_start:.2f}s"
            )

    # --- Aggregate ---
    if not rows:
        note_text = "\n".join(notes + skipped) or "No valid result"
        return (f"Run failed\n\n{note_text}",
                pd.DataFrame(), None, None, "", run_id)

    result_df = (
        pd.DataFrame(rows)
        .sort_values(
            ["subject", "has_seizure", "improvement"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )

    seizure_df = result_df[result_df["has_seizure"]]
    nonseizure_df = result_df[~result_df["has_seizure"]]
    summary_fig = build_summary_plot(result_df)

    if len(seizure_df) > 0:
        top_file = seizure_df.sort_values("improvement", ascending=False).iloc[0]["file"]
    else:
        top_file = result_df.iloc[0]["file"]
    detail_fig = build_detail_plot(detail_cache[top_file], baseline, solver_name)

    progress(0.96, desc="Saving results")

    meta = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "evaluation_protocol": "nested_leave_one_subject_out",
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "subjects": list(selected_subjects),
        "baseline": baseline,
        "solver_name": solver_name,
        "tune_mode": tune_mode,
        "tune_n_splits": tune_n_splits,
        "max_files_per_subject": max_files_per_subject,
        "n_jobs": n_jobs,
        "lambda_list": lambda_list,
        "threshold_list": threshold_list,
        "reuse_global_cache": reuse_global_cache,
        "tune_alpha": TUNE_ALPHA,
        "baseline_threshold": BASELINE_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "notes": notes,
        "skipped": skipped,
        "total_elapsed_sec": time.perf_counter() - run_start,
        "lstm_params": lstm_params if baseline == "lstm" else None,
    }

    saved_path = ""
    if save_pkl:
        try:
            saved_path = save_results_pkl(result_df, detail_cache, meta)
            # 訓練完成後清理 checkpoint
            clear_checkpoint(run_id)
        except Exception as exc:
            saved_path = f"(save failed: {exc})"

    progress(1.0, desc="Done")

    summary_text = (
        f"Run ID: {run_id}\n"
        "Evaluation: patient-independent nested leave-one-subject-out\n"
        f"Finished {len(result_df)} files "
        f"(seizure={len(seizure_df)}, non-seizure={len(nonseizure_df)})\n"
        f"Held-out subjects evaluated: {result_df['subject'].nunique()} "
        f"({', '.join(sorted(result_df['subject'].unique()))})\n"
        f"Baseline={baseline}, Solver={solver_name}, "
        f"Patient-grouped tuning={tune_mode}, Nfold={tune_n_splits}\n"
    )

    summary_text += "\n[Seizure files]"
    if len(seizure_df) > 0:
        summary_text += (
            f"\n  Mean baseline F1 = {seizure_df['baseline_f1'].mean():.4f}"
            f"\n  Mean QUBO F1     = {seizure_df['qubo_f1'].mean():.4f}"
            f"\n  Mean Δ F1        = {seizure_df['improvement'].mean():.4f}"
        )
    else:
        summary_text += "\n  (none)"

    summary_text += "\n\n[Non-seizure files]"
    if len(nonseizure_df) > 0:
        summary_text += (
            f"\n  Mean baseline FP rate = {nonseizure_df['baseline_fp_rate'].mean():.4f}"
            f"\n  Mean QUBO FP rate     = {nonseizure_df['qubo_fp_rate'].mean():.4f}"
        )
    else:
        summary_text += "\n  (none)"

    if notes or skipped:
        summary_text += "\n\n"
        if notes:
            summary_text += "Notes:\n" + "\n".join(f"- {n}" for n in notes)
        if skipped:
            summary_text += "\nSkipped:\n" + "\n".join(f"- {s}" for s in skipped[:10])
            if len(skipped) > 10:
                summary_text += f"\n- ... and {len(skipped) - 10} more"

    log_step(
        f"[Run] done, evaluated={len(result_df)}, "
        f"total={time.perf_counter() - run_start:.2f}s"
    )

    return summary_text, result_df, summary_fig, detail_fig, saved_path, run_id
