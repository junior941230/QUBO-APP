import gradio as gr

from config import (
    DEFAULT_BASELINE_THRESHOLD_LIST,
    DEFAULT_LAMBDA_LIST,
    DEFAULT_THRESHOLD_LIST,
)
from core.io import discover_subjects
from core.options import build_experiment_request
from pipeline_runner.experiment import run_experiment
from qubo.solvers import POSTPROCESSORS


def _grid(values):
    return ", ".join(str(value) for value in values)


def run_from_ui(
    selected_subjects,
    baseline,
    solver_name,
    tune_mode,
    tune_n_splits,
    max_files_per_subject,
    n_jobs,
    lambda_values,
    qubo_threshold_values,
    reuse_validation_cache,
    save_pkl,
    resume_enabled,
    force_restart,
    lstm_hidden,
    lstm_layers,
    lstm_epochs,
    lstm_lr,
    lstm_batch,
    lstm_dropout,
    tune_baseline_threshold,
    baseline_threshold_grid,
    xgb_class_weight_enabled,
    xgb_scale_pos_weight,
    xgb_max_delta_step_enabled,
    xgb_max_delta_step,
    patient_balanced_weights,
    negative_downsample_enabled,
    negative_keep_fraction,
    log_power,
    relative_power,
    robust_normalize,
    temporal_context_seconds,
    transformer_d_model=64,
    transformer_n_heads=4,
    transformer_num_layers=2,
    transformer_dropout=0.1,
    transformer_window_seconds=120,
    transformer_causal=True,
    transformer_segment_seconds=600,
    transformer_epochs=20,
    transformer_lr=5e-4,
    transformer_batch_size=32,
    progress=gr.Progress(),
):
    try:
        request = build_experiment_request(
            selected_subjects=tuple(selected_subjects or []),
            baseline=baseline,
            solver_name=solver_name,
            tune_mode=tune_mode,
            tune_n_splits=int(tune_n_splits),
            max_files_per_subject=int(max_files_per_subject),
            n_jobs=int(n_jobs),
            lambda_values=lambda_values,
            qubo_threshold_values=qubo_threshold_values,
            reuse_validation_cache=reuse_validation_cache,
            save_pkl=save_pkl,
            resume_enabled=resume_enabled,
            force_restart=force_restart,
            lstm_hidden=int(lstm_hidden),
            lstm_layers=int(lstm_layers),
            lstm_epochs=int(lstm_epochs),
            lstm_lr=float(lstm_lr),
            lstm_batch=int(lstm_batch),
            lstm_dropout=float(lstm_dropout),
            tune_baseline_threshold=tune_baseline_threshold,
            baseline_threshold_grid=baseline_threshold_grid,
            xgb_class_weight_enabled=xgb_class_weight_enabled,
            xgb_scale_pos_weight=float(xgb_scale_pos_weight),
            xgb_max_delta_step_enabled=xgb_max_delta_step_enabled,
            xgb_max_delta_step=float(xgb_max_delta_step),
            patient_balanced_weights=patient_balanced_weights,
            negative_downsample_enabled=negative_downsample_enabled,
            negative_keep_fraction=float(negative_keep_fraction),
            log_power=log_power,
            relative_power=relative_power,
            robust_normalize=robust_normalize,
            temporal_context_seconds=int(temporal_context_seconds),
            transformer_d_model=int(transformer_d_model),
            transformer_n_heads=int(transformer_n_heads),
            transformer_num_layers=int(transformer_num_layers),
            transformer_dropout=float(transformer_dropout),
            transformer_window_seconds=int(transformer_window_seconds),
            transformer_causal=bool(transformer_causal),
            transformer_segment_seconds=int(transformer_segment_seconds),
            transformer_epochs=int(transformer_epochs),
            transformer_lr=float(transformer_lr),
            transformer_batch_size=int(transformer_batch_size),
        )
    except (TypeError, ValueError) as exc:
        import pandas as pd
        return f"Invalid settings: {exc}", pd.DataFrame(), None, None, "", ""
    return run_experiment(request, progress=progress)


def build_training_tab():
    subjects = discover_subjects()
    with gr.Column():
        gr.Markdown("## 🧪 Training / Evaluation")
        gr.Markdown(
            "Patient-independent nested leave-one-subject-out evaluation. "
            "所有 baseline 強化與特徵選項預設關閉；一次只開一項即可做可比對的實驗。"
        )
        if not subjects:
            gr.Markdown("⚠️ **No subjects found under `DESTINATION/`.**")
        selected_subjects = gr.CheckboxGroup(
            choices=subjects,
            value=subjects[:3] if subjects else [],
            label="Subjects (at least 3)",
        )
        with gr.Row():
            baseline = gr.Radio(
                ["svm", "xgboost", "lstm", "transformer"], value="svm",
                label="Baseline",
            )
            solver_name = gr.Radio(
                list(POSTPROCESSORS),
                value="solve_chain_qubo_exact",
                label="QUBO Solver",
            )
        with gr.Row():
            tune_mode = gr.Radio(
                ["loso", "group_nfold"],
                value="group_nfold",
                label="Inner patient-grouped validation",
            )
            tune_n_splits = gr.Slider(2, 10, value=5, step=1, label="Inner folds")
        with gr.Row():
            max_files_per_subject = gr.Slider(
                0, 30, value=0, step=1, label="Max EDF per subject (0=all)"
            )
            n_jobs = gr.Slider(-1, 16, value=-1, step=1, label="Preprocess jobs")
        with gr.Row():
            lambda_values = gr.Textbox(
                value=_grid(DEFAULT_LAMBDA_LIST), label="QUBO lambda grid"
            )
            qubo_threshold_values = gr.Textbox(
                value=_grid(DEFAULT_THRESHOLD_LIST), label="QUBO threshold grid"
            )

        with gr.Accordion("Baseline improvements (optional)", open=True):
            with gr.Row():
                tune_baseline_threshold = gr.Checkbox(
                    value=False, label="Tune baseline threshold on inner validation"
                )
                baseline_threshold_grid = gr.Textbox(
                    value=_grid(DEFAULT_BASELINE_THRESHOLD_LIST),
                    label="Baseline threshold grid",
                )
            with gr.Row():
                xgb_class_weight_enabled = gr.Checkbox(
                    value=False, label="XGBoost scale_pos_weight"
                )
                xgb_scale_pos_weight = gr.Number(value=1.0, label="scale_pos_weight")
                xgb_max_delta_step_enabled = gr.Checkbox(
                    value=False, label="XGBoost max_delta_step"
                )
                xgb_max_delta_step = gr.Number(value=0.0, label="max_delta_step")
            with gr.Row():
                patient_balanced_weights = gr.Checkbox(
                    value=False, label="Equal total weight per training patient"
                )
                negative_downsample_enabled = gr.Checkbox(
                    value=False, label="Downsample negative epochs"
                )
                negative_keep_fraction = gr.Slider(
                    0.01, 1.0, value=1.0, step=0.01, label="Negative keep fraction"
                )

        with gr.Accordion("Feature improvements (optional)", open=False):
            gr.Markdown(
                "Robust normalization uses the whole individual recording and is "
                "appropriate for offline analysis. Temporal context is causal and "
                "does not cross EDF boundaries."
            )
            with gr.Row():
                log_power = gr.Checkbox(value=False, label="Log band power")
                relative_power = gr.Checkbox(value=False, label="Relative band power")
                robust_normalize = gr.Checkbox(
                    value=False, label="Per-record robust normalization"
                )
                temporal_context_seconds = gr.Slider(
                    0, 120, value=0, step=1, label="Causal context seconds (0=off)"
                )

        with gr.Accordion("LSTM hyperparameters", open=False):
            with gr.Row():
                lstm_hidden = gr.Slider(32, 512, value=32, step=32, label="Hidden dim")
                lstm_layers = gr.Slider(1, 4, value=1, step=1, label="Layers")
                lstm_dropout = gr.Slider(0, 0.6, value=0.2, step=0.05, label="Dropout")
            with gr.Row():
                lstm_epochs = gr.Slider(5, 100, value=50, step=1, label="Epochs")
                lstm_lr = gr.Number(value=5e-4, label="Learning rate")
                lstm_batch = gr.Slider(1, 16, value=4, step=1, label="Batch (files)")

        with gr.Accordion("Transformer hyperparameters", open=False):
            gr.Markdown(
                "Sliding-window attention over per-second features. Causal mode "
                "attends only to the past and suits online detection."
            )
            with gr.Row():
                transformer_d_model = gr.Slider(
                    16, 256, value=64, step=16, label="Model dim"
                )
                transformer_n_heads = gr.Slider(1, 8, value=4, step=1, label="Heads")
                transformer_num_layers = gr.Slider(1, 6, value=2, step=1, label="Layers")
                transformer_dropout = gr.Slider(
                    0, 0.6, value=0.1, step=0.05, label="Dropout"
                )
            with gr.Row():
                transformer_window_seconds = gr.Slider(
                    10, 600, value=120, step=10, label="Attention window (s)"
                )
                transformer_causal = gr.Checkbox(value=True, label="Causal (past only)")
                transformer_segment_seconds = gr.Slider(
                    120, 3600, value=600, step=60, label="Training segment (s)"
                )
            with gr.Row():
                transformer_epochs = gr.Slider(1, 100, value=20, step=1, label="Epochs")
                transformer_lr = gr.Number(value=5e-4, label="Learning rate")
                transformer_batch_size = gr.Slider(
                    1, 128, value=32, step=1, label="Batch (segments)"
                )

        with gr.Row():
            reuse_validation_cache = gr.Checkbox(
                value=True, label="Precompute validation cache"
            )
            save_pkl = gr.Checkbox(value=True, label="Save results")
            resume_enabled = gr.Checkbox(value=True, label="Resume checkpoint")
            force_restart = gr.Checkbox(value=False, label="Force restart")

        run_button = gr.Button("▶ Run Experiment", variant="primary")
        summary_output = gr.Textbox(label="Run Summary", lines=14)
        run_id_output = gr.Textbox(label="Run ID")
        result_table = gr.Dataframe(label="Per-file Metrics")
        summary_plot = gr.Plot(label="Overall Visualization")
        detail_plot = gr.Plot(label="File Detail")
        saved_path_output = gr.Textbox(label="Saved .pkl Path")

        run_button.click(
            fn=run_from_ui,
            inputs=[
                selected_subjects, baseline, solver_name, tune_mode, tune_n_splits,
                max_files_per_subject, n_jobs, lambda_values, qubo_threshold_values,
                reuse_validation_cache, save_pkl, resume_enabled, force_restart,
                lstm_hidden, lstm_layers, lstm_epochs, lstm_lr, lstm_batch,
                lstm_dropout, tune_baseline_threshold, baseline_threshold_grid,
                xgb_class_weight_enabled, xgb_scale_pos_weight,
                xgb_max_delta_step_enabled, xgb_max_delta_step,
                patient_balanced_weights, negative_downsample_enabled,
                negative_keep_fraction, log_power, relative_power,
                robust_normalize, temporal_context_seconds,
                transformer_d_model, transformer_n_heads, transformer_num_layers,
                transformer_dropout, transformer_window_seconds, transformer_causal,
                transformer_segment_seconds, transformer_epochs, transformer_lr,
                transformer_batch_size,
            ],
            outputs=[
                summary_output, result_table, summary_plot, detail_plot,
                saved_path_output, run_id_output,
            ],
        )
