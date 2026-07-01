import gradio as gr
from core.io import discover_subjects
from pipeline_runner.experiment import run_experiment
from config import DEFAULT_LAMBDA_LIST, DEFAULT_THRESHOLD_LIST

def build_training_tab():
    subjects = discover_subjects()

    with gr.Column():
        gr.Markdown("## 🧪 Training / Evaluation")
        gr.Markdown(
            "Leave-one-file-out evaluation with inner-validation QUBO tuning. "
            "中斷可續跑:勾選 **Resume** 後,相同設定會從 checkpoint 繼續。"
        )

        if not subjects:
            gr.Markdown(
                "⚠️ **No subjects found under `DESTINATION/`.** "
                "Please check that the directory exists."
            )

        selected_subjects = gr.CheckboxGroup(
            choices=subjects,
            value=subjects[:1] if subjects else [],
            label="Subjects (multi-select)",
        )

        with gr.Row():
            baseline = gr.Radio(choices=["svm", "xgboost", "lstm"], value="svm", label="Baseline")
            solver_name = gr.Radio(
                choices=["solve_qubo_seizure", "solve_chain_qubo_exact"],
                value="solve_chain_qubo_exact",
                label="QUBO Solver",
            )

        with gr.Row():
            tune_mode = gr.Radio(
                choices=["lofo", "nfold"], value="nfold", label="Tuning Strategy"
            )
            tune_n_splits = gr.Slider(2, 10, value=5, step=1, label="Nfold Splits")

        with gr.Row():
            max_files_per_subject = gr.Slider(
                0, 30, value=0, step=1, label="Max EDF files per subject (0=all)"
            )
            n_jobs = gr.Slider(-1, 16, value=-1, step=1, label="Preprocess parallel jobs")

        with gr.Row():
            lambda_text = gr.Textbox(
                value=", ".join(str(x) for x in DEFAULT_LAMBDA_LIST),
                label="Lambda grid",
            )
            threshold_text = gr.Textbox(
                value=", ".join(str(x) for x in DEFAULT_THRESHOLD_LIST),
                label="Threshold grid",
            )

        with gr.Accordion("🧠 LSTM Hyperparameters (only used when baseline=lstm)",
                          open=True):
            with gr.Row():
                lstm_hidden = gr.Slider(32, 512, value=32, step=32, label="Hidden dim")
                lstm_layers = gr.Slider(1, 4, value=1, step=1, label="Num layers")
                lstm_dropout = gr.Slider(0.0, 0.6, value=0.2, step=0.05, label="Dropout")
            with gr.Row():
                lstm_epochs = gr.Slider(5, 100, value=50, step=1, label="Epochs")
                lstm_lr = gr.Number(value=5e-4, label="Learning rate")
                lstm_batch = gr.Slider(1, 16, value=4, step=1, label="Batch size (files)")

        with gr.Row():
            reuse_global_cache = gr.Checkbox(value=True, label="Reuse global validation cache")
            save_pkl = gr.Checkbox(value=True, label="Save results to ./results/")

        with gr.Row():
            resume_enabled = gr.Checkbox(
                value=True,
                label="Resume from checkpoint (if exists)",
            )
            force_restart = gr.Checkbox(
                value=False,
                label="Force restart (ignore / clear checkpoint)",
            )

        run_button = gr.Button("▶ Run Experiment", variant="primary")

        summary_output = gr.Textbox(label="Run Summary", lines=14)
        run_id_output = gr.Textbox(label="Run ID", lines=1)
        result_table = gr.Dataframe(label="Per-file Metrics")
        summary_plot = gr.Plot(label="Overall Visualization")
        detail_plot = gr.Plot(label="Top-Improvement Seizure File Detail")
        saved_path_output = gr.Textbox(label="Saved .pkl Path", lines=1)

        run_button.click(
            fn=run_experiment,
            inputs=[
                selected_subjects, baseline, solver_name,
                tune_mode, tune_n_splits,
                max_files_per_subject, n_jobs,
                lambda_text, threshold_text,
                reuse_global_cache, save_pkl,
                resume_enabled, 
                lstm_hidden, lstm_layers, lstm_epochs,
                lstm_lr, lstm_batch, lstm_dropout,force_restart,
            ],
            outputs=[
                summary_output, result_table,
                summary_plot, detail_plot,
                saved_path_output, run_id_output,
            ],
        )