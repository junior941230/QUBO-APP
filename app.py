import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import gradio as gr
from config import DEFAULT_LAMBDA_LIST, DEFAULT_THRESHOLD_LIST
from core.io import discover_subjects
from pipeline_runner.experiment import run_experiment
from ui.training_tab import build_training_tab
from ui.viewer_tab import build_viewer_tab


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _float_grid(values):
    return ", ".join(str(value) for value in values)


class CliProgress:
    def __call__(self, value, desc=None):
        if desc:
            print(f"[{value:>5.0%}] {desc}", flush=True)


def build_ui():
    with gr.Blocks(title="QUBO Seizure UI") as demo:
        gr.Markdown("# 🧠 QUBO Seizure Experiment Dashboard")
        with gr.Tabs():
            with gr.Tab("🧪 Training"):
                build_training_tab()
            with gr.Tab("📂 Viewer"):
                build_viewer_tab()
    return demo


def serve(args):
    build_ui().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


def train(args):
    subjects = list(args.subjects)
    if subjects == ["all"]:
        subjects = discover_subjects()

    summary, result_df, _summary_fig, _detail_fig, saved_path, run_id = run_experiment(
        selected_subjects=subjects,
        baseline=args.baseline,
        solver_name=args.solver,
        tune_mode=args.tune_mode,
        tune_n_splits=args.tune_n_splits,
        max_files_per_subject=args.max_files_per_subject,
        n_jobs=args.n_jobs,
        lambda_text=args.lambdas,
        threshold_text=args.thresholds,
        reuse_global_cache=not args.no_reuse_global_cache,
        save_pkl=not args.no_save_pkl,
        resume_enabled=not args.no_resume,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_epochs=args.lstm_epochs,
        lstm_lr=args.lstm_lr,
        lstm_batch=args.lstm_batch,
        lstm_dropout=args.lstm_dropout,
        force_restart=args.force_restart,
        progress=CliProgress(),
    )

    print("\n" + summary)
    if run_id:
        print(f"\nRun ID: {run_id}")
    if saved_path:
        print(f"Saved PKL: {saved_path}")
    if args.output_csv and not result_df.empty:
        result_df.to_csv(args.output_csv, index=False)
        print(f"Saved CSV: {args.output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="QUBO seizure experiment UI and training CLI."
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the Gradio UI.")
    serve_parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    serve_parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    serve_parser.add_argument("--share", action="store_true", default=_env_bool("GRADIO_SHARE"))
    serve_parser.set_defaults(func=serve)

    train_parser = subparsers.add_parser("train", help="Run training/evaluation from the CLI.")
    train_parser.add_argument(
        "--subjects",
        nargs="+",
        required=True,
        help="At least 3 subject IDs such as chb01 chb02 chb03, or 'all'.",
    )
    train_parser.add_argument("--baseline", choices=["svm", "xgboost", "lstm"], default="svm")
    train_parser.add_argument(
        "--solver",
        choices=["solve_qubo_seizure", "solve_chain_qubo_exact"],
        default="solve_chain_qubo_exact",
    )
    train_parser.add_argument(
        "--tune-mode",
        choices=["loso", "group_nfold"],
        default="group_nfold",
        help="Patient-grouped inner validation strategy.",
    )
    train_parser.add_argument("--tune-n-splits", type=int, default=5)
    train_parser.add_argument("--max-files-per-subject", type=int, default=0)
    train_parser.add_argument("--n-jobs", type=int, default=-1)
    train_parser.add_argument("--lambdas", default=_float_grid(DEFAULT_LAMBDA_LIST))
    train_parser.add_argument("--thresholds", default=_float_grid(DEFAULT_THRESHOLD_LIST))
    train_parser.add_argument("--no-reuse-global-cache", action="store_true")
    train_parser.add_argument("--no-save-pkl", action="store_true")
    train_parser.add_argument("--no-resume", action="store_true")
    train_parser.add_argument("--force-restart", action="store_true")
    train_parser.add_argument("--output-csv", default="")
    train_parser.add_argument("--lstm-hidden", type=int, default=32)
    train_parser.add_argument("--lstm-layers", type=int, default=1)
    train_parser.add_argument("--lstm-epochs", type=int, default=50)
    train_parser.add_argument("--lstm-lr", type=float, default=5e-4)
    train_parser.add_argument("--lstm-batch", type=int, default=4)
    train_parser.add_argument("--lstm-dropout", type=float, default=0.2)
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["serve"])
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    parsed_args.func(parsed_args)
