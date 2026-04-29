import gradio as gr
from viz.plots import build_summary_plot, build_detail_plot
from core.results import list_result_pkls, load_result_pkl, format_meta
import pandas as pd

def refresh_pkl_list():
    files = list_result_pkls()
    return gr.update(choices=files, value=files[0] if files else None)


def load_and_display_pkl(pkl_path):
    if not pkl_path:
        return "No file selected", pd.DataFrame(), None, None, gr.update(choices=[])

    try:
        payload = load_result_pkl(pkl_path)
    except Exception as exc:
        return f"Failed to load: {exc}", pd.DataFrame(), None, None, gr.update(choices=[])

    meta = payload.get("meta", {})
    result_df = payload.get("result_df", pd.DataFrame())
    detail_cache = payload.get("detail_cache", {})

    meta_md = format_meta(meta)
    summary_fig = build_summary_plot(result_df) if len(result_df) > 0 else None
    if summary_fig:
        summary_fig.suptitle(f"Baseline {meta.get('baseline', '')}, tune mode: {meta.get('tune_mode', '')}", fontsize=16)
        summary_fig.tight_layout(rect=(0, 0, 1, 0.95))

    # 預設挑 improvement 最高的 seizure file
    detail_fig = None
    default_file = None
    if len(result_df) > 0:
        seizure_df = result_df[result_df["has_seizure"]]
        if len(seizure_df) > 0:
            default_file = seizure_df.sort_values("improvement", ascending=False).iloc[0]["file"]
        else:
            default_file = result_df.iloc[0]["file"]
        if default_file in detail_cache:
            detail_fig = build_detail_plot(
                detail_cache[default_file],
                meta.get("baseline", "svm"),
                meta.get("solver_name", ""),
            )

    file_choices = list(detail_cache.keys())
    return (
        meta_md,
        result_df,
        summary_fig,
        detail_fig,
        gr.update(choices=file_choices, value=default_file),
    )


def show_file_detail(pkl_path, file_name):
    if not pkl_path or not file_name:
        return None
    try:
        payload = load_result_pkl(pkl_path)
    except Exception:
        return None
    detail_cache = payload.get("detail_cache", {})
    meta = payload.get("meta", {})
    if file_name not in detail_cache:
        return None
    return build_detail_plot(
        detail_cache[file_name],
        meta.get("baseline", "svm"),
        meta.get("solver_name", ""),
    )


def build_viewer_tab():
    with gr.Column():
        gr.Markdown("## 📂 Result Viewer")
        gr.Markdown("載入 `results/` 下的 `.pkl` 檔,檢視先前實驗結果。")

        with gr.Row():
            pkl_dropdown = gr.Dropdown(
                choices=list_result_pkls(),
                label="Select .pkl file",
                interactive=True,
            )
            refresh_btn = gr.Button("🔄 Refresh list")
            load_btn = gr.Button("📥 Load", variant="primary")

        meta_output = gr.Markdown(label="Metadata")
        viewer_table = gr.Dataframe(label="Per-file Metrics")
        viewer_summary_plot = gr.Plot(label="Overall Visualization")

        with gr.Row():
            file_selector = gr.Dropdown(
                choices=[], label="Select file to view detail", interactive=True
            )
        viewer_detail_plot = gr.Plot(label="File Detail")

        refresh_btn.click(fn=refresh_pkl_list, inputs=[], outputs=[pkl_dropdown])

        load_btn.click(
            fn=load_and_display_pkl,
            inputs=[pkl_dropdown],
            outputs=[
                meta_output, viewer_table,
                viewer_summary_plot, viewer_detail_plot,
                file_selector,
            ],
        )

        file_selector.change(
            fn=show_file_detail,
            inputs=[pkl_dropdown, file_selector],
            outputs=[viewer_detail_plot],
        )
