# app.py
import gradio as gr
from ui.training_tab import build_training_tab
from ui.viewer_tab import build_viewer_tab


def build_ui():
    with gr.Blocks(title="QUBO Seizure UI") as demo:
        gr.Markdown("# 🧠 QUBO Seizure Experiment Dashboard")
        with gr.Tabs():
            with gr.Tab("🧪 Training"):
                build_training_tab()
            with gr.Tab("📂 Viewer"):
                build_viewer_tab()
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
