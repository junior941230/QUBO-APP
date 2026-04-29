import matplotlib.pyplot as plt
import numpy as np

def build_summary_plot(df):
    seizure_df = df[df["has_seizure"]]
    nonseizure_df = df[~df["has_seizure"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    if len(seizure_df) > 0:
        means_f1 = [seizure_df["baseline_f1"].mean(), seizure_df["qubo_f1"].mean()]
        axes[0].bar(["Baseline", "QUBO"], means_f1, color=["#5B8FF9", "#5AD8A6"])
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f"Mean F1 on Seizure Files (n={len(seizure_df)})")
        axes[0].set_ylabel("F1")
        for i, v in enumerate(means_f1):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")
    else:
        axes[0].text(0.5, 0.5, "No seizure files", ha="center", va="center")
        axes[0].set_title("Mean F1 on Seizure Files")
        axes[0].axis("off")

    if len(nonseizure_df) > 0:
        means_fp = [
            nonseizure_df["baseline_fp_rate"].mean(),
            nonseizure_df["qubo_fp_rate"].mean(),
        ]
        axes[1].bar(["Baseline", "QUBO"], means_fp, color=["#5B8FF9", "#5AD8A6"])
        axes[1].set_title(f"Mean FP Rate on Non-seizure Files (n={len(nonseizure_df)})")
        axes[1].set_ylabel("False Positive Rate")
        top = max(means_fp) if max(means_fp) > 0 else 0.05
        axes[1].set_ylim(0, top * 1.3)
        for i, v in enumerate(means_fp):
            axes[1].text(i, v + top * 0.03, f"{v:.4f}", ha="center")
    else:
        axes[1].text(0.5, 0.5, "No non-seizure files", ha="center", va="center")
        axes[1].set_title("Mean FP Rate on Non-seizure Files")
        axes[1].axis("off")

    target_df = seizure_df if len(seizure_df) > 0 else df
    bins = max(1, min(15, len(target_df)))
    axes[2].hist(target_df["improvement"], bins=bins, color="#F6BD16", edgecolor="black")
    axes[2].axvline(target_df["improvement"].mean(), color="red", linestyle="--", label="Mean")
    axes[2].set_title("QUBO Improvement (Seizure Files)")
    axes[2].set_xlabel("QUBO F1 - Baseline F1")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    fig.tight_layout()
    return fig


def build_detail_plot(detail, baseline, solver_name):
    y_true = detail["y_true"]
    y_baseline = detail["y_baseline"]
    y_qubo = detail["y_qubo"]
    scores = detail["scores"]
    timeline = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(timeline, y_true, where="post", label="Ground Truth", linewidth=2)
    ax.plot(timeline, scores, label="Baseline Probability", alpha=0.55)
    ax.step(timeline, y_baseline, where="post", label=f"{baseline.upper()} Binary")
    ax.step(timeline, y_qubo, where="post", label=solver_name)

    ax.set_title(f"File Detail: {detail['file_name']}")
    ax.set_xlabel("Epoch (1 sec)")
    ax.set_ylabel("Label / Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    return fig