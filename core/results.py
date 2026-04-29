from config import RESULTS_DIR
from core.logging_utils import log_step
import pickle
from datetime import datetime

def save_results_pkl(result_df, detail_cache, meta, output_dir=RESULTS_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qubo_run_{timestamp}.pkl"
    filepath = output_dir / filename

    payload = {
        "meta": meta,
        "result_df": result_df,
        "detail_cache": detail_cache,
        "saved_at": timestamp,
    }
    with open(filepath, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    log_step(f"[Save] results written to {filepath}")
    return str(filepath)


def list_result_pkls(output_dir=RESULTS_DIR):
    if not output_dir.exists():
        return []
    return sorted(
        (str(p) for p in output_dir.glob("qubo_run_*.pkl")),
        reverse=True,
    )


def load_result_pkl(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def format_meta(meta):
    lines = ["## Run Metadata"]
    for k, v in meta.items():
        if k in ("notes", "skipped"):
            continue
        lines.append(f"- **{k}**: {v}")
    if meta.get("notes"):
        lines.append("### Notes")
        lines.extend(f"- {n}" for n in meta["notes"])
    if meta.get("skipped"):
        lines.append("### Skipped")
        lines.extend(f"- {s}" for s in meta["skipped"][:20])
        if len(meta["skipped"]) > 20:
            lines.append(f"- ... and {len(meta['skipped']) - 20} more")
    return "\n".join(lines)