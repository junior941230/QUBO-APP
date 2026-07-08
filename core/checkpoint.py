from config import CHECKPOINT_DIR
from core.logging_utils import log_step
import hashlib
import pickle
from datetime import datetime

def make_run_id(config):
    """Create a deterministic run id from experiment config."""
    key_parts = [
        str(config.get("run_schema_version", 1)),
        ",".join(sorted(config["subjects"])),
        config["baseline"],
        config["solver_name"],
        config["tune_mode"],
        str(config["tune_n_splits"]),
        str(config["max_files_per_subject"]),
        ",".join(str(x) for x in config["lambda_list"]),
        ",".join(str(x) for x in config["threshold_list"]),
        str(config["reuse_global_cache"]),
    ]
    raw = "|".join(key_parts).encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()[:10]
    subj_tag = "-".join(config["subjects"][:3])
    if len(config["subjects"]) > 3:
        subj_tag += f"+{len(config['subjects']) - 3}"
    return f"{subj_tag}_{config['baseline']}_{config['solver_name']}_{digest}"


def checkpoint_path(run_id):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"ckpt_{run_id}.pkl"


def load_checkpoint(run_id):
    path = checkpoint_path(run_id)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        log_step(f"[Ckpt] loaded {path}, done_files={len(data.get('rows', []))}")
        return data
    except Exception as exc:
        log_step(f"[Ckpt] load failed ({exc}), starting fresh")
        return None


def save_checkpoint(run_id, rows, detail_cache, skipped, config):
    path = checkpoint_path(run_id)
    payload = {
        "rows": rows,
        "detail_cache": detail_cache,
        "skipped": skipped,
        "config": config,
        "updated_at": datetime.now().isoformat(),
    }
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def clear_checkpoint(run_id):
    path = checkpoint_path(run_id)
    if path.exists():
        path.unlink()
        log_step(f"[Ckpt] cleared {path}")
