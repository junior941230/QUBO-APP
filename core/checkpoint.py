from config import CHECKPOINT_DIR
from core.logging_utils import log_step
import hashlib
import json
import pickle
from datetime import datetime


def _config_signature(config):
    """Return a stable representation of the settings that define a run."""
    # Subject selection is a set for this experiment: the input order does not
    # change the files evaluated.  Preserve that existing run-ID behavior while
    # serializing every other setting exactly as supplied.
    canonical_config = dict(config)
    if "subjects" in canonical_config:
        canonical_config["subjects"] = sorted(canonical_config["subjects"])
    return json.dumps(
        canonical_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def make_run_id(config):
    """Create a deterministic run id from experiment config."""
    raw = _config_signature(config).encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()[:10]
    subj_tag = "-".join(config["subjects"][:3])
    if len(config["subjects"]) > 3:
        subj_tag += f"+{len(config['subjects']) - 3}"
    return f"{subj_tag}_{config['baseline']}_{config['solver_name']}_{digest}"


def checkpoint_path(run_id):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"ckpt_{run_id}.pkl"


def load_checkpoint(run_id, expected_config=None):
    """Load a checkpoint only when it was written for the expected config."""
    path = checkpoint_path(run_id)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fp:
            data = pickle.load(fp)
        if expected_config is not None:
            saved_config = data.get("config")
            if (
                not isinstance(saved_config, dict)
                or _config_signature(saved_config) != _config_signature(expected_config)
            ):
                log_step(f"[Ckpt] config mismatch for {path}, starting fresh")
                return None
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
