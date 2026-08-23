# config.py
from pathlib import Path

DESTINATION_DIR = Path("DESTINATION")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

DEFAULT_LAMBDA_LIST = [0.5, 1.0, 1.5, 2.0, 3.0]
DEFAULT_THRESHOLD_LIST = [0.3, 0.4, 0.45, 0.5, 0.6]
DEFAULT_BASELINE_THRESHOLD_LIST = [
    0.001, 0.003, 0.005, 0.01, 0.02, 0.03,
    0.05, 0.1, 0.2, 0.3, 0.5,
]

TUNE_ALPHA = 0.2
BASELINE_THRESHOLD = 0.5
RANDOM_SEED = 42
EPOCH_DURATION_SECONDS = 1.0

# Increment when preprocessing or evaluation semantics change so checkpoints
# produced by older code cannot be resumed into an incompatible run.
RUN_SCHEMA_VERSION = 8
