from dataclasses import asdict, dataclass, field

from config import (
    DEFAULT_BASELINE_THRESHOLD_LIST,
    DEFAULT_LAMBDA_LIST,
    DEFAULT_THRESHOLD_LIST,
    RANDOM_SEED,
    RUN_SCHEMA_VERSION,
)


def parse_float_grid(value):
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = value
    parsed = tuple(float(item) for item in values)
    if not parsed:
        raise ValueError("Grid must contain at least one number")
    return parsed


@dataclass(frozen=True)
class ExperimentOptions:
    tune_baseline_threshold: bool = False
    baseline_threshold_grid: tuple = field(
        default_factory=lambda: tuple(DEFAULT_BASELINE_THRESHOLD_LIST)
    )
    xgb_class_weight_enabled: bool = False
    xgb_scale_pos_weight: float = 1.0
    xgb_max_delta_step_enabled: bool = False
    xgb_max_delta_step: float = 0.0
    patient_balanced_weights: bool = False
    negative_downsample_enabled: bool = False
    negative_keep_fraction: float = 1.0
    log_power: bool = False
    relative_power: bool = False
    robust_normalize: bool = False
    temporal_context_seconds: int = 0

    def __post_init__(self):
        object.__setattr__(
            self, "baseline_threshold_grid",
            parse_float_grid(self.baseline_threshold_grid),
        )
        if not all(0.0 <= value <= 1.0 for value in self.baseline_threshold_grid):
            raise ValueError("Baseline threshold values must be between 0 and 1")
        if self.xgb_scale_pos_weight <= 0:
            raise ValueError("xgb_scale_pos_weight must be positive")
        if self.xgb_max_delta_step < 0:
            raise ValueError("xgb_max_delta_step cannot be negative")
        if not 0.0 < self.negative_keep_fraction <= 1.0:
            raise ValueError("negative_keep_fraction must be in (0, 1]")
        if self.temporal_context_seconds < 0:
            raise ValueError("temporal_context_seconds cannot be negative")

    def semantic_config(self):
        return {
            "tune_baseline_threshold": self.tune_baseline_threshold,
            "baseline_threshold_grid": (
                list(self.baseline_threshold_grid)
                if self.tune_baseline_threshold else None
            ),
            "xgb_class_weight_enabled": self.xgb_class_weight_enabled,
            "xgb_scale_pos_weight": (
                self.xgb_scale_pos_weight if self.xgb_class_weight_enabled else None
            ),
            "xgb_max_delta_step_enabled": self.xgb_max_delta_step_enabled,
            "xgb_max_delta_step": (
                self.xgb_max_delta_step if self.xgb_max_delta_step_enabled else None
            ),
            "patient_balanced_weights": self.patient_balanced_weights,
            "negative_downsample_enabled": self.negative_downsample_enabled,
            "negative_keep_fraction": (
                self.negative_keep_fraction
                if self.negative_downsample_enabled else None
            ),
            "log_power": self.log_power,
            "relative_power": self.relative_power,
            "robust_normalize": self.robust_normalize,
            "temporal_context_seconds": self.temporal_context_seconds,
        }


@dataclass(frozen=True)
class LSTMOptions:
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.2
    epochs: int = 50
    lr: float = 0.0005
    batch_size: int = 4

    def __post_init__(self):
        if min(self.hidden_dim, self.num_layers, self.epochs, self.batch_size) <= 0:
            raise ValueError("LSTM dimensions, epochs, and batch size must be positive")
        if self.lr <= 0:
            raise ValueError("LSTM learning rate must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LSTM dropout must be in [0, 1)")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExperimentRequest:
    selected_subjects: tuple
    baseline: str = "svm"
    solver_name: str = "solve_chain_qubo_exact"
    tune_mode: str = "group_nfold"
    tune_n_splits: int = 5
    max_files_per_subject: int = 0
    n_jobs: int = -1
    lambda_values: tuple = field(default_factory=lambda: tuple(DEFAULT_LAMBDA_LIST))
    qubo_threshold_values: tuple = field(
        default_factory=lambda: tuple(DEFAULT_THRESHOLD_LIST)
    )
    tune_alpha: float = 0.2
    random_seed: int = RANDOM_SEED
    reuse_validation_cache: bool = True
    save_pkl: bool = True
    resume_enabled: bool = True
    force_restart: bool = False
    options: ExperimentOptions = field(default_factory=ExperimentOptions)
    lstm: LSTMOptions = field(default_factory=LSTMOptions)

    def __post_init__(self):
        object.__setattr__(
            self, "selected_subjects", tuple(dict.fromkeys(self.selected_subjects))
        )
        object.__setattr__(self, "lambda_values", parse_float_grid(self.lambda_values))
        object.__setattr__(
            self, "qubo_threshold_values", parse_float_grid(self.qubo_threshold_values)
        )
        if self.baseline not in {"svm", "xgboost", "lstm"}:
            raise ValueError(f"Unknown baseline: {self.baseline}")
        if self.solver_name not in {
            "solve_qubo_seizure", "solve_chain_qubo_exact"
        }:
            raise ValueError(f"Unknown solver: {self.solver_name}")
        normalized_mode = {"lofo": "loso", "nfold": "group_nfold"}.get(
            self.tune_mode, self.tune_mode
        )
        object.__setattr__(self, "tune_mode", normalized_mode)
        if normalized_mode not in {"loso", "group_nfold"}:
            raise ValueError(f"Unknown tune mode: {self.tune_mode}")
        if self.tune_n_splits < 2:
            raise ValueError("tune_n_splits must be at least 2")
        if self.max_files_per_subject < 0:
            raise ValueError("max_files_per_subject cannot be negative")
        if any(value < 0 for value in self.lambda_values):
            raise ValueError("QUBO lambda values cannot be negative")
        if not all(0 <= value <= 1 for value in self.qubo_threshold_values):
            raise ValueError("QUBO threshold values must be between 0 and 1")
        if self.tune_alpha < 0:
            raise ValueError("tune_alpha cannot be negative")
        if self.n_jobs == 0:
            object.__setattr__(self, "n_jobs", 1)
        if self.baseline == "lstm" and (
            self.options.patient_balanced_weights
            or self.options.negative_downsample_enabled
        ):
            raise ValueError(
                "Patient weights and negative downsampling currently support "
                "only SVM/XGBoost"
            )
        if self.baseline != "xgboost" and (
            self.options.xgb_class_weight_enabled
            or self.options.xgb_max_delta_step_enabled
        ):
            raise ValueError("XGBoost options require baseline=xgboost")

    def semantic_config(self):
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "evaluation_protocol": "nested_leave_one_subject_out",
            "subjects": list(self.selected_subjects),
            "baseline": self.baseline,
            "solver_name": self.solver_name,
            "tune_mode": self.tune_mode,
            "tune_n_splits": self.tune_n_splits,
            "max_files_per_subject": self.max_files_per_subject,
            "lambda_values": list(self.lambda_values),
            "qubo_threshold_values": list(self.qubo_threshold_values),
            "tune_alpha": self.tune_alpha,
            "random_seed": self.random_seed,
            "options": self.options.semantic_config(),
            "lstm": asdict(self.lstm) if self.baseline == "lstm" else None,
        }


def build_experiment_request(**kwargs):
    option_fields = set(ExperimentOptions.__dataclass_fields__)
    lstm_fields = set(LSTMOptions.__dataclass_fields__)
    explicit_options = kwargs.pop("options", None)
    explicit_lstm = kwargs.pop("lstm", None)
    option_values = {key: kwargs.pop(key) for key in list(kwargs) if key in option_fields}
    lstm_aliases = {
        "lstm_hidden": "hidden_dim",
        "lstm_layers": "num_layers",
        "lstm_dropout": "dropout",
        "lstm_epochs": "epochs",
        "lstm_lr": "lr",
        "lstm_batch": "batch_size",
    }
    lstm_values = {}
    for key in list(kwargs):
        if key in lstm_aliases:
            lstm_values[lstm_aliases[key]] = kwargs.pop(key)
        elif key.startswith("lstm_") and key[5:] in lstm_fields:
            lstm_values[key[5:]] = kwargs.pop(key)
    if explicit_options is not None and option_values:
        raise ValueError("Pass either options or flat option fields, not both")
    if explicit_lstm is not None and lstm_values:
        raise ValueError("Pass either lstm or flat LSTM fields, not both")
    kwargs["options"] = explicit_options or ExperimentOptions(**option_values)
    kwargs["lstm"] = explicit_lstm or LSTMOptions(**lstm_values)
    return ExperimentRequest(**kwargs)
