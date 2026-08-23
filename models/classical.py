import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.svm import SVC as SklearnSVC

from config import RANDOM_SEED
from core.logging_utils import log_step

try:
    from cuml import SVC as CuMLSVC
    from cuml.preprocessing import StandardScaler as CuMLStandardScaler
    CUM_AVAILABLE = True
except ImportError:
    CuMLSVC = None
    CuMLStandardScaler = None
    CUM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    from xgboost.core import XGBoostError
except ImportError:
    XGBClassifier = None
    XGBoostError = RuntimeError


def _cuda_available():
    """Return whether this process can actually use a CUDA device."""
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, RuntimeError):
        return False


def _svm_backend():
    if CUM_AVAILABLE and _cuda_available():
        return CuMLSVC, CuMLStandardScaler, "cuda (cuML)"
    return SklearnSVC, SklearnStandardScaler, "cpu (scikit-learn)"


def _predict_svm_with_backend(
    svc_class,
    scaler_class,
    x_train,
    y_train,
    x_test,
    sample_weight=None,
    random_seed=RANDOM_SEED,
):
    scaler = scaler_class()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    clf = svc_class(
        probability=True, kernel="rbf",
        class_weight="balanced", random_state=int(random_seed), cache_size=512,
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    clf.fit(x_train_scaled, y_train, **fit_kwargs)
    return np.asarray(clf.predict_proba(x_test_scaled))[:, 1]


def predict_svm(
    x_train, y_train, x_test, sample_weight=None, random_seed=RANDOM_SEED
):
    svc_class, scaler_class, device = _svm_backend()
    log_step(f"[SVM] device={device}")
    try:
        return _predict_svm_with_backend(
            svc_class, scaler_class, x_train, y_train, x_test,
            sample_weight,
            random_seed,
        )
    except Exception as exc:
        if device != "cuda (cuML)":
            raise
        log_step(
            f"[SVM] cuML failed ({type(exc).__name__}: {exc}); "
            "retrying on CPU"
        )
        return _predict_svm_with_backend(
            SklearnSVC, SklearnStandardScaler, x_train, y_train, x_test,
            sample_weight,
            random_seed,
        )


def predict_xgboost(
    x_train,
    y_train,
    x_test,
    options=None,
    sample_weight=None,
    random_seed=RANDOM_SEED,
):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed")

    params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss",
        random_state=int(random_seed), n_jobs=4, tree_method="hist",
    )
    if options is not None and options.xgb_class_weight_enabled:
        params["scale_pos_weight"] = float(options.xgb_scale_pos_weight)
    if options is not None and options.xgb_max_delta_step_enabled:
        params["max_delta_step"] = float(options.xgb_max_delta_step)

    device = "cuda" if _cuda_available() else "cpu"
    log_step(f"[XGBoost] device={device}")
    clf = XGBClassifier(device=device, **params)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    try:
        clf.fit(x_train, y_train, **fit_kwargs)
    except XGBoostError:
        if device != "cuda":
            raise
        log_step("[XGBoost] CUDA training failed; retrying on CPU")
        clf = XGBClassifier(device="cpu", **params)
        clf.fit(x_train, y_train, **fit_kwargs)
    return clf.predict_proba(x_test)[:, 1]
