from config import RANDOM_SEED


def predict_scores(
    baseline,
    x_train,
    y_train,
    x_test,
    options=None,
    sample_weight=None,
    **kwargs,
):
    if baseline == "svm":
        from .classical import predict_svm
        return predict_svm(
            x_train,
            y_train,
            x_test,
            sample_weight=sample_weight,
            random_seed=kwargs.get("random_seed", RANDOM_SEED),
        )
    if baseline == "xgboost":
        from .classical import predict_xgboost
        return predict_xgboost(
            x_train,
            y_train,
            x_test,
            options=options,
            sample_weight=sample_weight,
            random_seed=kwargs.get("random_seed", RANDOM_SEED),
        )
    if baseline == "lstm":
        from .lstm import predict_lstm
        return predict_lstm(
            kwargs["train_files"], kwargs["test_file"],
            kwargs["features"], kwargs["labels"],
            lstm_params=kwargs.get("lstm_params") or {},
            random_seed=kwargs.get("random_seed", RANDOM_SEED),
        )
    raise ValueError(f"Unknown baseline: {baseline}")
