from . import classical, lstm

def predict_scores(baseline, x_train, y_train, x_test, **kwargs):
    if baseline == "svm":
        return classical.predict_svm(x_train, y_train, x_test)
    if baseline == "xgboost":
        return classical.predict_xgboost(x_train, y_train, x_test)
    if baseline == "lstm":
        return lstm.predict_lstm(
            kwargs["train_files"], kwargs["test_file"],
            kwargs["features"], kwargs["labels"],
            lstm_params=kwargs.get("lstm_params") or {},
        )
    raise ValueError(f"Unknown baseline: {baseline}")
