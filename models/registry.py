from config import RANDOM_SEED


def sequence_model(baseline):
    """(train, predict) for a baseline that treats each EDF as one sequence.

    ``train(train_files, features, labels, random_seed=..., **params)`` returns
    ``(model, mean, std, device)``; ``predict(model, seq, mean, std, device)``
    returns per-epoch probabilities. Looked up at call time so tests can patch
    the module functions.
    """
    if baseline == "lstm":
        from . import lstm
        return lstm._train_lstm_on_files, lstm._predict_lstm_sequence
    if baseline == "transformer":
        from . import transformer
        return (
            transformer._train_transformer_on_files,
            transformer._predict_transformer_sequence,
        )
    raise ValueError(f"Not a sequence baseline: {baseline}")


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
    if baseline == "transformer":
        from .transformer import predict_transformer
        return predict_transformer(
            kwargs["train_files"], kwargs["test_file"],
            kwargs["features"], kwargs["labels"],
            transformer_params=kwargs.get("transformer_params") or {},
            random_seed=kwargs.get("random_seed", RANDOM_SEED),
        )
    raise ValueError(f"Unknown baseline: {baseline}")
