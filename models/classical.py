from config import RANDOM_SEED
from cuml import SVC
from cuml.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

def predict_svm(x_train, y_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    clf = SVC(
        probability=True, kernel="rbf",
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    clf.fit(x_train_scaled, y_train)
    return clf.predict_proba(x_test_scaled)[:, 1]

def predict_xgboost(x_train, y_train, x_test):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed")
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_SEED, n_jobs=4,
    )
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]