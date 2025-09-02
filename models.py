from typing import Dict, Tuple
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    r2_score, mean_squared_error
)

def _clf_candidates():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=None),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBClassifier": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, n_jobs=4, eval_metric="logloss"
        )
    }

def _reg_candidates():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBRegressor": XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, n_jobs=4, objective="reg:squarederror"
        )
    }

def train_and_select_model(task_type: str, preprocessor, X_train, X_test, y_train, y_test):
    results: Dict[str, Dict] = {}
    trained_models: Dict[str, object] = {}

    candidates = _clf_candidates() if task_type == "classification" else _reg_candidates()

    best_name, best_score = None, -1e18

    for name, estimator in candidates.items():
        # Build a simple pipeline: (preprocessor already applied outside) -> model
        model = estimator
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "classification":
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            # try AUC for binary
            auc = None
            try:
                if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(y_test, y_prob)
            except Exception:
                pass
            score = (acc + f1) / 2.0
            results[name] = {"accuracy": acc, "f1": f1, "auc": auc}
        else:
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            score = r2  # primary
            results[name] = {"r2": r2, "rmse": rmse}

        trained_models[name] = model

        if score > best_score:
            best_score = score
            best_name = name

    return best_name, results, trained_models
