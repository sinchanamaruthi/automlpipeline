import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe(include="all").T
    desc["missing"] = df.isna().sum()
    return desc

def plot_missing_values(df: pd.DataFrame):
    mv = df.isna().sum()
    if (mv==0).all():
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    mv.plot(kind="bar", ax=ax)
    ax.set_title("Missing values per column")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def plot_confusion_matrix_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig

def plot_roc_curve_fig(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Binary)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

def _extract_feature_importances(trained_models: Dict[str, object]):
    # Prefer tree model importances if available
    for name in ["RandomForestClassifier", "RandomForestRegressor", "XGBClassifier", "XGBRegressor"]:
        if name in trained_models:
            m = trained_models[name]
            if hasattr(m, "feature_importances_"):
                return name, m.feature_importances_
            # XGB uses feature_importances_ as well (gain-based default)
    # LogisticRegression: use absolute coefficients if available
    for name in ["LogisticRegression"]:
        if name in trained_models:
            m = trained_models[name]
            if hasattr(m, "coef_"):
                import numpy as np
                return name, np.mean(np.abs(m.coef_), axis=0)
    return None, None

def plot_feature_importance_fig(trained_models: Dict[str, object], feature_names):
    name, importances = _extract_feature_importances(trained_models)
    if importances is None or feature_names is None or len(feature_names)==0:
        return None
    # Align lengths best-effort
    k = min(len(importances), len(feature_names))
    imp = pd.Series(importances[:k], index=feature_names[:k]).sort_values(ascending=False)[:20]
    fig, ax = plt.subplots(figsize=(6, 5))
    imp[::-1].plot(kind="barh", ax=ax)  # smallest to top, largest at bottom reversed
    ax.set_title(f"Top Feature Importances ({name})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig

def model_comparison_df(results: Dict[str, Dict], task_type: str) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    if task_type == "classification":
        cols = ["accuracy", "f1", "auc"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols].sort_values(by=["accuracy","f1"], ascending=False)
    else:
        cols = ["r2", "rmse"]
        return df[cols].sort_values(by=["r2"], ascending=False)

def generate_summary_text(best_name: str, results: Dict[str, Dict], task_type: str) -> str:
    if task_type == "classification":
        r = results[best_name]
        acc = r.get("accuracy", None)
        f1 = r.get("f1", None)
        auc = r.get("auc", None)
        parts = [f"Best model **{best_name}** achieved accuracy={acc:.3f}, F1={f1:.3f}"]
        if auc is not None:
            parts.append(f"ROC-AUC={auc:.3f}")
        parts.append("Model was selected automatically based on primary metrics.")
        return " | ".join(parts)
    else:
        r = results[best_name]
        r2 = r.get("r2", None)
        rmse = r.get("rmse", None)
        return f"Best model **{best_name}** achieved R²={r2:.3f} with RMSE={rmse:.3f}. Model was auto-selected based on R²."
