import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def _is_classification_target(y: pd.Series) -> bool:
    # If non-numeric or few unique values, treat as classification
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return True
    # numeric: small number of unique values indicates classification
    unique_vals = pd.unique(y.dropna())
    return len(unique_vals) <= 10

def detect_task_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature and one target column.")
    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    task_type = "classification" if _is_classification_target(y) else "regression"

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Build preprocessor
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = ColumnTransformer(
        transformers=[
            ("imputer", SimpleImputer(strategy="most_frequent"), cat_cols)
        ],
        remainder="drop"
    )
    # We want impute + one-hot in a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat",OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if task_type=="classification" else None)

    # Fit preprocessor on train, transform both
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # Build feature names for plotting (best-effort)
    feature_names = []
    feature_names.extend(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"]
        if hasattr(ohe, "get_feature_names_out"):
            feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

    return {
        "task_type": task_type,
        "X_train": X_train_t,
        "X_test": X_test_t,
        "y_train": y_train.values if hasattr(y_train, "values") else y_train,
        "y_test": y_test.values if hasattr(y_test, "values") else y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names
    }
