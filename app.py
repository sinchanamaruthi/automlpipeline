import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from preprocessing import detect_task_and_split
from models import train_and_select_model
from utils import (
    describe_dataframe, 
    plot_missing_values, 
    plot_confusion_matrix_fig, 
    plot_roc_curve_fig, 
    plot_feature_importance_fig, 
    model_comparison_df, 
    generate_summary_text
)

st.set_page_config(page_title="AutoML Phase 1", page_icon="🤖", layout="wide")

st.title("🤖 AutoML Pipeline — Phase 1 (Polished)")

with st.sidebar:
    st.markdown("### Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    st.caption("Assumptions:")
    st.caption("- Last column is the target")
    st.caption("- Handles classification & regression")
    st.caption("- Minimal preprocessing: impute + one-hot")

if not uploaded_file:
    st.info("Upload a CSV to get started.")
    st.stop()

# Read data
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📂 Data Overview", "⚙️ Preprocessing", "🏋️ Model Training", "🔮 Predictions & Insights"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.subheader("Quick Stats")
    stats = describe_dataframe(df)
    st.dataframe(stats, use_container_width=True)
    st.subheader("Missing Values")
    fig_mv = plot_missing_values(df)
    if fig_mv:
        st.pyplot(fig_mv)
    else:
        st.write("No missing values detected. ✅")

# Split & detect task
with tab2:
    st.subheader("Task Detection & Split")
    try:
        info = detect_task_and_split(df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    X_train, X_test = info["X_train"], info["X_test"]
    y_train, y_test = info["y_train"], info["y_test"]
    task_type = info["task_type"]
    preprocessor = info["preprocessor"]
    feature_names = info["feature_names"]

    st.write(f"**Detected Task:** `{task_type}`")
    st.write(f"Train size: {X_train.shape} | Test size: {X_test.shape}")
    st.write("Preprocessing: SimpleImputer (median/mode) + OneHotEncoder(handle_unknown='ignore')")

with tab3:
    st.subheader("Training & Auto-Selection")
    try:
        best_name, results, trained_models = train_and_select_model(
            task_type, preprocessor, X_train, X_test, y_train, y_test
        )
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    st.success(f"✅ Best Model: **{best_name}**")
    comp_df = model_comparison_df(results, task_type)
    st.dataframe(comp_df, use_container_width=True)

    st.subheader("Model Score Comparison")
    if task_type == "classification":
        st.bar_chart(comp_df[["accuracy", "f1"]])
    else:
        st.bar_chart(comp_df[["r2", "rmse"]])

with tab4:
    st.subheader("Evaluation Visuals & Insights")
    if task_type == "classification":
        # Confusion matrix & ROC (binary only)
        best = trained_models[best_name]
        y_pred = best.predict(X_test)
        from sklearn.metrics import roc_curve, roc_auc_score
        from sklearn.preprocessing import label_binarize

        # Confusion matrix
        fig_cm = plot_confusion_matrix_fig(y_test, y_pred)
        st.pyplot(fig_cm)

        # ROC curve (only if binary)
        try:
            classes = np.unique(y_test)
            if len(classes) == 2:
                if hasattr(best, "predict_proba"):
                    y_prob = best.predict_proba(X_test)[:, 1]
                elif hasattr(best, "decision_function"):
                    y_scores = best.decision_function(X_test)
                    # scale to 0-1 via min-max for plotting probabilities-like
                    y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9)
                else:
                    y_prob = None

                if y_prob is not None:
                    fig_roc = plot_roc_curve_fig(y_test, y_prob)
                    st.pyplot(fig_roc)
            else:
                st.caption("ROC curve shown only for binary classification.")
        except Exception as e:
            st.warning(f"ROC plot skipped: {e}")

        # Feature importance (if available)
        try:
            fig_fi = plot_feature_importance_fig(trained_models, feature_names)
            if fig_fi:
                st.pyplot(fig_fi)
        except Exception as e:
            st.warning(f"Feature importance not available: {e}")
    else:
        # Regression: feature importance if available
        try:
            fig_fi = plot_feature_importance_fig(trained_models, feature_names)
            if fig_fi:
                st.pyplot(fig_fi)
        except Exception as e:
            st.warning(f"Feature importance not available: {e}")

    st.subheader("Auto Summary")
    st.write(generate_summary_text(best_name, results, task_type))

st.markdown("---")
st.caption("Made with ❤️ for your final year project")
