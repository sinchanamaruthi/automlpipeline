import streamlit as st
import pandas as pd
from preprocessing.cleaner import preprocess_data
from models.trainer import train_and_select_model

st.title("Phase 1 AutoML Pipeline 🚀")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(data.head())
    st.write(f"Shape: {data.shape}")

    try:
        X_train, X_test, y_train, y_test = preprocess_data(data)
        best_model, results, all_preds = train_and_select_model(X_train, X_test, y_train, y_test)

        st.subheader("✅ Model Results")
        st.write(results)
        st.success(f"Best Model Selected: {best_model}")

        # --- Bar Chart of Model Scores ---
        st.subheader("📈 Model Comparison")
        score_df = pd.DataFrame(results).T
        st.bar_chart(score_df[["accuracy", "f1_score"]])

    except Exception as e:
        st.error(f"Error during training: {e}")
