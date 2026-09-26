"""Streamlit dashboard: compare regression models and predict Nigerian house prices.

Run with: streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import CATEGORICAL_COLS, NUMERIC_COLS, load_data
from src.models import slugify
from src.train import MODELS_DIR, train_all

st.set_page_config(page_title="Nigerian House Price Predictor", layout="wide")


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_metrics_and_models():
    metrics_path = MODELS_DIR / "metrics.csv"
    if not metrics_path.exists():
        with st.spinner("No trained models found — training all models now (one-time)..."):
            train_all()

    metrics_df = pd.read_csv(metrics_path)
    models = {}
    for name in metrics_df["Model"]:
        models[name] = joblib.load(MODELS_DIR / f"{slugify(name)}.joblib")
    return metrics_df, models


df = get_dataset()
metrics_df, models = get_metrics_and_models()

st.title("🏠 Nigerian House Price Predictor")
st.caption(f"Trained on {len(df):,} listings across {df['City'].nunique()} cities.")

tab_compare, tab_predict = st.tabs(["Model Comparison", "Predict a Price"])

with tab_compare:
    st.subheader("Test-set performance by model")

    ordered = metrics_df.sort_values("R2", ascending=False)
    model_order = ordered["Model"].tolist()
    colors = plt.get_cmap("tab10").colors

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(model_order, ordered["R2"], color=colors[: len(model_order)])
        ax.set_xlabel("R² (higher is better)")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.grid(axis="x", color="lightgray", linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(model_order, ordered["RMSE"], color=colors[: len(model_order)])
        ax.set_xlabel("RMSE in ₦ (lower is better)")
        ax.invert_yaxis()
        ax.grid(axis="x", color="lightgray", linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)

    st.subheader("Full metrics table")
    st.dataframe(
        ordered.set_index("Model").style.format(
            {"RMSE": "₦{:,.0f}", "MAE": "₦{:,.0f}", "R2": "{:.3f}", "CV_R2_mean": "{:.3f}", "CV_R2_std": "{:.3f}"}
        ),
        use_container_width=True,
    )

    best_model = model_order[0]
    st.info(f"**Best performing model:** {best_model} (R² = {ordered.iloc[0]['R2']:.3f})")

with tab_predict:
    st.subheader("Enter a listing's details")

    model_choice = st.selectbox("Model to use for prediction", model_order)

    input_cols = st.columns(3)
    inputs = {}
    for i, col in enumerate(CATEGORICAL_COLS):
        options = sorted(df[col].dropna().unique())
        inputs[col] = input_cols[i % 3].selectbox(col.replace("_", " "), options)

    numeric_cols = st.columns(len(NUMERIC_COLS))
    for i, col in enumerate(NUMERIC_COLS):
        series = df[col]
        if col in ("Bedrooms", "Bathrooms"):
            inputs[col] = numeric_cols[i].number_input(
                col, min_value=0, max_value=int(series.max()) + 2, value=int(series.median()), step=1
            )
        elif col == "Size_sqm":
            inputs[col] = numeric_cols[i].number_input(
                "Size (sqm)", min_value=1.0, value=float(series.median()), step=1.0
            )
        else:
            inputs[col] = numeric_cols[i].number_input(
                col, value=float(series.median()), format="%.5f"
            )

    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame([inputs])
        pipeline = models[model_choice]
        prediction = pipeline.predict(input_df)[0]
        st.success(f"Estimated price ({model_choice}): **₦{prediction:,.0f}**")

        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            st.subheader(f"What drives this prediction — {model_choice} feature importance")
            feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
            importances = pipeline.named_steps["model"].feature_importances_
            imp_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .head(12)
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(imp_df["Feature"], imp_df["Importance"], color=plt.get_cmap("tab10").colors[0])
            ax.invert_yaxis()
            ax.grid(axis="x", color="lightgray", linewidth=0.5)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            st.pyplot(fig)
