"""Streamlit dashboard: compare regression models and predict Nigerian house prices.

Run with: streamlit run app.py
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import NUMERIC_COLS, load_data
from src.models import MODEL_COLORS, slugify
from src.train import ARTIFACTS_DIR, train_all

st.set_page_config(page_title="Nigerian House Price Predictor", layout="wide")


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_metrics_and_models():
    metrics_path = ARTIFACTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        with st.spinner("No trained models found — training all models now (one-time)..."):
            train_all()

    metrics_df = pd.read_csv(metrics_path)
    models = {}
    for name in metrics_df["Model"]:
        models[name] = joblib.load(ARTIFACTS_DIR / f"{slugify(name)}.joblib")
    return metrics_df, models


def show_barh(labels, values, colors, xlabel, zero_line=False):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    if zero_line:
        ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", color="lightgray", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)


df = get_dataset()
metrics_df, models = get_metrics_and_models()

st.title("🏠 Nigerian House Price Predictor")
st.caption(
    f"Trained on {len(df):,} listings across {df['town'].nunique()} towns "
    f"in {df['state'].nunique()} states."
)

tab_compare, tab_predict = st.tabs(["Model Comparison", "Predict a Price"])

with tab_compare:
    st.subheader("Test-set performance by model")

    ordered = metrics_df.sort_values("R2_log", ascending=False)
    model_order = ordered["Model"].tolist()
    colors = [MODEL_COLORS[m] for m in model_order]

    col1, col2 = st.columns(2)
    with col1:
        show_barh(
            model_order, ordered["R2_log"], colors,
            "R² on log(price) (higher is better)", zero_line=True,
        )
    with col2:
        show_barh(model_order, ordered["RMSE"] / 1e6, colors, "RMSE, ₦ millions (lower is better)")

    st.subheader("Full metrics table")
    st.dataframe(
        ordered.set_index("Model").style.format(
            {
                "R2_log": "{:.3f}",
                "RMSE": "₦{:,.0f}",
                "MAE": "₦{:,.0f}",
                "CV_R2_log_mean": "{:.3f}",
                "CV_R2_log_std": "{:.3f}",
            }
        ),
        width="stretch",
    )

    best = ordered.iloc[0]
    st.info(
        f"**Best performing model:** {best['Model']} "
        f"(R² on log price = {best['R2_log']:.3f}, MAE = ₦{best['MAE'] / 1e6:,.1f}M)"
    )

with tab_predict:
    st.subheader("Enter a listing's details")

    model_choice = st.selectbox("Model to use for prediction", model_order)

    inputs = {}
    cat_cols = st.columns(3)
    states = sorted(df["state"].unique())
    inputs["state"] = cat_cols[0].selectbox(
        "State", states, index=states.index(df["state"].mode()[0])
    )
    # Only offer towns in the chosen state (some town names exist in several states).
    towns = sorted(df.loc[df["state"] == inputs["state"], "town"].unique())
    inputs["town"] = cat_cols[1].selectbox("Town", towns)
    inputs["title"] = cat_cols[2].selectbox("Property type", sorted(df["title"].unique()))

    num_cols = st.columns(len(NUMERIC_COLS))
    for i, col in enumerate(NUMERIC_COLS):
        series = df[col]
        inputs[col] = num_cols[i].number_input(
            col.replace("_", " ").capitalize(),
            min_value=int(series.min()),
            max_value=int(series.max()),
            value=int(series.median()),
            step=1,
        )

    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame([inputs])
        pipeline = models[model_choice]
        prediction = pipeline.predict(input_df)[0]
        st.success(f"Estimated price ({model_choice}): **₦{prediction:,.0f}**")

        # The fitted estimator sits inside the TransformedTargetRegressor wrapper.
        estimator = pipeline.named_steps["model"].regressor_
        if hasattr(estimator, "feature_importances_"):
            st.subheader(f"Overall feature importance — {model_choice}")
            feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
            imp_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": estimator.feature_importances_})
                .sort_values("Importance", ascending=False)
                .head(12)
            )
            imp_df["Feature"] = imp_df["Feature"].str.replace(r"^(cat|num)__", "", regex=True)
            show_barh(imp_df["Feature"], imp_df["Importance"], MODEL_COLORS[model_choice], "Importance")
