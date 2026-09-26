"""Generate the comparison charts embedded in README.md.

Run with: python -m scripts.generate_report_assets
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import NUMERIC_COLS, load_data
from src.train import ARTIFACTS_DIR

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# Fixed color per model identity (not re-cycled by rank), consistent with app.py.
MODEL_COLORS = {
    "Linear Regression": "#4C78A8",
    "Ridge Regression": "#F58518",
    "Decision Tree": "#54A24B",
    "Random Forest": "#E45756",
    "Gradient Boosting": "#72B7B2",
    "Neural Net (MLP)": "#B279A2",
}


def _style_axes(ax):
    ax.grid(axis="x", color="lightgray", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    ordered = metrics_df.sort_values("R2", ascending=False)
    colors = [MODEL_COLORS[m] for m in ordered["Model"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.barh(ordered["Model"], ordered["R2"], color=colors)
    ax1.set_xlabel("R² on held-out test set (higher is better)")
    ax1.invert_yaxis()
    ax1.axvline(0, color="black", linewidth=0.8)
    _style_axes(ax1)

    ax2.barh(ordered["Model"], ordered["RMSE"] / 1e6, color=colors)
    ax2.set_xlabel("RMSE, ₦ millions (lower is better)")
    ax2.invert_yaxis()
    ax2.set_yticklabels([])
    _style_axes(ax2)

    fig.suptitle("Model comparison — test-set performance", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_feature_correlation() -> None:
    df = load_data()
    corr = df[NUMERIC_COLS + ["Price_NGN"]].corr()["Price_NGN"].drop("Price_NGN")
    corr = corr.sort_values()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(corr.index, corr.values, color="#9D9D9D")
    ax.set_xlabel("Correlation with Price_NGN")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-0.05, 0.05)
    _style_axes(ax)
    fig.suptitle("Feature correlation with price (near zero across the board)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "feature_correlation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS_DIR.mkdir(exist_ok=True)
    metrics_df = pd.read_csv(ARTIFACTS_DIR / "metrics.csv")
    plot_model_comparison(metrics_df)
    plot_feature_correlation()
    print(f"Saved charts to {ASSETS_DIR}")
