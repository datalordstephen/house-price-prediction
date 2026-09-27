"""Generate the charts embedded in README.md (model comparison + price breakdowns).

Run with: python -m scripts.generate_report_assets
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import TARGET, load_data
from src.models import MODEL_COLORS
from src.train import ARTIFACTS_DIR

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# Neutral colour for data (not model) charts, distinct from every MODEL_COLORS entry.
DATA_COLOR = "#5A6B7B"


def _style_axes(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color="lightgray", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    ordered = metrics_df.sort_values("R2_log", ascending=False)
    colors = [MODEL_COLORS[m] for m in ordered["Model"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.barh(ordered["Model"], ordered["R2_log"], color=colors)
    ax1.set_xlabel("R² on log(price), held-out test set (higher is better)")
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


def _median_price_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.groupby(col)[TARGET].agg(median="median", count="size")


def plot_price_by_bedrooms(df: pd.DataFrame) -> None:
    stats = _median_price_by(df, "bedrooms")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(stats.index.astype(int).astype(str), stats["median"] / 1e6, color=DATA_COLOR)
    ax.bar_label(bars, labels=[f"n={c:,}" for c in stats["count"]], fontsize=8, padding=2)
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Median price, ₦ millions")
    _style_axes(ax, grid_axis="y")
    fig.suptitle("Median listing price by number of bedrooms", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "price_by_bedrooms.png", dpi=150)
    plt.close(fig)


def plot_price_by_title(df: pd.DataFrame) -> None:
    stats = _median_price_by(df, "title").sort_values("median", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(stats.index, stats["median"] / 1e6, color=DATA_COLOR)
    ax.bar_label(bars, labels=[f"n={c:,}" for c in stats["count"]], fontsize=8, padding=3)
    ax.invert_yaxis()
    ax.set_xlabel("Median price, ₦ millions")
    _style_axes(ax)
    fig.suptitle("Median listing price by property type", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "price_by_title.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS_DIR.mkdir(exist_ok=True)
    metrics_df = pd.read_csv(ARTIFACTS_DIR / "metrics.csv")
    plot_model_comparison(metrics_df)
    df = load_data()
    plot_price_by_bedrooms(df)
    plot_price_by_title(df)
    print(f"Saved charts to {ASSETS_DIR}")
