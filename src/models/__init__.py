"""Registry aggregating every algorithm module for src.train and app.py."""

from src.models import (
    baseline,
    decision_tree,
    gradient_boosting,
    linear_regression,
    neural_net,
    random_forest,
    ridge_regression,
)

_MODEL_MODULES = [
    baseline,
    linear_regression,
    ridge_regression,
    decision_tree,
    random_forest,
    gradient_boosting,
    neural_net,
]

MODEL_REGISTRY = {module.NAME: module.build for module in _MODEL_MODULES}

# Fixed colour per model identity (not re-cycled by rank), shared by app.py and
# scripts/generate_report_assets.py so each model keeps one colour everywhere.
MODEL_COLORS = {
    "Baseline (Mean)": "#9D9D9D",
    "Linear Regression": "#4C78A8",
    "Ridge Regression": "#F58518",
    "Decision Tree": "#54A24B",
    "Random Forest": "#E45756",
    "Gradient Boosting": "#72B7B2",
    "Neural Net (MLP)": "#B279A2",
}


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")
