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


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")
