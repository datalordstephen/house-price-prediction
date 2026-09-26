"""L2-regularized linear regression."""

from sklearn.linear_model import Ridge

from src.config import RANDOM_STATE

NAME = "Ridge Regression"


def build():
    return Ridge(alpha=1.0, random_state=RANDOM_STATE)
