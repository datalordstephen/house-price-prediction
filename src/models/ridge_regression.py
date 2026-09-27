"""L2-regularized linear regression."""

from sklearn.linear_model import Ridge

NAME = "Ridge Regression"


def build():
    return Ridge(alpha=1.0)
