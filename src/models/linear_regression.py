"""Plain linear regression — the baseline every other model is compared against."""

from sklearn.linear_model import LinearRegression

NAME = "Linear Regression"


def build():
    return LinearRegression()
