"""Plain linear regression."""

from sklearn.linear_model import LinearRegression

NAME = "Linear Regression"


def build():
    return LinearRegression()
