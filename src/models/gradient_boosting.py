"""Gradient boosted trees regressor."""

from sklearn.ensemble import GradientBoostingRegressor

from src.config import RANDOM_STATE

NAME = "Gradient Boosting"


def build():
    return GradientBoostingRegressor(random_state=RANDOM_STATE)
