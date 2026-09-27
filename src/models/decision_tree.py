"""Single decision tree regressor."""

from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE

NAME = "Decision Tree"


def build():
    return DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE)
