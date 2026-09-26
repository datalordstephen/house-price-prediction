"""Random forest ensemble regressor."""

from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE

NAME = "Random Forest"


def build():
    return RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    )
