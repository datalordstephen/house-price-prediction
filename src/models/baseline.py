"""Baseline that always predicts the training-set mean (of log price)."""

from sklearn.dummy import DummyRegressor

NAME = "Baseline (Mean)"


def build():
    return DummyRegressor()
