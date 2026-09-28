import numpy as np
import pytest

from src.metrics import mae, r2, rmse


def test_perfect_predictions():
    y_true = [1.0, 2.0, 3.0]
    assert rmse(y_true, y_true) == 0
    assert mae(y_true, y_true) == 0
    assert r2(y_true, y_true) == 1


def test_known_errors():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 2.0]
    assert mae(y_true, y_pred) == pytest.approx(2 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2 / 3))
    # predicting the mean everywhere gives R2 == 0 by definition
    assert r2(y_true, y_pred) == pytest.approx(0.0)

