import numpy as np
import pytest

from src.metrics import mae, mdape, r2, rmse


def test_perfect_predictions():
    y_true = [1.0, 2.0, 3.0]
    assert rmse(y_true, y_true) == 0
    assert mae(y_true, y_true) == 0
    assert r2(y_true, y_true) == 1
    assert mdape(y_true, y_true) == 0


def test_known_errors():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [2.0, 2.0, 2.0]
    assert mae(y_true, y_pred) == pytest.approx(2 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2 / 3))
    # predicting the mean everywhere gives R2 == 0 by definition
    assert r2(y_true, y_pred) == pytest.approx(0.0)



def test_mdape_is_median_of_absolute_percentage_errors():
    y_true = [100.0, 200.0, 400.0]
    y_pred = [110.0, 100.0, 400.0]  # errors of 10%, 50%, 0%
    assert mdape(y_true, y_pred) == pytest.approx(0.10)
    # over- and under-prediction by the same fraction count equally
    assert mdape([100.0], [80.0]) == mdape([100.0], [120.0]) == pytest.approx(0.20)
