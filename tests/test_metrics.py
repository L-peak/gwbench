import numpy as np

from gwbench.metrics import kge, mae, nse, persistence, rmse, seasonal_naive


def test_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert nse(y, y) == 1.0
    assert kge(y, y) == 1.0


def test_nan_pairing():
    y_true = np.array([1.0, np.nan, 3.0])
    y_pred = np.array([1.0, 2.0, np.nan])
    assert rmse(y_true, y_pred) == 0.0


def test_persistence():
    assert np.allclose(persistence([1.0, 2.0, 3.0], horizon=2), [3.0, 3.0])


def test_seasonal_naive():
    history = np.arange(10, dtype=float)
    assert np.allclose(seasonal_naive(history, horizon=3, seasonality=4), [6.0, 7.0, 8.0])
