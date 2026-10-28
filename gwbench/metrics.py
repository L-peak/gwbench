"""Evaluation metrics for hydrological time-series models.

All functions ignore rows where either ``y_true`` or ``y_pred`` is NaN, so
they can be applied directly to masked benchmark outputs.
"""

from __future__ import annotations

import numpy as np


def _paired(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


def rmse(y_true, y_pred) -> float:
    true, pred = _paired(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(y_true, y_pred) -> float:
    true, pred = _paired(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)))


def nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe efficiency (1 is perfect, <0 is worse than the mean)."""
    true, pred = _paired(y_true, y_pred)
    if true.size < 2:
        return float("nan")
    denominator = np.sum((true - true.mean()) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.sum((true - pred) ** 2) / denominator)


def kge(y_true, y_pred) -> float:
    """Kling-Gupta efficiency (1 is perfect)."""
    true, pred = _paired(y_true, y_pred)
    if true.size < 2:
        return float("nan")

    true_std = np.std(true)
    pred_std = np.std(pred)
    true_mean = np.mean(true)
    pred_mean = np.mean(pred)

    if true_std == 0 or true_mean == 0:
        return float("nan")

    correlation = np.corrcoef(true, pred)[0, 1]
    alpha = pred_std / true_std
    beta = pred_mean / true_mean
    return float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def mase(y_true, y_pred, y_train, *, seasonality: int = 1) -> float:
    """Mean absolute scaled error using the training series as the scale."""
    true, pred = _paired(y_true, y_pred)
    train = np.asarray(y_train, dtype=float).ravel()
    train = train[np.isfinite(train)]
    if true.size == 0 or train.size <= seasonality:
        return float("nan")

    scale = np.mean(np.abs(train[seasonality:] - train[:-seasonality]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / scale)


def persistence(y_history, horizon: int = 1) -> np.ndarray:
    """Persistence forecast: repeat the last observed value."""
    values = np.asarray(y_history, dtype=float).ravel()
    if values.size == 0:
        return np.empty(horizon, dtype=float)
    return np.full(horizon, values[-1], dtype=float)


def seasonal_naive(y_history, horizon: int = 1, *, seasonality: int = 365) -> np.ndarray:
    """Seasonal naive forecast: repeat values from ``seasonality`` steps ago."""
    values = np.asarray(y_history, dtype=float).ravel()
    if values.size < seasonality:
        return persistence(values, horizon)
    forecast = []
    for step in range(horizon):
        index = len(values) - seasonality + (step % seasonality)
        forecast.append(values[index])
    return np.asarray(forecast, dtype=float)
