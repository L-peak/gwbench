"""Windowed forecasting datasets for groundwater panels.

The panel is stored as a dense ``[n_sites, n_days]`` array.  Windows are
sliced on demand, so memory scales with the panel size, not with the number of
training windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .metrics import kge, mae, nse, rmse


@dataclass
class PanelArrays:
    sites: list[str]
    dates: pd.DatetimeIndex
    values: np.ndarray  # [n_sites, n_days], float32
    observed: np.ndarray  # [n_sites, n_days], bool


def panel_to_arrays(panel: pd.DataFrame, *, value_col: str = "value") -> PanelArrays:
    """Pivot a long panel into dense site x date arrays."""
    values = panel.pivot(index="date", columns="site_no", values=value_col).sort_index()
    observed = (
        panel.pivot(index="date", columns="site_no", values="observed")
        .sort_index()
        .astype(bool)
    )
    common = values.columns.intersection(observed.columns)
    values = values[common]
    observed = observed[common]
    return PanelArrays(
        sites=[str(site) for site in common],
        dates=pd.DatetimeIndex(values.index),
        values=values.to_numpy(dtype=np.float32).T,
        observed=observed.to_numpy(dtype=bool).T,
    )


def compute_site_stats(
    arrays: PanelArrays,
    *,
    train_dates: np.ndarray | None = None,
    train_sites: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-site mean/std from training observations, with a global fallback.

    ``train_sites`` is used for leave-site-out splits: sites not in the
    training set receive the global training mean/std instead of leaking their
    own statistics.
    """
    n_sites = len(arrays.sites)
    if train_dates is None:
        train_dates = np.ones(len(arrays.dates), dtype=bool)
    if train_sites is None:
        train_sites = np.ones(n_sites, dtype=bool)

    means = np.zeros(n_sites, dtype=np.float32)
    stds = np.ones(n_sites, dtype=np.float32)

    train_values = arrays.values[:, train_dates]
    train_observed = arrays.observed[:, train_dates]
    global_values = train_values[train_sites][train_observed[train_sites]]
    global_mean = float(np.mean(global_values)) if global_values.size else 0.0
    global_std = float(np.std(global_values)) if global_values.size else 1.0
    if not np.isfinite(global_std) or global_std < 1e-6:
        global_std = 1.0

    for site in range(n_sites):
        if not train_sites[site]:
            means[site] = global_mean
            stds[site] = global_std
            continue
        values = train_values[site][train_observed[site]]
        if values.size >= 10:
            means[site] = float(np.mean(values))
            std = float(np.std(values))
            stds[site] = std if std > 1e-6 else 1.0
        else:
            means[site] = global_mean
            stds[site] = global_std

    return means, stds


def build_anchors(
    arrays: PanelArrays,
    *,
    lookback: int,
    horizon: int,
    stride: int = 7,
    min_history_obs: int = 30,
) -> np.ndarray:
    """Return valid ``(site_index, anchor_index)`` pairs.

    An anchor is valid when the target day is observed and the input window
    contains at least ``min_history_obs`` observations.
    """
    anchors: list[tuple[int, int]] = []
    n_dates = len(arrays.dates)
    for site in range(len(arrays.sites)):
        for anchor in range(lookback - 1, n_dates - horizon, stride):
            if not arrays.observed[site, anchor + horizon]:
                continue
            history = arrays.observed[site, anchor - lookback + 1 : anchor + 1]
            if history.sum() < min_history_obs:
                continue
            anchors.append((site, anchor))
    if not anchors:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(anchors, dtype=np.int64)


def split_anchors_temporal(
    anchors: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    train_end: str = "2015-12-31",
    val_end: str = "2018-12-31",
) -> dict[str, np.ndarray]:
    """Split anchors by the calendar date of the input window's last day."""
    if anchors.size == 0:
        return {"train": anchors, "val": anchors, "test": anchors}
    anchor_dates = dates[anchors[:, 1]]
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    train = anchors[anchor_dates <= train_end_ts]
    val = anchors[(anchor_dates > train_end_ts) & (anchor_dates <= val_end_ts)]
    test = anchors[anchor_dates > val_end_ts]
    return {"train": train, "val": val, "test": test}


def split_anchors_by_site(
    anchors: np.ndarray,
    site_split: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split anchors by a per-site label array (``train``/``val``/``test``)."""
    if anchors.size == 0:
        return {"train": anchors, "val": anchors, "test": anchors}
    labels = site_split[anchors[:, 0]]
    return {
        "train": anchors[labels == "train"],
        "val": anchors[labels == "val"],
        "test": anchors[labels == "test"],
    }


class WindowDataset(Dataset):
    """On-demand windows with value, observation-mask, and time features."""

    def __init__(
        self,
        arrays: PanelArrays,
        anchors: np.ndarray,
        *,
        lookback: int,
        horizon: int,
        means: np.ndarray,
        stds: np.ndarray,
        add_time_features: bool = True,
        residual: bool = False,
        mask_fraction: float = 0.0,
        mask_seed: int = 0,
    ) -> None:
        self.arrays = arrays
        self.anchors = anchors
        self.lookback = lookback
        self.horizon = horizon
        self.means = means
        self.stds = stds
        self.add_time_features = add_time_features
        self.residual = residual
        self.mask_fraction = mask_fraction
        self.mask_seed = mask_seed

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int):
        site, anchor = self.anchors[index]
        start = anchor - self.lookback + 1

        values = self.arrays.values[site, start : anchor + 1]
        observed_mask = self.arrays.observed[site, start : anchor + 1]
        input_mask = observed_mask
        mean = float(self.means[site])
        std = float(self.stds[site])

        if self.mask_fraction > 0 and observed_mask.any():
            observed_indices = np.flatnonzero(observed_mask)
            n_hide = int(round(self.mask_fraction * len(observed_indices)))
            if n_hide > 0:
                rng = np.random.default_rng(self.mask_seed + index)
                hide = rng.choice(observed_indices, size=n_hide, replace=False)
                input_mask = observed_mask.copy()
                input_mask[hide] = False

        normalized = (values - mean) / std
        x = np.stack(
            [
                np.where(input_mask, normalized, 0.0),
                input_mask.astype(np.float32),
            ],
            axis=-1,
        )

        if self.add_time_features:
            day_of_year = self.arrays.dates[start : anchor + 1].dayofyear
            angle = 2.0 * np.pi * day_of_year / 365.25
            x = np.concatenate(
                [
                    x,
                    np.stack(
                        [np.sin(angle), np.cos(angle)], axis=-1
                    ).astype(np.float32),
                ],
                axis=-1,
            )

        target = (self.arrays.values[site, anchor + self.horizon] - mean) / std
        if self.residual:
            observed_indices = np.flatnonzero(observed_mask)
            if observed_indices.size:
                last_value = values[observed_indices[-1]]
                last_normalized = (last_value - mean) / std
            else:
                last_normalized = 0.0
            target = target - last_normalized

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(site, dtype=torch.long),
            torch.tensor(anchor, dtype=torch.long),
        )


def predict_loader(model, loader, device: torch.device) -> np.ndarray:
    """Return model predictions in normalized space."""
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for x, _, site_id, _ in loader:
            x = x.to(device)
            site_id = site_id.to(device)
            predictions.append(model(x, site_id).detach().cpu().numpy())
    if not predictions:
        return np.array([], dtype=np.float32)
    return np.concatenate(predictions)


def build_prediction_frame(
    arrays: PanelArrays,
    anchors: np.ndarray,
    predictions_normalized: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    *,
    lookback: int,
    horizon: int,
    residual: bool = False,
) -> pd.DataFrame:
    """Convert normalized predictions to a tidy frame with raw targets and baselines."""
    rows: list[dict] = []
    for (site, anchor), prediction in zip(anchors, predictions_normalized):
        target_index = anchor + horizon
        raw_target = float(arrays.values[site, target_index])

        start = anchor - lookback + 1
        history_values = arrays.values[site, start : anchor + 1]
        history_mask = arrays.observed[site, start : anchor + 1]
        last_observed = (
            float(history_values[history_mask][-1])
            if history_mask.any()
            else np.nan
        )
        last_normalized = (
            (last_observed - float(means[site])) / float(stds[site])
            if np.isfinite(last_observed)
            else 0.0
        )
        predicted_normalized = (
            last_normalized + float(prediction) if residual else float(prediction)
        )
        raw_prediction = predicted_normalized * float(stds[site]) + float(
            means[site]
        )

        seasonal_index = target_index - 365
        if seasonal_index >= 0 and arrays.observed[site, seasonal_index]:
            seasonal = float(arrays.values[site, seasonal_index])
        else:
            seasonal = np.nan

        rows.append(
            {
                "site_no": arrays.sites[site],
                "date": arrays.dates[target_index],
                "y_true": raw_target,
                "y_pred": raw_prediction,
                "persistence": last_observed,
                "seasonal_naive": seasonal,
            }
        )
    return pd.DataFrame(rows)


def evaluate_prediction_frame(frame: pd.DataFrame, *, min_points: int = 30) -> tuple[pd.DataFrame, dict]:
    """Evaluate model/persistence/seasonal predictions per site on a tidy frame."""
    metric_cols = ["y_pred", "persistence", "seasonal_naive"]
    per_site_rows: list[dict] = []

    for site, group in frame.groupby("site_no", sort=True):
        if len(group) < min_points:
            continue
        row = {"site_no": site, "n": len(group)}
        y_true = group["y_true"].to_numpy(dtype=float)
        for column in metric_cols:
            y_pred = group[column].to_numpy(dtype=float)
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() < min_points:
                row[f"{column}_rmse"] = np.nan
                row[f"{column}_mae"] = np.nan
                row[f"{column}_nse"] = np.nan
                row[f"{column}_kge"] = np.nan
                continue
            true = y_true[mask]
            pred = y_pred[mask]
            row[f"{column}_rmse"] = rmse(true, pred)
            row[f"{column}_mae"] = mae(true, pred)
            row[f"{column}_nse"] = nse(true, pred)
            row[f"{column}_kge"] = kge(true, pred)
        per_site_rows.append(row)

    per_site = pd.DataFrame(per_site_rows)
    if per_site.empty:
        return per_site, {"n_sites": 0}

    aggregate = {"n_sites": int(len(per_site))}
    for column in metric_cols:
        for metric in ("rmse", "mae", "nse", "kge"):
            key = f"{column}_{metric}"
            if key in per_site.columns:
                aggregate[f"{key}_mean"] = float(per_site[key].mean())
                aggregate[f"{key}_median"] = float(per_site[key].median())
    return per_site, aggregate
