"""Panel construction, splits, and baseline forecasts for groundwater levels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import kge, mae, nse, rmse


def build_daily_panel(
    df: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    value_col: str = "value",
    site_col: str = "site_no",
) -> pd.DataFrame:
    """Build a complete daily panel with an explicit observed/missing mask.

    Each site is reindexed to every calendar day between ``start`` and ``end``.
    Days without an observation stay NaN and are marked ``observed=False``;
    this is deliberate, so models are evaluated on real gaps rather than on
    silently interpolated values.
    """
    required = {site_col, "datetime", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df.empty:
        return pd.DataFrame(columns=[site_col, "date", value_col, "observed"])

    work = df.copy()
    work["date"] = pd.to_datetime(work["datetime"]).dt.normalize()
    work = work.dropna(subset=[site_col, "date", value_col])
    work = (
        work.groupby([site_col, "date"], as_index=False)[value_col]
        .mean()
        .sort_values([site_col, "date"])
    )
    if work.empty:
        return pd.DataFrame(columns=[site_col, "date", value_col, "observed"])

    start_ts = pd.Timestamp(start) if start is not None else work["date"].min()
    end_ts = pd.Timestamp(end) if end is not None else work["date"].max()
    dates = pd.date_range(start_ts, end_ts, freq="D")

    frames: list[pd.DataFrame] = []
    for site, group in work.groupby(site_col, sort=True):
        panel = (
            group.set_index("date")
            .reindex(dates)
            .rename_axis("date")
            .reset_index()
        )
        panel[site_col] = site
        panel["observed"] = panel[value_col].notna()
        frames.append(panel)

    result = pd.concat(frames, ignore_index=True)
    return (
        result[[site_col, "date", value_col, "observed"]]
        .sort_values([site_col, "date"])
        .reset_index(drop=True)
    )


def to_wide(panel: pd.DataFrame, *, value_col: str = "value") -> pd.DataFrame:
    """Pivot a long panel to ``date x site`` wide format."""
    return (
        panel.pivot(index="date", columns="site_no", values=value_col)
        .sort_index()
    )


def temporal_split(
    panel: pd.DataFrame,
    *,
    train_end: str = "2015-12-31",
    val_end: str = "2018-12-31",
) -> pd.DataFrame:
    """Label rows as train/val/test by calendar date."""
    panel = panel.copy()
    dates = panel["date"]
    panel["split"] = np.select(
        [dates <= pd.Timestamp(train_end), dates <= pd.Timestamp(val_end)],
        ["train", "val"],
        default="test",
    )
    return panel


def site_split(
    panel: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    stratify_col: str | None = "state_cd",
) -> pd.DataFrame:
    """Assign whole sites to train/val/test, optionally stratified by state."""
    meta_cols = ["site_no"]
    if stratify_col and stratify_col in panel.columns:
        meta_cols.append(stratify_col)
    sites = panel[meta_cols].drop_duplicates("site_no").copy()
    rng = np.random.default_rng(seed)

    def label_group(group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        order = rng.permutation(n)
        labels = np.array(["test"] * n, dtype=object)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        labels[order[:n_train]] = "train"
        labels[order[n_train : n_train + n_val]] = "val"
        group = group.copy()
        group["split"] = labels
        return group

    if stratify_col and stratify_col in sites.columns:
        sites = (
            sites.groupby(stratify_col, group_keys=False)
            .apply(label_group, include_groups=False)
            .reset_index(drop=True)
        )
    else:
        sites = label_group(sites)

    return panel.merge(sites[["site_no", "split"]], on="site_no", how="left")


def add_baselines(panel: pd.DataFrame, *, value_col: str = "value") -> pd.DataFrame:
    """Add persistence and seasonal-naive forecasts.

    Persistence uses the most recent previous observation (no future data).
    Seasonal naive uses the observation from 365 days earlier; it is left NaN
    when that value is missing rather than filled forward, which could leak
    information from later observations.
    """
    panel = panel.sort_values(["site_no", "date"]).copy()
    grouped = panel.groupby("site_no", sort=False)[value_col]
    panel["persistence"] = grouped.transform(lambda s: s.shift(1).ffill())
    panel["seasonal_naive"] = grouped.transform(lambda s: s.shift(365))
    return panel


def evaluate_predictions(
    panel: pd.DataFrame,
    pred_col: str,
    *,
    split: str = "test",
    value_col: str = "value",
    min_points: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate one forecast column per site and return per-site + aggregate rows."""
    if "split" not in panel.columns:
        raise ValueError("panel must contain a 'split' column")

    subset = panel[
        (panel["split"] == split)
        & panel[value_col].notna()
        & panel[pred_col].notna()
    ]
    rows: list[dict] = []
    for site, group in subset.groupby("site_no", sort=True):
        y_true = group[value_col].to_numpy(dtype=float)
        y_pred = group[pred_col].to_numpy(dtype=float)
        if len(group) < min_points:
            continue
        rows.append(
            {
                "site_no": site,
                "n": len(group),
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "nse": nse(y_true, y_pred),
                "kge": kge(y_true, y_pred),
            }
        )

    per_site = pd.DataFrame(rows)
    if per_site.empty:
        return per_site, {"n_sites": 0, "metric": pred_col}

    aggregate = {
        "metric": pred_col,
        "n_sites": int(len(per_site)),
        "rmse_mean": float(per_site["rmse"].mean()),
        "rmse_median": float(per_site["rmse"].median()),
        "mae_mean": float(per_site["mae"].mean()),
        "nse_mean": float(per_site["nse"].mean()),
        "nse_median": float(per_site["nse"].median()),
        "kge_mean": float(per_site["kge"].mean()),
        "kge_median": float(per_site["kge"].median()),
        "nse_positive": float((per_site["nse"] > 0).mean()),
        "kge_positive": float((per_site["kge"] > 0).mean()),
    }
    return per_site, aggregate
