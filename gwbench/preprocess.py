"""Preprocessing helpers for groundwater-level time series."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Depth to water below land surface is normally positive.  Large negative
# values can be real (artesian wells) but are rare; these bounds only flag
# obviously corrupt records.
DEFAULT_MIN_DEPTH_FT = -500.0
DEFAULT_MAX_DEPTH_FT = 5000.0


def qc_water_levels(
    df: pd.DataFrame,
    *,
    min_depth_ft: float = DEFAULT_MIN_DEPTH_FT,
    max_depth_ft: float = DEFAULT_MAX_DEPTH_FT,
) -> pd.DataFrame:
    """Flag physically implausible depth-to-water values.

    Adds a boolean ``qc_flag`` column; rows are not removed so that downstream
    code can decide whether to drop or repair them.
    """
    df = df.copy()
    df["qc_flag"] = ~df["value"].between(min_depth_ft, max_depth_ft)
    return df


def to_daily(
    df: pd.DataFrame,
    *,
    agg: str = "mean",
    value_col: str = "value",
) -> pd.DataFrame:
    """Aggregate observations to one value per site, parameter, and date."""
    required = {"site_no", "parameter_cd", "datetime", value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df.empty:
        return df.copy()

    work = df.copy()
    work["date"] = pd.to_datetime(work["datetime"]).dt.normalize()
    daily = (
        work.groupby(["site_no", "parameter_cd", "date"], as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: "value"})
    )
    return daily.sort_values(["site_no", "parameter_cd", "date"]).reset_index(
        drop=True
    )


def coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize record length and observation density for each series."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "site_no",
                "parameter_cd",
                "n_obs",
                "start",
                "end",
                "span_days",
                "coverage",
                "mean_value",
            ]
        )

    work = df.copy()
    work["date"] = pd.to_datetime(work["datetime"]).dt.normalize()
    summary = (
        work.groupby(["site_no", "parameter_cd"], as_index=False)
        .agg(
            n_obs=("value", "size"),
            start=("date", "min"),
            end=("date", "max"),
            mean_value=("value", "mean"),
        )
    )
    summary["span_days"] = (summary["end"] - summary["start"]).dt.days + 1
    summary["coverage"] = summary["n_obs"] / summary["span_days"].clip(lower=1)
    return summary.sort_values(["site_no", "parameter_cd"]).reset_index(drop=True)


def add_missing_mask(
    daily: pd.DataFrame,
    *,
    site_col: str = "site_no",
    date_col: str = "date",
    value_col: str = "value",
) -> pd.DataFrame:
    """Reindex each series to a complete daily calendar and add a mask.

    The output has one row per site per day, with ``observed`` telling whether
    the original value was present and ``value`` left as NaN on missing days.
    """
    if daily.empty:
        return daily.copy()

    frames: list[pd.DataFrame] = []
    work = daily.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.normalize()

    for site_no, group in work.groupby(site_col, sort=True):
        full_index = pd.date_range(group[date_col].min(), group[date_col].max(), freq="D")
        reindexed = (
            group.set_index(date_col)
            .reindex(full_index)
            .rename_axis(date_col)
            .reset_index()
        )
        reindexed[site_col] = site_no
        reindexed["observed"] = reindexed[value_col].notna()
        frames.append(reindexed)

    return pd.concat(frames, ignore_index=True)


def missing_rate(series: pd.Series) -> float:
    """Fraction of missing values in a series."""
    if len(series) == 0:
        return float("nan")
    return float(series.isna().mean())


def fill_short_gaps(
    series: pd.Series, *, max_gap_days: int = 7, method: str = "linear"
) -> pd.Series:
    """Interpolate short gaps only, leaving long gaps as NaN.

    This is useful for benchmark preprocessing: short sensor outages can be
    repaired, but long gaps should stay missing so models are evaluated on
    them honestly.
    """
    series = series.copy()
    if series.empty or max_gap_days <= 0:
        return series

    limit = max_gap_days if method != "linear" else None
    filled = series.interpolate(method=method, limit=limit, limit_area="inside")

    # Undo interpolation for gaps longer than max_gap_days.
    is_na = series.isna()
    if is_na.any():
        groups = (is_na != is_na.shift()).cumsum()
        gap_lengths = is_na.groupby(groups).transform("sum")
        long_gap = is_na & (gap_lengths > max_gap_days)
        filled[long_gap] = np.nan

    return filled
