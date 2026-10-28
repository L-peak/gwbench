"""Quality-control checks for groundwater-level panels.

Groundwater records often contain abrupt step changes caused by datum changes,
sensor replacement, or a switch between manual and continuous measurements.
These are not hydrological events, and they can dominate forecast errors.  The
functions here flag such jumps without silently "fixing" them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_level_jumps(
    panel: pd.DataFrame,
    *,
    threshold_ft: float = 5.0,
    max_gap_days: int = 2,
    value_col: str = "value",
) -> pd.DataFrame:
    """Flag large day-to-day changes that occur without a long data gap.

    A jump is only flagged when the two observations are at most
    ``max_gap_days`` apart.  This avoids confusing real seasonal change across
    a multi-month gap with an instrument/datum step.
    """
    panel = panel.sort_values(["site_no", "date"]).copy()
    grouped = panel.groupby("site_no", sort=False)[value_col]
    panel["value_diff"] = grouped.diff()
    panel["gap_days"] = (
        panel.groupby("site_no", sort=False)["date"].diff().dt.days
    )
    panel["jump_flag"] = (
        (panel["gap_days"] <= max_gap_days)
        & (panel["value_diff"].abs() > threshold_ft)
    )
    return panel


def qc_summary(
    panel: pd.DataFrame,
    *,
    threshold_ft: float = 5.0,
    max_gap_days: int = 2,
    value_col: str = "value",
) -> pd.DataFrame:
    """Summarize jump count, gap structure, and value range per site."""
    flagged = detect_level_jumps(
        panel,
        threshold_ft=threshold_ft,
        max_gap_days=max_gap_days,
        value_col=value_col,
    )
    rows = []
    for site, group in flagged.groupby("site_no", sort=True):
        observed = group[group["observed"]]
        gaps = group["date"].diff().dt.days.dropna()
        rows.append(
            {
                "site_no": site,
                "n_days": len(group),
                "n_observed": int(group["observed"].sum()),
                "coverage": float(group["observed"].mean()),
                "n_jumps": int(group["jump_flag"].sum()),
                "max_abs_jump": float(group["value_diff"].abs().max()),
                "median_abs_diff": float(group["value_diff"].abs().median()),
                "n_gaps_gt_7d": int((gaps > 7).sum()),
                "longest_gap_days": int(gaps.max()) if len(gaps) else 0,
                "value_min": float(observed[value_col].min()),
                "value_max": float(observed[value_col].max()),
                "value_std": float(observed[value_col].std()),
            }
        )
    return pd.DataFrame(rows).sort_values("n_jumps", ascending=False).reset_index(drop=True)


def flag_flat_periods(
    panel: pd.DataFrame,
    *,
    min_run: int = 10,
    value_col: str = "value",
) -> pd.DataFrame:
    """Flag runs of identical consecutive values (possible stuck sensors)."""
    panel = panel.sort_values(["site_no", "date"]).copy()
    panel["same_as_previous"] = panel.groupby("site_no", sort=False)[
        value_col
    ].diff().eq(0)
    # Run length within each site.
    run_id = (
        panel.groupby("site_no", sort=False)["same_as_previous"]
        .transform(lambda s: (~s).cumsum())
    )
    run_length = panel.groupby(["site_no", run_id])["same_as_previous"].transform(
        "size"
    )
    panel["flat_flag"] = panel["same_as_previous"] & (run_length >= min_run)
    return panel
