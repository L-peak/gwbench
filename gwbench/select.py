"""Utilities for selecting benchmark wells from an NWIS series catalog."""

from __future__ import annotations

import pandas as pd


def prepare_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Normalize the columns returned by the NWIS site/SeriesCatalog service."""
    df = catalog.copy()
    if "count_nu" in df.columns:
        df["count_nu"] = pd.to_numeric(df["count_nu"], errors="coerce")
    for column in ("begin_date", "end_date"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    if "span_days" not in df.columns and {"begin_date", "end_date"}.issubset(df.columns):
        df["span_days"] = (df["end_date"] - df["begin_date"]).dt.days
    return df


def select_long_records(
    catalog: pd.DataFrame,
    *,
    parameter_cd: str = "72019",
    statistic_id: str | None = None,
    min_obs: int = 500,
    min_span_days: int = 3650,
    min_last_date: str | None = None,
    per_state: int | None = 10,
    limit: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Select wells with long, reasonably dense groundwater-level records.

    Parameters
    ----------
    catalog:
        Output of :func:`gwbench.usgs.fetch_gw_sites` (which returns one row
        per site/parameter because the NWIS site service is queried with
        ``outputDataTypeCd``).
    parameter_cd:
        NWIS parameter code.  72019 is depth to water below land surface.
    statistic_id:
        Optional daily-value statistic code (e.g. ``"00001"`` for maximum).
    min_obs:
        Minimum number of observations in the catalog.
    min_span_days:
        Minimum first-to-last observation span.
    min_last_date:
        If set, require the catalog ``end_date`` to be on or after this date.
        This removes wells whose records stop decades ago.
    per_state:
        If set, keep at most this many wells per state (highest counts first).
    limit:
        Optional global cap after per-state filtering.
    """
    df = prepare_catalog(catalog)

    if "parm_cd" in df.columns:
        df = df[df["parm_cd"].astype(str) == str(parameter_cd)]
    if statistic_id is not None and "stat_cd" in df.columns:
        df = df[df["stat_cd"].astype(str) == str(statistic_id)]
    if "site_tp_cd" in df.columns:
        df = df[df["site_tp_cd"] == "GW"]
    if "count_nu" in df.columns:
        df = df[df["count_nu"] >= min_obs]
    if "span_days" in df.columns:
        df = df[df["span_days"] >= min_span_days]
    if min_last_date is not None and "end_date" in df.columns:
        df = df[df["end_date"] >= pd.Timestamp(min_last_date)]

    # One row per site; keep the parameter row with the most observations.
    sort_cols = ["count_nu"] if "count_nu" in df.columns else []
    df = df.sort_values(sort_cols, ascending=False).drop_duplicates("site_no")

    if per_state is not None and "state_cd" in df.columns:
        df = df.groupby("state_cd", group_keys=False).head(per_state)

    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)

    if {"state_cd", "count_nu"}.issubset(df.columns):
        df = df.sort_values(["state_cd", "count_nu"], ascending=[True, False])
    return df.reset_index(drop=True)
