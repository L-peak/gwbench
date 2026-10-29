#!/usr/bin/env python
"""Persistence and seasonal-naive skill on the common anchor protocol.

This implementation walks one well at a time instead of pivoting the full
panel, so the 279-well panel can be evaluated in bounded memory. It is used
for the cross-panel sensitivity analysis (full / core / clean core).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gwbench.metrics import kge, nse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_core_clean.parquet")
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--lookback", type=int, default=365)
    parser.add_argument("--min-history-obs", type=int, default=30)
    parser.add_argument("--test-start", default="2019-01-01")
    parser.add_argument("--out", default="data/processed/panel_sensitivity.csv")
    return parser.parse_args()


def site_skill(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    observed: np.ndarray,
    *,
    horizon: int,
    lookback: int,
    min_history_obs: int,
    test_start: pd.Timestamp,
) -> dict | None:
    start = max(lookback - 1, int(np.searchsorted(dates, test_start)))
    stop = len(dates) - horizon
    y_true: list[float] = []
    persistence: list[float] = []
    seasonal: list[float] = []

    for anchor in range(start, stop):
        target_index = anchor + horizon
        if not observed[target_index]:
            continue
        history = observed[anchor - lookback + 1 : anchor + 1]
        if history.sum() < min_history_obs:
            continue
        y_true.append(float(values[target_index]))
        persistence.append(float(values[anchor - lookback + 1 : anchor + 1][history][-1]))
        seasonal_index = target_index - 365
        if seasonal_index >= 0 and observed[seasonal_index]:
            seasonal.append(float(values[seasonal_index]))
        else:
            seasonal.append(np.nan)

    if len(y_true) < 30:
        return None

    true = np.asarray(y_true, dtype=float)
    pers = np.asarray(persistence, dtype=float)
    seas = np.asarray(seasonal, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pers)
    row = {
        "n": int(mask.sum()),
        "persistence_nse": nse(true[mask], pers[mask]),
        "persistence_kge": kge(true[mask], pers[mask]),
    }
    mask = np.isfinite(true) & np.isfinite(seas)
    row["seasonal_nse"] = nse(true[mask], seas[mask]) if mask.sum() >= 30 else np.nan
    row["seasonal_kge"] = kge(true[mask], seas[mask]) if mask.sum() >= 30 else np.nan
    return row


def main() -> None:
    args = parse_args()
    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    dates = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    test_start = pd.Timestamp(args.test_start)
    rows: list[dict] = []

    for horizon in horizons:
        per_site: list[dict] = []
        for site, group in panel.groupby("site_no", sort=True):
            group = group.set_index("date").sort_index()
            values = group["value"].reindex(dates).to_numpy(dtype=float)
            observed = (
                group["observed"].reindex(dates).fillna(False).to_numpy(dtype=bool)
            )
            skill = site_skill(
                dates,
                values,
                observed,
                horizon=horizon,
                lookback=args.lookback,
                min_history_obs=args.min_history_obs,
                test_start=test_start,
            )
            if skill is not None:
                skill["site_no"] = str(site)
                per_site.append(skill)

        frame = pd.DataFrame(per_site)
        row = {
            "panel": Path(args.panel).stem,
            "horizon": horizon,
            "n_sites": int(len(frame)),
            "persistence_nse_mean": float(frame["persistence_nse"].mean()),
            "persistence_nse_median": float(frame["persistence_nse"].median()),
            "persistence_kge_mean": float(frame["persistence_kge"].mean()),
            "persistence_kge_median": float(frame["persistence_kge"].median()),
            "seasonal_nse_mean": float(frame["seasonal_nse"].mean()),
            "seasonal_nse_median": float(frame["seasonal_nse"].median()),
        }
        rows.append(row)
        print(
            f"{row['panel']} h={horizon}: n={row['n_sites']} "
            f"persistence NSE mean/median="
            f"{row['persistence_nse_mean']:.4f}/{row['persistence_nse_median']:.4f} "
            f"KGE mean/median="
            f"{row['persistence_kge_mean']:.4f}/{row['persistence_kge_median']:.4f}"
        )

    summary = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        old = pd.read_csv(out_path)
        summary = pd.concat([old, summary], ignore_index=True)
        summary = summary.drop_duplicates(subset=["panel", "horizon"], keep="last")
    summary.to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
