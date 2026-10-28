#!/usr/bin/env python
"""Evaluate persistence and seasonal-naive skill as the forecast horizon grows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from gwbench.panel import evaluate_predictions, temporal_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_pilot.parquet")
    parser.add_argument("--train-end", default="2015-12-31")
    parser.add_argument("--val-end", default="2018-12-31")
    parser.add_argument("--horizons", default="1,7,30,90,180")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="data/processed/horizon_baselines.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    panel = temporal_split(
        panel, train_end=args.train_end, val_end=args.val_end
    ).sort_values(["site_no", "date"])

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    rows: list[dict] = []

    grouped = panel.groupby("site_no", sort=False)["value"]
    panel["seasonal_naive"] = grouped.transform(lambda s: s.shift(365))
    _, seasonal_agg = evaluate_predictions(
        panel, "seasonal_naive", split=args.split
    )

    for horizon in horizons:
        column = f"persistence_h{horizon}"
        panel[column] = grouped.transform(lambda s, h=horizon: s.shift(h).ffill())
        per_site, aggregate = evaluate_predictions(
            panel, column, split=args.split
        )
        row = {
            "horizon_days": horizon,
            "n_sites": aggregate.get("n_sites", 0),
            "persistence_rmse_mean": aggregate.get("rmse_mean"),
            "persistence_mae_mean": aggregate.get("mae_mean"),
            "persistence_nse_mean": aggregate.get("nse_mean"),
            "persistence_nse_median": aggregate.get("nse_median"),
            "persistence_kge_mean": aggregate.get("kge_mean"),
            "persistence_kge_median": aggregate.get("kge_median"),
            "seasonal_naive_nse_mean": seasonal_agg.get("nse_mean"),
            "seasonal_naive_nse_median": seasonal_agg.get("nse_median"),
        }
        rows.append(row)
        print(pd.DataFrame([row]).to_string(index=False))

    summary = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
