#!/usr/bin/env python
"""Aggregate repeated missing-input stress tests across mask seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--out", default="data/processed/missing_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for horizon in [int(h) for h in args.horizons.split(",") if h.strip()]:
        paths = sorted(
            (PROJECT_ROOT / "data/processed").glob(f"missing_h{horizon}_mask*.csv")
        )
        if not paths:
            print(f"no files for horizon {horizon}")
            continue
        data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        grouped = data.groupby("mask_fraction").agg(
            n_seeds=("y_pred_nse_mean", "size"),
            nse_mean=("y_pred_nse_mean", "mean"),
            nse_mean_std=("y_pred_nse_mean", "std"),
            nse_median=("y_pred_nse_median", "mean"),
            nse_median_std=("y_pred_nse_median", "std"),
            kge_mean=("y_pred_kge_mean", "mean"),
            kge_mean_std=("y_pred_kge_mean", "std"),
            kge_median=("y_pred_kge_median", "mean"),
            kge_median_std=("y_pred_kge_median", "std"),
            persistence_nse_mean=("persistence_nse_mean", "mean"),
            persistence_kge_mean=("persistence_kge_mean", "mean"),
        )
        grouped = grouped.reset_index()
        grouped.insert(0, "horizon", horizon)
        grouped["n_files"] = len(paths)
        rows.append(grouped)
        print(f"\nhorizon {horizon} ({len(paths)} mask seeds)")
        print(grouped.round(4).to_string(index=False))

    summary = pd.concat(rows, ignore_index=True)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
