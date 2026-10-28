#!/usr/bin/env python
"""Build a complete daily panel from raw groundwater observations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from gwbench.panel import build_daily_panel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/levels_daily_pilot.parquet")
    parser.add_argument("--parameter-cd", default="72019")
    parser.add_argument("--statistic-id", default="00001")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--min-value",
        type=float,
        default=-500.0,
        help="Drop observations below this depth/altitude (sentinel values)",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=5000.0,
        help="Drop observations above this value (sentinels such as 100000)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Optionally keep only sites with at least this observed fraction",
    )
    parser.add_argument(
        "--catalog",
        default="data/raw/daily_catalog_multi.parquet",
        help="Optional site catalog used to attach state/station metadata",
    )
    parser.add_argument("--out", default="data/processed/panel_pilot.parquet")
    parser.add_argument("--summary", default="data/processed/panel_pilot_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = PROJECT_ROOT / args.raw
    if not raw_path.exists():
        raise SystemExit(f"Raw file not found: {raw_path}; run 02_fetch_levels.py first")

    raw = pd.read_parquet(raw_path)
    if "parameter_cd" in raw.columns and args.parameter_cd:
        raw = raw[raw["parameter_cd"].astype(str) == str(args.parameter_cd)]
    if "statistic_id" in raw.columns and args.statistic_id:
        raw = raw[raw["statistic_id"].astype(str) == str(args.statistic_id)]
    if raw.empty:
        raise SystemExit("No rows left after parameter/statistic filtering")

    before = len(raw)
    raw = raw[raw["value"].between(args.min_value, args.max_value)]
    print(
        f"Dropped {before - len(raw):,} rows outside "
        f"[{args.min_value:g}, {args.max_value:g}]"
    )
    if raw.empty:
        raise SystemExit("No rows left after value-range filtering")

    panel = build_daily_panel(raw, start=args.start, end=args.end)

    if args.min_coverage > 0:
        coverage = panel.groupby("site_no")["observed"].mean()
        keep = coverage[coverage >= args.min_coverage].index
        panel = panel[panel["site_no"].isin(keep)].copy()
        print(
            f"Kept {panel['site_no'].nunique()} sites with coverage >= "
            f"{args.min_coverage:.2f}"
        )

    catalog_path = PROJECT_ROOT / args.catalog
    if catalog_path.exists():
        catalog = pd.read_parquet(catalog_path)
        meta_cols = [
            column
            for column in (
                "site_no",
                "state_cd",
                "station_nm",
                "dec_lat_va",
                "dec_long_va",
                "huc_cd",
            )
            if column in catalog.columns
        ]
        meta = catalog[meta_cols].drop_duplicates("site_no")
        panel = panel.merge(meta, on="site_no", how="left")
        print(f"Merged site metadata from {catalog_path}")

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    summary = (
        panel.groupby("site_no", as_index=False)
        .agg(
            n_days=("date", "size"),
            n_observed=("observed", "sum"),
            start=("date", "min"),
            end=("date", "max"),
            mean_value=("value", "mean"),
            min_value=("value", "min"),
            max_value=("value", "max"),
        )
    )
    summary["coverage"] = summary["n_observed"] / summary["n_days"]
    summary_path = PROJECT_ROOT / args.summary
    summary.to_csv(summary_path, index=False)
    summary.to_parquet(summary_path.with_suffix(".parquet"), index=False)

    print(f"Saved panel: {len(panel):,} rows, {panel['site_no'].nunique()} sites -> {out_path}")
    print(f"Saved summary -> {summary_path}")
    display = summary[
        ["site_no", "n_observed", "coverage", "mean_value", "min_value", "max_value"]
    ].sort_values("coverage", ascending=False)
    print(display.head(20).to_string(index=False))
    if len(display) > 20:
        print(f"... showing 20 of {len(display)} sites")
    print("\nCoverage quantiles:")
    print(summary["coverage"].describe().to_string())


if __name__ == "__main__":
    main()
