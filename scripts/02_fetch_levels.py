#!/usr/bin/env python
"""Fetch a pilot sample of USGS groundwater-level records."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from gwbench.select import prepare_catalog, select_long_records  # noqa: E402
from gwbench.usgs import (  # noqa: E402
    fetch_daily_values,
    fetch_daily_values_legacy,
    fetch_field_measurements,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", default="data/raw/sites.parquet")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--parameter-cd", default="72019")
    parser.add_argument(
        "--statistic-id",
        default="00001",
        help="Daily-value statistic code; use an empty string for field catalogs",
    )
    parser.add_argument("--min-obs", type=int, default=500)
    parser.add_argument("--min-span-days", type=int, default=3650)
    parser.add_argument(
        "--min-last-date",
        default="2015-01-01",
        help="Require the catalog record to extend to at least this date",
    )
    parser.add_argument(
        "--per-state",
        type=int,
        default=10,
        help="Keep at most this many wells per state before applying --limit",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/raw/levels_pilot.parquet")
    parser.add_argument("--summary", default="data/raw/levels_pilot_summary.csv")
    parser.add_argument(
        "--cache-dir",
        default="data/raw/cache_daily",
        help="Per-site parquet cache; reruns resume from this directory",
    )
    parser.add_argument(
        "--backend",
        choices=["legacy", "ogc"],
        default="legacy",
        help="Daily values backend; legacy batches many sites per request",
    )
    parser.add_argument("--skip-gwlevels", action="store_true")
    parser.add_argument("--skip-dv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sites_path = PROJECT_ROOT / args.sites
    if not sites_path.exists():
        raise SystemExit(f"Site file not found: {sites_path}; run 01_fetch_sites.py first")

    catalog = prepare_catalog(pd.read_parquet(sites_path))
    sites = select_long_records(
        catalog,
        parameter_cd=args.parameter_cd,
        statistic_id=args.statistic_id or None,
        min_obs=args.min_obs,
        min_span_days=args.min_span_days,
        min_last_date=args.min_last_date or None,
        per_state=args.per_state if args.per_state > 0 else None,
        limit=args.limit if args.limit > 0 else None,
        seed=args.seed,
    )
    if sites.empty:
        raise SystemExit("No sites passed the selection filters")

    site_ids = sites["site_no"].astype(str).tolist()
    print(f"Selected {len(site_ids)} wells")
    if "state_cd" in sites.columns:
        print(sites["state_cd"].value_counts().to_string())
    print(f"Fetching records from {args.start} to {args.end}")

    frames: list[pd.DataFrame] = []
    if not args.skip_gwlevels:
        print("  - discrete field measurements")
        gw = fetch_field_measurements(
            site_ids,
            args.start,
            args.end,
            parameter_cd=args.parameter_cd,
        )
        if not gw.empty:
            gw["source"] = "field_measurements"
            frames.append(gw)
            print(f"    {len(gw):,} observations, {gw['site_no'].nunique()} sites")

    if not args.skip_dv:
        print(
            "  - daily values "
            f"(parameter {args.parameter_cd}, statistic {args.statistic_id or 'all'})"
        )
        if args.backend == "legacy":
            dv = fetch_daily_values_legacy(
                site_ids,
                args.start,
                args.end,
                parameter_cd=args.parameter_cd,
                statistic_id=args.statistic_id or None,
                cache_dir=PROJECT_ROOT / args.cache_dir,
            )
        else:
            dv = fetch_daily_values(
                site_ids,
                args.start,
                args.end,
                parameter_cd=args.parameter_cd,
                statistic_id=args.statistic_id or None,
                cache_dir=PROJECT_ROOT / args.cache_dir,
            )
        if not dv.empty:
            dv["source"] = "daily_values"
            frames.append(dv)
            print(f"    {len(dv):,} observations, {dv['site_no'].nunique()} sites")

    if not frames:
        raise SystemExit("No observations returned")

    combined = pd.concat(frames, ignore_index=True)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    summary = (
        combined.groupby(["site_no", "parameter_cd", "source"], as_index=False)
        .agg(
            n_obs=("value", "size"),
            start=("datetime", "min"),
            end=("datetime", "max"),
            mean_value=("value", "mean"),
        )
        .sort_values(["n_obs"], ascending=False)
    )
    summary_path = PROJECT_ROOT / args.summary
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved {len(combined):,} observations to {out_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Sites with data: {combined['site_no'].nunique()}")
    print("\nTop series by observation count:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
