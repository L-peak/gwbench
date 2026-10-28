#!/usr/bin/env python
"""Download NWIS groundwater site metadata for one or more states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from gwbench.usgs import fetch_gw_sites  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        default="CA",
        help="Comma-separated two-letter state codes, e.g. CA,TX,FL",
    )
    parser.add_argument(
        "--out",
        default="data/raw/sites.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--data-type",
        choices=["gw", "dv"],
        default="gw",
        help="gw = discrete groundwater levels; dv = daily values",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
    if not states:
        raise SystemExit("No state codes provided")

    sites = fetch_gw_sites(
        states,
        has_data_type=args.data_type,
        output_data_type=args.data_type,
    )
    if sites.empty:
        raise SystemExit("No sites returned")

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sites.to_parquet(out_path, index=False)

    print(f"Saved {len(sites):,} sites to {out_path}")
    display_cols = [
        "state_cd",
        "site_no",
        "station_nm",
        "dec_lat_va",
        "dec_long_va",
        "alt_va",
        "huc_cd",
    ]
    display_cols = [c for c in display_cols if c in sites.columns]
    print(sites[display_cols].head(10).to_string(index=False))
    print("\nSites per state:")
    print(sites["state_cd"].value_counts().to_string())


if __name__ == "__main__":
    main()
