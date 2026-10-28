#!/usr/bin/env python
"""Drop sites with too many large day-to-day jumps from a panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_core.parquet")
    parser.add_argument("--qc", default="data/processed/qc_core.csv")
    parser.add_argument("--max-jumps", type=int, default=5)
    parser.add_argument("--out", default="data/processed/panel_core_clean.parquet")
    parser.add_argument("--summary", default="data/processed/panel_core_clean_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    qc = pd.read_csv(PROJECT_ROOT / args.qc, dtype={"site_no": str})
    keep = set(qc.loc[qc["n_jumps"] <= args.max_jumps, "site_no"])
    filtered = panel[panel["site_no"].isin(keep)].copy()

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(out_path, index=False)

    summary = (
        filtered.groupby("site_no", as_index=False)
        .agg(
            n_days=("date", "size"),
            n_observed=("observed", "sum"),
            start=("date", "min"),
            end=("date", "max"),
            mean_value=("value", "mean"),
        )
    )
    summary["coverage"] = summary["n_observed"] / summary["n_days"]
    summary_path = PROJECT_ROOT / args.summary
    summary.to_csv(summary_path, index=False)
    summary.to_parquet(summary_path.with_suffix(".parquet"), index=False)

    dropped = sorted(set(panel["site_no"]) - keep)
    print(
        f"Kept {filtered['site_no'].nunique()} of {panel['site_no'].nunique()} sites "
        f"(max_jumps={args.max_jumps})"
    )
    print(f"Dropped: {dropped}")
    print(f"Saved {len(filtered):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
