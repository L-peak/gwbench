#!/usr/bin/env python
"""Run QC checks on a groundwater panel and save a per-site summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from gwbench.qc import qc_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_pilot.parquet")
    parser.add_argument("--threshold-ft", type=float, default=5.0)
    parser.add_argument("--max-gap-days", type=int, default=2)
    parser.add_argument("--out", default="data/processed/qc_pilot.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    summary = qc_summary(
        panel,
        threshold_ft=args.threshold_ft,
        max_gap_days=args.max_gap_days,
    )
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(f"QC summary for {len(summary)} sites -> {out_path}")
    print(
        summary[
            [
                "site_no",
                "coverage",
                "n_jumps",
                "max_abs_jump",
                "n_gaps_gt_7d",
                "longest_gap_days",
                "value_std",
            ]
        ].to_string(index=False)
    )
    print("\nSites with >5 jumps:")
    print(summary[summary["n_jumps"] > 5]["site_no"].tolist())


if __name__ == "__main__":
    main()
