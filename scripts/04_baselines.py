#!/usr/bin/env python
"""Evaluate persistence and seasonal-naive baselines on a panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from gwbench.panel import (  # noqa: E402
    add_baselines,
    evaluate_predictions,
    temporal_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_pilot.parquet")
    parser.add_argument("--train-end", default="2015-12-31")
    parser.add_argument("--val-end", default="2018-12-31")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel_path = PROJECT_ROOT / args.panel
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}; run 03_build_panel.py first")

    panel = pd.read_parquet(panel_path)
    panel = temporal_split(panel, train_end=args.train_end, val_end=args.val_end)
    panel = add_baselines(panel)

    print(f"Rows per split:\n{panel['split'].value_counts().to_string()}\n")

    all_metrics: list[dict] = []
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for pred_col in ("persistence", "seasonal_naive"):
        per_site, aggregate = evaluate_predictions(
            panel, pred_col, split=args.split
        )
        all_metrics.append(aggregate)
        print(f"--- {pred_col} ({args.split}) ---")
        print(json.dumps(aggregate, indent=2, default=float))
        if not per_site.empty:
            per_site.to_csv(out_dir / f"baseline_{pred_col}_{args.split}_per_site.csv", index=False)
            print(per_site.sort_values("nse", ascending=False).to_string(index=False))
        print()

    pd.DataFrame(all_metrics).to_csv(
        out_dir / f"baseline_summary_{args.split}.csv", index=False
    )
    print(f"Saved summaries -> {out_dir}")


if __name__ == "__main__":
    main()
