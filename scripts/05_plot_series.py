#!/usr/bin/env python
"""Plot a few panel series for a quick visual data audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_pilot.parquet")
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("--out", default="data/processed/panel_pilot.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    sites = panel["site_no"].drop_duplicates().head(args.n_sites).tolist()

    n_cols = 2
    n_rows = (len(sites) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharex=False)
    axes = axes.ravel()

    for ax, site in zip(axes, sites):
        group = panel[panel["site_no"] == site].sort_values("date")
        ax.plot(group["date"], group["value"], linewidth=0.8, color="#2563eb")
        missing = group[~group["observed"]]
        ax.scatter(
            missing["date"],
            [group["value"].min()] * len(missing),
            s=1,
            color="#ef4444",
            alpha=0.3,
            label="missing",
        )
        ax.set_title(str(site), fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.2)

    for ax in axes[len(sites) :]:
        ax.axis("off")

    fig.suptitle("Groundwater level series (daily panel)", fontsize=11)
    fig.tight_layout()
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
