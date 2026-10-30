#!/usr/bin/env python
"""Hierarchical bootstrap over training seeds and wells.

Sensitivity analysis for the seed-conditional intervals in the main text.
With only three seeds this is necessarily coarse, so it is reported as an
appendix check rather than as the primary inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="dist/kaggle_results_purged")
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20261029)
    parser.add_argument("--out", default="data/processed/hierarchical_bootstrap.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / args.results_dir
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rng = np.random.default_rng(args.rng_seed)
    rows = []

    for horizon in horizons:
        deltas = {}
        for seed in seeds:
            frame = pd.read_csv(
                results_dir / f"gru_h{horizon}_seed{seed}" /
                "test_metrics_per_site.csv",
                dtype={"site_no": str},
            )
            deltas[seed] = (
                frame["y_pred_nse"].to_numpy()
                - frame["persistence_nse"].to_numpy()
            )
        n_sites = len(next(iter(deltas.values())))
        means = np.empty(args.n_boot)
        medians = np.empty(args.n_boot)
        for i in range(args.n_boot):
            chosen_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            pooled = []
            for seed in chosen_seeds:
                values = deltas[int(seed)]
                pooled.append(values[rng.integers(0, len(values), len(values))])
            sample = np.concatenate(pooled)
            means[i] = sample.mean()
            medians[i] = np.median(sample)
        rows.append(
            {
                "horizon": horizon,
                "n_seeds": len(seeds),
                "n_sites": n_sites,
                "mean_ci_lo": float(np.percentile(means, 2.5)),
                "mean_ci_hi": float(np.percentile(means, 97.5)),
                "median_ci_lo": float(np.percentile(medians, 2.5)),
                "median_ci_hi": float(np.percentile(medians, 97.5)),
                "mean_point": float(
                    np.mean([d.mean() for d in deltas.values()])
                ),
                "median_point": float(
                    np.median(np.concatenate(list(deltas.values())))
                ),
            }
        )
        print(
            f"h={horizon}: mean {rows[-1]['mean_point']:+.4f} "
            f"[{rows[-1]['mean_ci_lo']:+.4f}, {rows[-1]['mean_ci_hi']:+.4f}] "
            f"median {rows[-1]['median_point']:+.4f} "
            f"[{rows[-1]['median_ci_lo']:+.4f}, {rows[-1]['median_ci_hi']:+.4f}]"
        )

    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
