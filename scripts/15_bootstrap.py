#!/usr/bin/env python
"""Paired site-level bootstrap for the calibrated GRU versus persistence.

The bootstrap resamples wells (the independent unit), not forecast days, so
the confidence intervals reflect cross-well heterogeneity rather than
autocorrelation within a well. Metrics are first averaged over training seeds
1--3 for each well, then resampled 10,000 times.
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
    parser.add_argument("--results-dir", default="dist/kaggle_results_seeds")
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--outlier", default="413156098591201")
    parser.add_argument("--out", default="data/processed/bootstrap_summary.csv")
    return parser.parse_args()


def load_seed_metrics(results_dir: Path, horizon: int, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = (
            results_dir
            / f"gru_h{horizon}_seed{seed}"
            / "test_metrics_per_site.csv"
        )
        frame = pd.read_csv(path, dtype={"site_no": str})
        frame["seed"] = seed
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    numeric = [
        "y_pred_nse",
        "y_pred_kge",
        "persistence_nse",
        "persistence_kge",
        "y_pred_rmse",
        "persistence_rmse",
    ]
    averaged = data.groupby("site_no", as_index=False)[numeric].mean()
    return averaged


def bootstrap_ci(
    values: np.ndarray, *, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float, float, float, float]:
    n = len(values)
    means = np.empty(n_boot)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        means[i] = sample.mean()
        medians[i] = np.median(sample)
    return (
        float(values.mean()),
        float(np.median(values)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / args.results_dir
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    for horizon in horizons:
        data = load_seed_metrics(results_dir, horizon, seeds)
        for metric in ("nse", "kge"):
            delta = (data[f"y_pred_{metric}"] - data[f"persistence_{metric}"]).to_numpy()
            mean, median, mean_lo, mean_hi, med_lo, med_hi = bootstrap_ci(
                delta, n_boot=args.n_boot, rng=rng
            )
            row = {
                "horizon": horizon,
                "metric": metric.upper(),
                "n_sites": len(data),
                "n_seeds": len(seeds),
                "delta_mean": mean,
                "delta_mean_ci_lo": mean_lo,
                "delta_mean_ci_hi": mean_hi,
                "delta_median": median,
                "delta_median_ci_lo": med_lo,
                "delta_median_ci_hi": med_hi,
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                "model_mean": float(data[f"y_pred_{metric}"].mean()),
                "model_median": float(data[f"y_pred_{metric}"].median()),
                "persistence_mean": float(data[f"persistence_{metric}"].mean()),
                "persistence_median": float(data[f"persistence_{metric}"].median()),
            }
            mask = data["site_no"].astype(str) != args.outlier
            if mask.any() and (~mask).any():
                row["delta_mean_excl_outlier"] = float(delta[mask].mean())
                row["model_mean_excl_outlier"] = float(
                    data.loc[mask, f"y_pred_{metric}"].mean()
                )
                row["persistence_mean_excl_outlier"] = float(
                    data.loc[mask, f"persistence_{metric}"].mean()
                )
            rows.append(row)
            print(
                f"h={horizon} {metric.upper()} n={len(data)}: "
                f"mean={mean:+.4f} [{mean_lo:+.4f}, {mean_hi:+.4f}] "
                f"median={median:+.4f} [{med_lo:+.4f}, {med_hi:+.4f}] "
                f"W/L/T={row['wins']}/{row['losses']}/{row['ties']}"
            )

    summary = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
