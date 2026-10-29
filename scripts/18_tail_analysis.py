#!/usr/bin/env python
"""Per-site tail diagnostics for persistence, ridge, raw GRU, and calibrated GRU."""

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
    parser.add_argument("--ridge-dir", default="data/processed")
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--out", default="data/processed/tail_analysis.csv")
    return parser.parse_args()


def summarize(values: pd.Series, horizon: int, model: str) -> dict:
    values = values.dropna().to_numpy()
    return {
        "horizon": horizon,
        "model": model,
        "n_sites": int(len(values)),
        "nse_mean": float(np.mean(values)),
        "nse_median": float(np.median(values)),
        "nse_min": float(np.min(values)),
        "n_below_m5": int((values < -5).sum()),
        "n_below_m1": int((values < -1).sum()),
        "n_below_m05": int((values < -0.5).sum()),
    }


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / args.results_dir
    ridge_dir = PROJECT_ROOT / args.ridge_dir
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows: list[dict] = []

    for horizon in horizons:
        calibrated = []
        raw = []
        for seed in seeds:
            base = results_dir / f"gru_h{horizon}_seed{seed}"
            calibrated.append(
                pd.read_csv(base / "test_metrics_per_site_calibrated.csv",
                            dtype={"site_no": str})
            )
            raw.append(
                pd.read_csv(base / "test_metrics_per_site_raw.csv",
                            dtype={"site_no": str})
            )
        cal = pd.concat(calibrated).groupby("site_no", as_index=False)[
            ["y_pred_nse", "persistence_nse"]
        ].mean()
        raw_frame = pd.concat(raw).groupby("site_no", as_index=False)[
            ["y_pred_nse"]
        ].mean()
        ridge_path = ridge_dir / f"ridge_fast_h{horizon}_per_site.csv"
        if not ridge_path.exists():
            ridge_path = ridge_dir / f"ridge_aligned_h{horizon}_per_site.csv"
        ridge = pd.read_csv(ridge_path, dtype={"site_no": str})
        rows.append(summarize(cal["persistence_nse"], horizon, "persistence"))
        rows.append(summarize(cal["y_pred_nse"], horizon, "calibrated_gru"))
        rows.append(summarize(raw_frame["y_pred_nse"], horizon, "raw_gru"))
        ridge_nse = ridge["ridge_nse"] if "ridge_nse" in ridge else ridge["y_pred_nse"]
        rows.append(summarize(ridge_nse, horizon, "ridge"))

    summary = pd.DataFrame(rows)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.round(3).to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
