#!/usr/bin/env python
"""Fusion ablation from saved test predictions: persistence, raw, fixed, per-site."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gwbench.windows import evaluate_prediction_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="dist/kaggle_results_seeds")
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / args.results_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for horizon in horizons:
        rows = []
        for seed in seeds:
            frame = pd.read_parquet(
                results_dir / f"gru_h{horizon}_seed{seed}" / "test_predictions.parquet",
                columns=["site_no", "y_true", "y_pred", "persistence", "alpha",
                         "y_pred_calibrated"],
            )
            alpha_file = pd.read_csv(
                results_dir / f"gru_h{horizon}_seed{seed}" / "persistence_blend_alpha.csv",
                dtype={"site_no": str},
            )
            alpha_median = float(alpha_file["alpha"].median())
            variants = {
                "persistence": frame["persistence"],
                "raw_gru": frame["y_pred"],
                "fixed_alpha": frame["persistence"]
                + alpha_median * (frame["y_pred"] - frame["persistence"]),
                "per_site_alpha": frame["y_pred_calibrated"],
            }
            for strategy, prediction in variants.items():
                work = frame[["site_no", "y_true", "persistence"]].copy()
                work["y_pred"] = prediction
                work["seasonal_naive"] = np.nan
                _, aggregate = evaluate_prediction_frame(work)
                aggregate["strategy"] = strategy
                aggregate["seed"] = seed
                aggregate["fixed_alpha"] = alpha_median
                rows.append(aggregate)
        data = pd.DataFrame(rows)
        out = out_dir / f"fusion_ablation_h{horizon}{args.tag}.csv"
        data.to_csv(out, index=False)
        summary = data.groupby("strategy").agg(
            nse_mean=("y_pred_nse_mean", "mean"),
            nse_mean_std=("y_pred_nse_mean", "std"),
            nse_median=("y_pred_nse_median", "mean"),
            nse_median_std=("y_pred_nse_median", "std"),
            kge_mean=("y_pred_kge_mean", "mean"),
            kge_mean_std=("y_pred_kge_mean", "std"),
            kge_median=("y_pred_kge_median", "mean"),
            kge_median_std=("y_pred_kge_median", "std"),
            fixed_alpha=("fixed_alpha", "mean"),
        )
        print(f"\nhorizon {horizon}")
        print(summary.round(4).to_string())
        print(f"saved {out}")


if __name__ == "__main__":
    main()
