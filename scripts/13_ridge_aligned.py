#!/usr/bin/env python
"""Per-site ridge baseline evaluated on exactly the same anchors as the GRU."""

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
from sklearn.linear_model import Ridge  # noqa: E402

from gwbench.windows import (  # noqa: E402
    build_anchors,
    evaluate_prediction_frame,
    panel_to_arrays,
    split_anchors_temporal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_core_clean.parquet")
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--lookback", type=int, default=365)
    parser.add_argument("--min-history-obs", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--out", default="data/processed/ridge_aligned.csv")
    return parser.parse_args()


def anchor_features(arrays, site: int, anchor: int) -> tuple[list[float], float]:
    start = anchor - 365 + 1
    values = arrays.values[site, start : anchor + 1]
    observed = arrays.observed[site, start : anchor + 1]
    observed_values = values[observed]
    last_obs = float(observed_values[-1]) if observed_values.size else np.nan

    features = [last_obs]
    for window in (7, 30, 90, 365):
        segment_values = values[-window:]
        segment_mask = observed[-window:]
        segment = segment_values[segment_mask]
        features.append(float(segment.mean()) if segment.size else last_obs)
        features.append(float(segment.std()) if segment.size > 1 else 0.0)

    day_of_year = arrays.dates[anchor].dayofyear
    angle = 2.0 * np.pi * day_of_year / 365.25
    features.extend([float(np.sin(angle)), float(np.cos(angle))])
    return features, last_obs


def main() -> None:
    args = parse_args()
    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    arrays = panel_to_arrays(panel)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    rows: list[dict] = []

    for horizon in horizons:
        anchors = build_anchors(
            arrays,
            lookback=args.lookback,
            horizon=horizon,
            stride=1,
            min_history_obs=args.min_history_obs,
        )
        splits = split_anchors_temporal(
            anchors,
            arrays.dates,
            train_end="2015-12-31",
            val_end="2018-12-31",
        )
        train_anchors = splits["train"]
        test_anchors = splits["test"]
        records: list[dict] = []

        for site in range(len(arrays.sites)):
            train_site = train_anchors[train_anchors[:, 0] == site]
            test_site = test_anchors[test_anchors[:, 0] == site]
            if len(train_site) < args.min_train or len(test_site) == 0:
                continue

            x_train, y_train = [], []
            for _, anchor in train_site:
                features, _ = anchor_features(arrays, site, anchor)
                target = arrays.values[site, anchor + horizon]
                if np.isfinite(target) and all(np.isfinite(features)):
                    x_train.append(features)
                    y_train.append(float(target))
            if len(x_train) < args.min_train:
                continue

            x_train = np.asarray(x_train, dtype=float)
            y_train = np.asarray(y_train, dtype=float)
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            std[std < 1e-8] = 1.0
            model = Ridge(alpha=args.alpha)
            model.fit((x_train - mean) / std, y_train)

            for _, anchor in test_site:
                features, last_obs = anchor_features(arrays, site, anchor)
                target = float(arrays.values[site, anchor + horizon])
                if not np.isfinite(target) or not all(np.isfinite(features)):
                    continue
                x_test = ((np.asarray(features) - mean) / std).reshape(1, -1)
                prediction = float(model.predict(x_test)[0])
                seasonal_index = anchor + horizon - 365
                seasonal = (
                    float(arrays.values[site, seasonal_index])
                    if seasonal_index >= 0 and arrays.observed[site, seasonal_index]
                    else np.nan
                )
                records.append(
                    {
                        "site_no": arrays.sites[site],
                        "date": arrays.dates[anchor + horizon],
                        "y_true": target,
                        "y_pred": prediction,
                        "persistence": last_obs,
                        "seasonal_naive": seasonal,
                    }
                )

        frame = pd.DataFrame(records)
        per_site, aggregate = evaluate_prediction_frame(frame)
        aggregate["horizon"] = horizon
        rows.append(aggregate)
        print(
            f"horizon {horizon}: n={aggregate.get('n_sites')} "
            f"ridge NSE mean/median={aggregate.get('y_pred_nse_mean'):.4f}/"
            f"{aggregate.get('y_pred_nse_median'):.4f} "
            f"KGE mean/median={aggregate.get('y_pred_kge_mean'):.4f}/"
            f"{aggregate.get('y_pred_kge_median'):.4f}"
        )
        per_site.insert(0, "horizon", horizon)
        per_site.to_csv(
            PROJECT_ROOT / f"data/processed/ridge_aligned_h{horizon}_per_site.csv",
            index=False,
        )

    summary = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"saved {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
