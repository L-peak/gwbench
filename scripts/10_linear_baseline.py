#!/usr/bin/env python
"""Per-site ridge-regression baseline at multiple forecast horizons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from gwbench.panel import temporal_split  # noqa: E402
from gwbench.windows import evaluate_prediction_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_core_clean.parquet")
    parser.add_argument("--train-end", default="2015-12-31")
    parser.add_argument("--val-end", default="2018-12-31")
    parser.add_argument("--horizons", default="7,30,90")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument(
        "--clip-std",
        type=float,
        default=3.0,
        help="Clip predictions to train min/max plus this many std; 0 disables",
    )
    parser.add_argument(
        "--drop-trend",
        action="store_true",
        help="Drop the linear time-index feature to reduce extrapolation",
    )
    parser.add_argument("--out-dir", default="data/processed/linear")
    return parser.parse_args()


def feature_frame(group: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Features known at the anchor date, plus the target at anchor + horizon."""
    group = group.sort_values("date").copy()
    value = group["value"]
    shifted = value.shift(horizon)

    group["last_obs"] = shifted.ffill()
    for window in (7, 30, 90, 365):
        group[f"mean_{window}"] = shifted.rolling(
            window, min_periods=max(3, window // 5)
        ).mean()
    group["std_30"] = shifted.rolling(30, min_periods=5).std()
    group["std_365"] = shifted.rolling(365, min_periods=30).std()
    group["lag_365"] = value.shift(365)

    day_of_year = group["date"].dt.dayofyear
    group["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    group["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    group["t_index"] = (group["date"] - pd.Timestamp("2000-01-01")).dt.days
    group["target"] = value
    return group


def main() -> None:
    args = parse_args()
    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    panel = temporal_split(
        panel, train_end=args.train_end, val_end=args.val_end
    )

    feature_cols = [
        "last_obs",
        "mean_7",
        "mean_30",
        "mean_90",
        "mean_365",
        "std_30",
        "std_365",
        "lag_365",
        "doy_sin",
        "doy_cos",
        "t_index",
    ]
    if args.drop_trend:
        feature_cols.remove("t_index")
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for horizon in horizons:
        frames: list[pd.DataFrame] = []
        for site, group in panel.groupby("site_no", sort=True):
            features = feature_frame(group, horizon)
            train = features[
                (features["split"] == "train")
                & features["target"].notna()
                & features[feature_cols].notna().all(axis=1)
            ]
            test = features[
                (features["split"] == "test")
                & features["target"].notna()
                & features[feature_cols].notna().all(axis=1)
            ]
            if len(train) < args.min_train or test.empty:
                continue

            x_train = train[feature_cols].to_numpy(dtype=float)
            x_test = test[feature_cols].to_numpy(dtype=float)
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            std[std < 1e-8] = 1.0
            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std

            model = Ridge(alpha=args.alpha)
            model.fit(x_train, train["target"].to_numpy(dtype=float))
            prediction = model.predict(x_test)
            if args.clip_std > 0:
                target_std = float(train["target"].std())
                prediction = np.clip(
                    prediction,
                    float(train["target"].min()) - args.clip_std * target_std,
                    float(train["target"].max()) + args.clip_std * target_std,
                )

            frames.append(
                pd.DataFrame(
                    {
                        "site_no": site,
                        "date": test["date"].to_numpy(),
                        "y_true": test["target"].to_numpy(dtype=float),
                        "y_pred": prediction,
                        "persistence": test["last_obs"].to_numpy(dtype=float),
                        "seasonal_naive": test["lag_365"].to_numpy(dtype=float),
                    }
                )
            )

        if not frames:
            print(f"horizon {horizon}: no sites passed")
            continue
        frame = pd.concat(frames, ignore_index=True)
        per_site, aggregate = evaluate_prediction_frame(frame)
        per_site.insert(0, "horizon_days", horizon)
        per_site.to_csv(out_dir / f"linear_h{horizon}_per_site.csv", index=False)
        aggregate["horizon_days"] = horizon
        all_rows.append(aggregate)
        print(f"\n=== horizon {horizon} ===")
        print(pd.DataFrame([aggregate]).to_string(index=False))

    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary.to_csv(out_dir / "linear_summary.csv", index=False)
        print(f"\nSaved summary -> {out_dir / 'linear_summary.csv'}")


if __name__ == "__main__":
    main()
