#!/usr/bin/env python
"""Vectorized per-site ridge baseline on the common anchor protocol.

The feature matrix is built with pandas rolling windows for one well at a
time, avoiding the Python loop over anchors in the original implementation.
Used for the 279-well panel, where the loop is prohibitively slow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from gwbench.metrics import kge, nse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_full.parquet")
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--lookback", type=int, default=365)
    parser.add_argument("--min-history-obs", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--train-end", default="2015-12-31")
    parser.add_argument("--test-start", default="2019-01-01")
    parser.add_argument("--out", default="data/processed/ridge_aligned_full.csv")
    return parser.parse_args()


def site_frame(dates: pd.DatetimeIndex, group: pd.DataFrame) -> pd.DataFrame:
    group = group.set_index("date").sort_index()
    values = group["value"].reindex(dates)
    observed = group["observed"].reindex(dates).fillna(False).astype(bool)
    frame = pd.DataFrame({"value": values, "observed": observed}, index=dates)
    last = frame["value"].ffill()
    features = {"last": last}
    for window in (7, 30, 90, 365):
        features[f"mean{window}"] = frame["value"].rolling(window, min_periods=1).mean()
        features[f"std{window}"] = frame["value"].rolling(window, min_periods=1).std(ddof=0)
    features["history"] = frame["observed"].rolling(365, min_periods=1).sum()
    angle = 2.0 * np.pi * dates.dayofyear.to_numpy() / 365.25
    features["sin"] = pd.Series(np.sin(angle), index=dates)
    features["cos"] = pd.Series(np.cos(angle), index=dates)
    return pd.concat([frame, pd.DataFrame(features, index=dates)], axis=1)


def main() -> None:
    args = parse_args()
    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    dates = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    rows: list[dict] = []

    for horizon in horizons:
        per_site: list[dict] = []
        for site, group in panel.groupby("site_no", sort=True):
            frame = site_frame(dates, group)
            target = frame["value"].shift(-horizon)
            target_observed = frame["observed"].shift(-horizon, fill_value=False)
            valid = (
                target_observed
                & frame["history"].ge(args.min_history_obs)
                & frame["last"].notna()
            )
            train_mask = valid & (dates <= train_end)
            test_mask = valid & (dates >= test_start)
            if train_mask.sum() < args.min_train or test_mask.sum() < 30:
                continue

            feature_cols = ["last", "mean7", "std7", "mean30", "std30",
                            "mean90", "std90", "mean365", "std365", "sin", "cos"]
            x_train = frame.loc[train_mask, feature_cols].to_numpy(dtype=float)
            y_train = target.loc[train_mask].to_numpy(dtype=float)
            finite = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train)
            if finite.sum() < args.min_train:
                continue
            x_train = x_train[finite]
            y_train = y_train[finite]
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            std[std < 1e-8] = 1.0
            model = Ridge(alpha=args.alpha)
            model.fit((x_train - mean) / std, y_train)

            x_test = frame.loc[test_mask, feature_cols].to_numpy(dtype=float)
            y_test = target.loc[test_mask].to_numpy(dtype=float)
            finite = np.isfinite(x_test).all(axis=1) & np.isfinite(y_test)
            if finite.sum() < 30:
                continue
            x_test = x_test[finite]
            y_test = y_test[finite]
            prediction = model.predict((x_test - mean) / std)
            persistence = frame.loc[test_mask, "last"].to_numpy(dtype=float)[finite]
            row = {
                "site_no": str(site),
                "n": int(finite.sum()),
                "ridge_nse": nse(y_test, prediction),
                "ridge_kge": kge(y_test, prediction),
                "persistence_nse": nse(y_test, persistence),
                "persistence_kge": kge(y_test, persistence),
            }
            per_site.append(row)

        frame = pd.DataFrame(per_site)
        summary = {
            "horizon": horizon,
            "n_sites": int(len(frame)),
            "ridge_nse_mean": float(frame["ridge_nse"].mean()),
            "ridge_nse_median": float(frame["ridge_nse"].median()),
            "ridge_kge_mean": float(frame["ridge_kge"].mean()),
            "ridge_kge_median": float(frame["ridge_kge"].median()),
            "persistence_nse_mean": float(frame["persistence_nse"].mean()),
            "persistence_nse_median": float(frame["persistence_nse"].median()),
            "persistence_kge_mean": float(frame["persistence_kge"].mean()),
            "persistence_kge_median": float(frame["persistence_kge"].median()),
        }
        rows.append(summary)
        frame.insert(0, "horizon", horizon)
        frame.to_csv(
            PROJECT_ROOT / f"data/processed/ridge_fast_h{horizon}_per_site.csv",
            index=False,
        )
        print(
            f"horizon {horizon}: n={summary['n_sites']} "
            f"ridge NSE mean/median={summary['ridge_nse_mean']:.4f}/"
            f"{summary['ridge_nse_median']:.4f} "
            f"KGE mean/median={summary['ridge_kge_mean']:.4f}/"
            f"{summary['ridge_kge_median']:.4f}"
        )

    out = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
