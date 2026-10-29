#!/usr/bin/env python
"""Per-site ridge on the common anchor protocol, memory-bounded.

This is the streaming counterpart of ``13_ridge_aligned.py``. It walks one
well at a time and never pivots the full panel, so it can be used on the
279-well panel in bounded memory.
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


def anchor_features(
    values: np.ndarray, observed: np.ndarray, dates: pd.DatetimeIndex, anchor: int
) -> tuple[list[float], float] | None:
    start = anchor - 365 + 1
    segment_values = values[start : anchor + 1]
    segment_mask = observed[start : anchor + 1]
    observed_values = segment_values[segment_mask]
    if observed_values.size == 0:
        return None
    last_obs = float(observed_values[-1])
    features = [last_obs]
    for window in (7, 30, 90, 365):
        vals = segment_values[-window:]
        mask = segment_mask[-window:]
        window_values = vals[mask]
        features.append(float(window_values.mean()) if window_values.size else last_obs)
        features.append(float(window_values.std()) if window_values.size > 1 else 0.0)
    angle = 2.0 * np.pi * dates[anchor].dayofyear / 365.25
    features.extend([float(np.sin(angle)), float(np.cos(angle))])
    return features, last_obs


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
        per_site_rows: list[dict] = []
        for site, group in panel.groupby("site_no", sort=True):
            group = group.set_index("date").sort_index()
            values = group["value"].reindex(dates).to_numpy(dtype=float)
            observed = (
                group["observed"].reindex(dates).fillna(False).to_numpy(dtype=bool)
            )
            start = 365 - 1
            stop = len(dates) - horizon
            train_x: list[list[float]] = []
            train_y: list[float] = []
            test_records: list[tuple[int, list[float], float, float, float]] = []

            for anchor in range(start, stop):
                if not observed[anchor + horizon]:
                    continue
                history = observed[anchor - 365 + 1 : anchor + 1]
                if history.sum() < args.min_history_obs:
                    continue
                extracted = anchor_features(values, observed, dates, anchor)
                if extracted is None:
                    continue
                features, last_obs = extracted
                target = float(values[anchor + horizon])
                anchor_date = dates[anchor]
                if anchor_date <= train_end:
                    if all(np.isfinite(features)):
                        train_x.append(features)
                        train_y.append(target)
                elif anchor_date >= test_start:
                    seasonal_index = anchor + horizon - 365
                    seasonal = (
                        float(values[seasonal_index])
                        if seasonal_index >= 0 and observed[seasonal_index]
                        else np.nan
                    )
                    test_records.append((anchor, features, target, last_obs, seasonal))

            if len(train_x) < args.min_train or len(test_records) < 30:
                continue
            x_train = np.asarray(train_x, dtype=float)
            y_train = np.asarray(train_y, dtype=float)
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            std[std < 1e-8] = 1.0
            model = Ridge(alpha=args.alpha)
            model.fit((x_train - mean) / std, y_train)

            x_test = np.asarray([record[1] for record in test_records], dtype=float)
            predictions = model.predict((x_test - mean) / std)
            true = np.asarray([record[2] for record in test_records], dtype=float)
            persistence = np.asarray([record[3] for record in test_records], dtype=float)
            seasonal = np.asarray([record[4] for record in test_records], dtype=float)
            row = {
                "site_no": str(site),
                "n": int(len(true)),
                "ridge_nse": nse(true, predictions),
                "ridge_kge": kge(true, predictions),
                "persistence_nse": nse(true, persistence),
                "persistence_kge": kge(true, persistence),
            }
            mask = np.isfinite(seasonal)
            row["seasonal_nse"] = nse(true[mask], seasonal[mask]) if mask.sum() >= 30 else np.nan
            row["seasonal_kge"] = kge(true[mask], seasonal[mask]) if mask.sum() >= 30 else np.nan
            per_site_rows.append(row)

        frame = pd.DataFrame(per_site_rows)
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
            "seasonal_nse_mean": float(frame["seasonal_nse"].mean()),
            "seasonal_nse_median": float(frame["seasonal_nse"].median()),
        }
        rows.append(summary)
        frame.insert(0, "horizon", horizon)
        frame.to_csv(
            PROJECT_ROOT / f"data/processed/ridge_long_h{horizon}_per_site.csv",
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
