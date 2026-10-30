#!/usr/bin/env python
"""Sensitivity of per-site blending weights to the calibration window.

For each purged model, validation predictions from 2016-2018 are generated
once. The blending weight is then fitted on 2016-2017, 2018 only, or the full
2016-2018 window, and the resulting test metrics are compared. This addresses
the dual use of the validation period for early stopping and calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from gwbench.calibration import apply_persistence_blend, fit_persistence_blend  # noqa: E402
from gwbench.models.gru import GroundwaterGRU  # noqa: E402
from gwbench.windows import (  # noqa: E402
    WindowDataset,
    build_anchors,
    build_prediction_frame,
    compute_site_stats,
    evaluate_prediction_frame,
    panel_to_arrays,
    predict_loader,
    split_anchors_temporal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_core_clean.parquet")
    parser.add_argument("--results-dir", default="dist/kaggle_results_purged")
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-dir", default="data/processed/calibration_windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    arrays = panel_to_arrays(panel)
    train_dates = arrays.dates <= pd.Timestamp("2015-12-31")
    means, stds = compute_site_stats(arrays, train_dates=train_dates)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    windows = {
        "early_2016_2017": (pd.Timestamp("2015-12-31"), pd.Timestamp("2017-12-31")),
        "late_2018": (pd.Timestamp("2017-12-31"), pd.Timestamp("2018-12-31")),
        "full_2016_2018": (pd.Timestamp("2015-12-31"), pd.Timestamp("2018-12-31")),
    }

    for horizon in horizons:
        anchors = build_anchors(
            arrays, lookback=365, horizon=horizon, stride=1, min_history_obs=30
        )
        splits = split_anchors_temporal(anchors, arrays.dates)
        val = splits["val"]
        val = val[arrays.dates[val[:, 1] + horizon] <= pd.Timestamp("2018-12-31")]
        val_loader = DataLoader(
            WindowDataset(
                arrays, val, lookback=365, horizon=horizon,
                means=means, stds=stds, residual=True,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        val_target_dates = arrays.dates[val[:, 1] + horizon]

        for seed in seeds:
            model_dir = PROJECT_ROOT / args.results_dir / f"gru_h{horizon}_seed{seed}"
            config = json.loads((model_dir / "config.json").read_text())
            model = GroundwaterGRU(
                input_dim=4,
                num_sites=len(arrays.sites),
                hidden_size=config.get("hidden", 64),
                num_layers=config.get("layers", 2),
                dropout=config.get("dropout", 0.2),
                site_embedding_dim=8,
                use_site_embedding=not config.get("no_site_embedding", False),
            )
            state = torch.load(model_dir / "best_model.pt", map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            prediction = predict_loader(model, val_loader, torch.device("cpu"))
            frame = build_prediction_frame(
                arrays, val, prediction, means, stds,
                lookback=365, horizon=horizon, residual=True,
            )
            test = pd.read_parquet(model_dir / "test_predictions.parquet")
            alphas = {}
            for name, (lower, upper) in windows.items():
                mask = (val_target_dates > lower) & (val_target_dates <= upper)
                subset = frame.loc[mask].copy()
                alpha = fit_persistence_blend(subset)
                alphas[name] = alpha
                alpha.to_csv(
                    out_dir / f"alpha_{name}_h{horizon}_seed{seed}.csv", index=False
                )
                alpha = alpha.rename(columns={"alpha": "alpha_window"})
                calibrated = apply_persistence_blend(
                    test, alpha, alpha_col="alpha_window"
                )
                evaluation = calibrated.copy()
                evaluation["y_pred"] = evaluation["y_pred_calibrated"]
                _, aggregate = evaluate_prediction_frame(evaluation)
                rows.append(
                    {
                        "horizon": horizon,
                        "seed": seed,
                        "window": name,
                        "n_val": int(mask.sum()),
                        "alpha_median": float(alpha["alpha_window"].median()),
                        "alpha_mean": float(alpha["alpha_window"].mean()),
                        "nse_mean": aggregate["y_pred_nse_mean"],
                        "nse_median": aggregate["y_pred_nse_median"],
                        "kge_mean": aggregate["y_pred_kge_mean"],
                        "kge_median": aggregate["y_pred_kge_median"],
                    }
                )
            merged = alphas["full_2016_2018"][["site_no", "alpha"]].rename(
                columns={"alpha": "full"}
            )
            for name in ("early_2016_2017", "late_2018"):
                merged = merged.merge(
                    alphas[name][["site_no", "alpha"]].rename(
                        columns={"alpha": name}
                    ),
                    on="site_no",
                )
            corr_early = float(merged["full"].corr(merged["early_2016_2017"]))
            corr_late = float(merged["full"].corr(merged["late_2018"]))
            print(
                f"h={horizon} seed={seed} alpha corr "
                f"full-early={corr_early:.3f} full-late={corr_late:.3f}"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "calibration_window_summary.csv", index=False)
    agg = summary.groupby(["horizon", "window"]).agg(
        nse_mean=("nse_mean", "mean"),
        nse_median=("nse_median", "mean"),
        kge_mean=("kge_mean", "mean"),
        kge_median=("kge_median", "mean"),
        alpha_median=("alpha_median", "mean"),
    )
    print(agg.round(4).to_string())
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
