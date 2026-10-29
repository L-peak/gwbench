#!/usr/bin/env python
"""Re-fit persistence blending on purged validation anchors.

This is a leakage sensitivity analysis for the already-trained models. It
removes validation targets that fall into the test period from the calibration
set. It does not retrain the model, so early stopping still used the original
validation set; the purged retraining kernel is the definitive experiment.
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
    parser.add_argument("--results-dir", default="dist/kaggle_results_seeds")
    parser.add_argument("--horizons", default="30,90")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--model-file", default="best_model.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out-dir", default="data/processed/purged_calibration")
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
    rows = []

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
            state = torch.load(model_dir / args.model_file, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            val_pred = predict_loader(model, val_loader, torch.device("cpu"))
            val_frame = build_prediction_frame(
                arrays, val, val_pred, means, stds,
                lookback=365, horizon=horizon, residual=True,
            )
            alpha = fit_persistence_blend(val_frame)
            alpha.to_csv(out_dir / f"alpha_purged_h{horizon}_seed{seed}.csv", index=False)

            test = pd.read_parquet(model_dir / "test_predictions.parquet")
            old_alpha = pd.read_csv(
                model_dir / "persistence_blend_alpha.csv", dtype={"site_no": str}
            )
            alpha = alpha.rename(columns={"alpha": "alpha_purged"})
            calibrated = apply_persistence_blend(
                test, alpha, alpha_col="alpha_purged"
            )
            evaluation = calibrated.copy()
            evaluation["y_pred"] = evaluation["y_pred_calibrated"]
            per_site, aggregate = evaluate_prediction_frame(evaluation)
            aggregate["horizon"] = horizon
            aggregate["seed"] = seed
            aggregate["alpha_mean"] = float(alpha["alpha_purged"].mean())
            aggregate["alpha_median"] = float(alpha["alpha_purged"].median())
            aggregate["old_alpha_median"] = float(old_alpha["alpha"].median())
            rows.append(aggregate)
            per_site.insert(0, "seed", seed)
            per_site.insert(0, "horizon", horizon)
            per_site.to_csv(
                out_dir / f"test_metrics_per_site_purgedcal_h{horizon}_seed{seed}.csv",
                index=False,
            )
            print(
                f"h={horizon} seed={seed} n={aggregate['n_sites']} "
                f"alpha median={aggregate['alpha_median']:.4f} "
                f"(old {aggregate['old_alpha_median']:.4f}) "
                f"NSE mean/median={aggregate['y_pred_nse_mean']:.4f}/"
                f"{aggregate['y_pred_nse_median']:.4f}"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "purged_calibration_summary.csv", index=False)
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
