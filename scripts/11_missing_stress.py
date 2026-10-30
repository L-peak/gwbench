#!/usr/bin/env python
"""Evaluate a trained gwbench model under artificial input missingness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Windows/Anaconda OpenMP workaround, same as the training script.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from gwbench.calibration import apply_persistence_blend  # noqa: E402
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
    parser.add_argument(
        "--model-dir",
        default="dist/kaggle_results_h90/gru_h90",
        help="Directory containing best_model.pt, config.json, and optional alphas",
    )
    parser.add_argument("--fractions", default="0,0.1,0.3,0.5")
    parser.add_argument(
        "--block-lengths",
        default="",
        help="Comma-separated contiguous missing blocks (days); overrides fractions",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Evaluation stride; 0 uses the config's stride_eval",
    )
    parser.add_argument("--mask-seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="data/processed/missing_stress.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = PROJECT_ROOT / args.model_dir
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("split_mode") != "temporal":
        raise SystemExit("This stress test currently supports temporal splits only")

    panel = pd.read_parquet(PROJECT_ROOT / args.panel)
    arrays = panel_to_arrays(panel)
    lookback = int(config["lookback"])
    horizon = int(config["horizon"])
    residual = not bool(config.get("absolute_target", False))

    train_dates = arrays.dates <= pd.Timestamp(config["train_end"])
    means, stds = compute_site_stats(arrays, train_dates=train_dates)
    anchors = build_anchors(
        arrays,
        lookback=lookback,
        horizon=horizon,
        stride=1,
        min_history_obs=int(config.get("min_history_obs", 30)),
    )
    splits = split_anchors_temporal(
        anchors,
        arrays.dates,
        train_end=config["train_end"],
        val_end=config["val_end"],
    )
    stride_eval = (
        args.stride
        if args.stride > 0
        else max(int(config.get("stride_eval", 1)), 1)
    )
    test_anchors = splits["test"][::stride_eval]
    print(f"Panel: {len(arrays.sites)} sites x {len(arrays.dates)} days")
    print(f"Test anchors: {len(test_anchors):,}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model = GroundwaterGRU(
        input_dim=4,
        num_sites=len(arrays.sites),
        hidden_size=int(config["hidden"]),
        num_layers=int(config["layers"]),
        dropout=float(config["dropout"]),
        use_site_embedding=not bool(config.get("no_site_embedding", False)),
    ).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    alpha_path = model_dir / "persistence_blend_alpha.csv"
    alphas = (
        pd.read_csv(alpha_path, dtype={"site_no": str})
        if alpha_path.exists()
        else None
    )
    if alphas is not None:
        print(f"Loaded blending weights: {len(alphas)} sites")

    fractions = [float(value) for value in args.fractions.split(",") if value.strip()]
    block_lengths = [
        int(value) for value in args.block_lengths.split(",") if value.strip()
    ]
    conditions = (
        [("block", value) for value in block_lengths]
        if block_lengths
        else [("fraction", value) for value in fractions]
    )
    rows: list[dict] = []

    for kind, value in conditions:
        dataset = WindowDataset(
            arrays,
            test_anchors,
            lookback=lookback,
            horizon=horizon,
            means=means,
            stds=stds,
            residual=residual,
            mask_fraction=value if kind == "fraction" else 0.0,
            mask_block_length=value if kind == "block" else 0,
            mask_seed=args.mask_seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        predictions = predict_loader(model, loader, device)
        frame = build_prediction_frame(
            arrays,
            test_anchors,
            predictions,
            means,
            stds,
            lookback=lookback,
            horizon=horizon,
            residual=residual,
        )
        if alphas is not None:
            frame = apply_persistence_blend(frame, alphas)
            frame["y_pred"] = frame["y_pred_calibrated"]

        per_site, aggregate = evaluate_prediction_frame(frame)
        aggregate = dict(aggregate)
        aggregate["mask_fraction"] = value if kind == "fraction" else 0.0
        aggregate["mask_block_length"] = value if kind == "block" else 0
        rows.append(aggregate)
        print(
            f"{kind}={value} n_sites={aggregate.get('n_sites')} "
            f"nse_mean={aggregate.get('y_pred_nse_mean'):.4f} "
            f"nse_median={aggregate.get('y_pred_nse_median'):.4f} "
            f"kge_mean={aggregate.get('y_pred_kge_mean'):.4f}"
        )

    summary = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
