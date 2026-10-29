#!/usr/bin/env python
"""Train a global GRU on a groundwater panel and evaluate against baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Windows/Anaconda can load two OpenMP runtimes (PyTorch + MKL).  This is the
# standard workaround; it is harmless on Linux/Kaggle.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from gwbench.calibration import (  # noqa: E402
    apply_persistence_blend,
    fit_persistence_blend,
)
from gwbench.models.gru import GroundwaterGRU  # noqa: E402
from gwbench.panel import site_split  # noqa: E402
from gwbench.windows import (  # noqa: E402
    WindowDataset,
    build_anchors,
    build_prediction_frame,
    compute_site_stats,
    evaluate_prediction_frame,
    panel_to_arrays,
    predict_loader,
    split_anchors_by_site,
    split_anchors_temporal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel_pilot.parquet")
    parser.add_argument("--split-mode", choices=["temporal", "site"], default="temporal")
    parser.add_argument("--train-end", default="2015-12-31")
    parser.add_argument("--val-end", default="2018-12-31")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--lookback", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride-train", type=int, default=7)
    parser.add_argument("--stride-eval", type=int, default=1)
    parser.add_argument("--min-history-obs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--no-site-embedding",
        action="store_true",
        help="Remove the site embedding; use for leave-site-out transfer",
    )
    parser.add_argument(
        "--absolute-target",
        action="store_true",
        help="Predict the absolute normalized level instead of a residual from the last observation",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip validation-based persistence blending",
    )
    parser.add_argument(
        "--purge-targets",
        action="store_true",
        help=(
            "Remove training/validation anchors whose target date crosses the "
            "split boundary (purged temporal split)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", default="data/processed/gru")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_rmse(model, loader, device: torch.device) -> float:
    model.eval()
    squared: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, site_id, _ in loader:
            x = x.to(device)
            y = y.to(device)
            site_id = site_id.to(device)
            prediction = model(x, site_id)
            squared.append(((prediction - y) ** 2).cpu().numpy())
    if not squared:
        return float("nan")
    values = np.concatenate(squared)
    return float(np.sqrt(np.mean(values)))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    panel_path = PROJECT_ROOT / args.panel
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}; run 03_build_panel.py first")
    panel = pd.read_parquet(panel_path)

    arrays = panel_to_arrays(panel)
    print(f"Panel: {len(arrays.sites)} sites x {len(arrays.dates)} days")

    if args.split_mode == "temporal":
        train_dates = arrays.dates <= pd.Timestamp(args.train_end)
        means, stds = compute_site_stats(arrays, train_dates=train_dates)
        all_anchors = build_anchors(
            arrays,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=1,
            min_history_obs=args.min_history_obs,
        )
        splits = split_anchors_temporal(
            all_anchors,
            arrays.dates,
            train_end=args.train_end,
            val_end=args.val_end,
        )
        if args.purge_targets:
            train_target_dates = arrays.dates[splits["train"][:, 1] + args.horizon]
            val_target_dates = arrays.dates[splits["val"][:, 1] + args.horizon]
            n_train_before = len(splits["train"])
            n_val_before = len(splits["val"])
            splits["train"] = splits["train"][
                train_target_dates <= pd.Timestamp(args.train_end)
            ]
            splits["val"] = splits["val"][
                val_target_dates <= pd.Timestamp(args.val_end)
            ]
            print(
                "Purged boundary-crossing anchors: "
                f"train {n_train_before - len(splits['train']):,}, "
                f"val {n_val_before - len(splits['val']):,}"
            )
    else:
        meta_cols = ["site_no"]
        if "state_cd" in panel.columns:
            meta_cols.append("state_cd")
        site_meta = panel[meta_cols].drop_duplicates("site_no")
        site_meta = site_split(
            site_meta,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            stratify_col="state_cd" if "state_cd" in site_meta.columns else None,
        )
        label_map = site_meta.set_index("site_no")["split"]
        site_labels = label_map.reindex(arrays.sites).fillna("train").to_numpy()
        train_sites = site_labels == "train"
        means, stds = compute_site_stats(arrays, train_sites=train_sites)
        all_anchors = build_anchors(
            arrays,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=1,
            min_history_obs=args.min_history_obs,
        )
        splits = split_anchors_by_site(all_anchors, site_labels)

    train_anchors = splits["train"][:: max(args.stride_train, 1)]
    val_anchors = splits["val"][:: max(args.stride_eval, 1)]
    test_anchors = splits["test"][:: max(args.stride_eval, 1)]
    print(
        f"Anchors: train={len(train_anchors):,} "
        f"val={len(val_anchors):,} test={len(test_anchors):,}"
    )
    if len(train_anchors) == 0:
        raise SystemExit("No training anchors; check the split settings")

    residual = not args.absolute_target
    print(f"Target: {'residual from last observation' if residual else 'absolute level'}")
    train_ds = WindowDataset(
        arrays,
        train_anchors,
        lookback=args.lookback,
        horizon=args.horizon,
        means=means,
        stds=stds,
        residual=residual,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    val_loader = None
    if len(val_anchors):
        val_loader = DataLoader(
            WindowDataset(
                arrays,
                val_anchors,
                lookback=args.lookback,
                horizon=args.horizon,
                means=means,
                stds=stds,
                residual=residual,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    test_loader = None
    if len(test_anchors):
        test_loader = DataLoader(
            WindowDataset(
                arrays,
                test_anchors,
                lookback=args.lookback,
                horizon=args.horizon,
                means=means,
                stds=stds,
                residual=residual,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model = GroundwaterGRU(
        input_dim=4,
        num_sites=len(arrays.sites),
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        use_site_embedding=not args.no_site_embedding,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y, site_id, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            site_id = site_id.to(device)
            optimizer.zero_grad()
            prediction = model(x, site_id)
            loss = loss_fn(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        train_rmse = math.sqrt(float(np.mean(losses)))
        if val_loader is not None:
            val_rmse = loader_rmse(model, val_loader, device)
        else:
            val_rmse = train_rmse

        improved = val_rmse < best_val
        if improved:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        torch.save(model.state_dict(), out_dir / "last_model.pt")

        print(
            f"epoch {epoch:03d} train_rmse={train_rmse:.4f} "
            f"val_rmse={val_rmse:.4f}{' *' if improved else ''}"
        )
        if patience_counter >= args.patience:
            print("Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pt")

    config = vars(args).copy()
    config["n_sites"] = len(arrays.sites)
    config["n_dates"] = len(arrays.dates)
    config["best_val_rmse"] = best_val
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    if test_loader is not None:
        predictions = predict_loader(model, test_loader, device)
        frame = build_prediction_frame(
            arrays,
            test_anchors,
            predictions,
            means,
            stds,
            lookback=args.lookback,
            horizon=args.horizon,
            residual=residual,
        )
        per_site_raw, aggregate_raw = evaluate_prediction_frame(frame)
        per_site_raw.to_csv(
            out_dir / "test_metrics_per_site_raw.csv", index=False
        )
        (out_dir / "test_metrics_raw.json").write_text(
            json.dumps(aggregate_raw, indent=2, default=float)
        )
        print("\nRaw test metrics:")
        print(json.dumps(aggregate_raw, indent=2, default=float))

        if not args.no_calibration and val_loader is not None and len(val_anchors):
            val_predictions = predict_loader(model, val_loader, device)
            val_frame = build_prediction_frame(
                arrays,
                val_anchors,
                val_predictions,
                means,
                stds,
                lookback=args.lookback,
                horizon=args.horizon,
                residual=residual,
            )
            alphas = fit_persistence_blend(val_frame)
            alphas.to_csv(out_dir / "persistence_blend_alpha.csv", index=False)
            frame = apply_persistence_blend(frame, alphas)
            evaluation_frame = frame.copy()
            evaluation_frame["y_pred"] = evaluation_frame["y_pred_calibrated"]
            per_site, aggregate = evaluate_prediction_frame(evaluation_frame)
            per_site.to_csv(
                out_dir / "test_metrics_per_site_calibrated.csv", index=False
            )
            (out_dir / "test_metrics_calibrated.json").write_text(
                json.dumps(aggregate, indent=2, default=float)
            )
            print("\nCalibrated test metrics:")
            print(json.dumps(aggregate, indent=2, default=float))
        else:
            per_site, aggregate = per_site_raw, aggregate_raw
            print("\nCalibration skipped; headline metrics are raw.")

        frame.to_parquet(out_dir / "test_predictions.parquet", index=False)
        per_site.to_csv(out_dir / "test_metrics_per_site.csv", index=False)
        (out_dir / "test_metrics.json").write_text(
            json.dumps(aggregate, indent=2, default=float)
        )
        print("\nHeadline test metrics:")
        print(json.dumps(aggregate, indent=2, default=float))
        print(per_site.head(20).to_string(index=False))
    else:
        print("No test anchors; model saved without evaluation")

    print(f"\nArtifacts -> {out_dir}")


if __name__ == "__main__":
    main()
