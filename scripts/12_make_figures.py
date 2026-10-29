#!/usr/bin/env python
"""Generate the manuscript figures from processed panels and experiment results."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gwbench.panel import evaluate_predictions, temporal_split  # noqa: E402

FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    }
)

STATE_COLORS = {"ca": "#2563eb", "ne": "#16a34a", "tx": "#ea580c"}


def fig_study_map() -> None:
    panel = pd.read_parquet(
        PROJECT_ROOT / "data/processed/panel_full.parquet",
        columns=["site_no", "state_cd", "dec_lat_va", "dec_long_va"],
    )
    sites = panel.drop_duplicates("site_no").copy()
    sites["dec_lat_va"] = pd.to_numeric(sites["dec_lat_va"], errors="coerce")
    sites["dec_long_va"] = pd.to_numeric(sites["dec_long_va"], errors="coerce")
    sites = sites.dropna(subset=["dec_lat_va", "dec_long_va"])
    core = pd.read_parquet(
        PROJECT_ROOT / "data/processed/panel_core_clean.parquet",
        columns=["site_no"],
    )["site_no"].unique()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for state, group in sites.groupby("state_cd"):
        ax.scatter(
            group["dec_long_va"],
            group["dec_lat_va"],
            s=10,
            alpha=0.35,
            color=STATE_COLORS.get(state, "gray"),
            label=f"{state.upper()} all wells",
        )
    core_sites = sites[sites["site_no"].isin(core)]
    for state, group in core_sites.groupby("state_cd"):
        ax.scatter(
            group["dec_long_va"],
            group["dec_lat_va"],
            s=18,
            alpha=0.9,
            color=STATE_COLORS.get(state, "black"),
            edgecolor="black",
            linewidth=0.2,
            label=f"{state.upper()} clean core",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.savefig(FIG_DIR / "fig1_study_map.png")
    plt.close(fig)


def fig_examples() -> None:
    panel = pd.read_parquet(PROJECT_ROOT / "data/processed/panel_full.parquet")
    examples = [
        ("404618098504401", "Clean record (NE)", "#16a34a"),
        ("315712106361803", "Frequent jumps (TX)", "#ea580c"),
        ("340615117170902", "Large late gap (CA)", "#2563eb"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    for ax, (site, title, color) in zip(axes, examples):
        group = panel[panel["site_no"].astype(str) == site].sort_values("date")
        ax.plot(group["date"], group["value"], color=color, linewidth=0.8)
        ax.set_ylabel("Depth to water (ft)")
        ax.set_title(f"{site}: {title}", fontsize=9, loc="left")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_examples.png")
    plt.close(fig)


def fig_horizon_skill() -> None:
    curve = pd.read_csv(PROJECT_ROOT / "data/processed/persistence_anchor_protocol.csv")
    curve = curve.rename(
        columns={
            "y_pred_nse_mean": "nse_mean",
            "y_pred_nse_median": "nse_median",
            "y_pred_kge_mean": "kge_mean",
        }
    )
    horizons = curve["horizon"].tolist()

    # Calibrated GRU, seeds 1-3, from the seed experiment.
    model = pd.DataFrame(
        [
            {"horizon": 30, "nse_mean": 0.5753, "nse_mean_std": 0.0120,
             "nse_median": 0.7653, "nse_median_std": 0.0103},
            {"horizon": 90, "nse_mean": -0.1885, "nse_mean_std": 0.0925,
             "nse_median": 0.2669, "nse_median_std": 0.0221},
        ]
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(curve["horizon"], curve["nse_mean"], "-o", color="#6b7280", label="Persistence mean")
    ax.plot(curve["horizon"], curve["nse_median"], "--s", color="#9ca3af", label="Persistence median")
    ax.errorbar(
        model["horizon"],
        model["nse_mean"],
        yerr=model["nse_mean_std"],
        fmt="o",
        color="#2563eb",
        capsize=2,
        label="Calibrated GRU mean",
    )
    ax.errorbar(
        model["horizon"],
        model["nse_median"],
        yerr=model["nse_median_std"],
        fmt="s",
        color="#60a5fa",
        capsize=2,
        label="Calibrated GRU median",
    )
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("NSE")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(FIG_DIR / "fig3_horizon_skill.png")
    plt.close(fig)


def fig_scatter() -> None:
    frame = pd.read_parquet(
        PROJECT_ROOT / "dist/kaggle_results_h90/gru_h90/test_predictions.parquet"
    )
    if len(frame) > 20000:
        frame = frame.sample(n=20000, random_state=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
    panels = [
        ("persistence", "Persistence"),
        ("y_pred_calibrated", "Calibrated GRU"),
    ]
    for ax, (column, title) in zip(axes, panels):
        ax.scatter(frame["y_true"], frame[column], s=2, alpha=0.25, color="#2563eb")
        low = float(np.nanmin(frame["y_true"]))
        high = float(np.nanmax(frame["y_true"]))
        ax.plot([low, high], [low, high], color="black", linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Observed depth (ft)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted depth (ft)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_scatter_h90.png")
    plt.close(fig)


def fig_alpha() -> None:
    rows = []
    for horizon in (30, 90):
        for seed in (1, 2, 3):
            path = (
                PROJECT_ROOT
                / "dist/kaggle_results_seeds"
                / f"gru_h{horizon}_seed{seed}"
                / "test_metrics_per_site.csv"
            )
            cal = pd.read_csv(path, dtype={"site_no": str})
            alpha = pd.read_csv(
                path.parent / "persistence_blend_alpha.csv", dtype={"site_no": str}
            )
            merged = cal.merge(alpha, on="site_no")
            merged["horizon"] = horizon
            merged["seed"] = seed
            rows.append(merged)
    data = pd.concat(rows, ignore_index=True)
    data["delta_nse"] = data["y_pred_nse"] - data["persistence_nse"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for horizon, color in ((30, "#2563eb"), (90, "#ea580c")):
        subset = data[data["horizon"] == horizon]
        axes[0].hist(
            subset["alpha"],
            bins=np.linspace(0, 1, 21),
            alpha=0.45,
            color=color,
            label=f"{horizon} days",
        )
        axes[1].scatter(
            subset["alpha"],
            subset["delta_nse"],
            s=6,
            alpha=0.3,
            color=color,
            label=f"{horizon} days",
        )
    axes[0].set_xlabel("Blending weight alpha")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Alpha distribution", fontsize=9)
    axes[1].axhline(0, color="black", linewidth=0.6, alpha=0.5)
    axes[1].set_xlabel("Blending weight alpha")
    axes[1].set_ylabel("Delta NSE (model - persistence)")
    axes[1].set_title("Per-site effect", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_alpha.png")
    plt.close(fig)


def fig_missingness() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for horizon, color in ((30, "#2563eb"), (90, "#ea580c")):
        path = PROJECT_ROOT / f"data/processed/missing_stress_h{horizon}.csv"
        df = pd.read_csv(path)
        axes[0].plot(
            df["mask_fraction"] * 100,
            df["y_pred_nse_mean"],
            "-o",
            color=color,
            label=f"{horizon} days",
        )
        axes[1].plot(
            df["mask_fraction"] * 100,
            df["y_pred_kge_mean"],
            "-o",
            color=color,
            label=f"{horizon} days",
        )
    axes[0].set_xlabel("Masked input observations (%)")
    axes[0].set_ylabel("NSE mean")
    axes[0].set_title("NSE under missing inputs", fontsize=9)
    axes[1].set_xlabel("Masked input observations (%)")
    axes[1].set_ylabel("KGE mean")
    axes[1].set_title("KGE under missing inputs", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_missingness.png")
    plt.close(fig)


def main() -> None:
    fig_study_map()
    fig_examples()
    fig_horizon_skill()
    fig_scatter()
    fig_alpha()
    fig_missingness()
    print(f"Wrote figures to {FIG_DIR}")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(" ", path.name, f"{path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
