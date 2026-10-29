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
    hexbin = None
    for ax, (column, title) in zip(axes, panels):
        hexbin = ax.hexbin(
            frame["y_true"],
            frame[column],
            gridsize=45,
            bins="log",
            cmap="Blues",
            mincnt=1,
            linewidths=0.2,
        )
        low = float(np.nanmin(frame["y_true"]))
        high = float(np.nanmax(frame["y_true"]))
        ax.plot([low, high], [low, high], color="black", linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Observed depth (ft)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Predicted depth (ft)")
    fig.colorbar(hexbin, ax=axes, shrink=0.8, label="log$_{10}$(count)")
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
        paths = sorted(
            (PROJECT_ROOT / "data/processed").glob(f"missing_h{horizon}_mask*.csv")
        )
        if not paths:
            continue
        data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        grouped = (
            data.groupby("mask_fraction")["y_pred_nse_mean"]
            .agg(["mean", "std"])
            .reset_index()
        )
        grouped_kge = (
            data.groupby("mask_fraction")["y_pred_kge_mean"]
            .agg(["mean", "std"])
            .reset_index()
        )
        axes[0].errorbar(
            grouped["mask_fraction"] * 100,
            grouped["mean"],
            yerr=grouped["std"].fillna(0.0),
            fmt="-o",
            color=color,
            capsize=2,
            label=f"{horizon} days",
        )
        axes[1].errorbar(
            grouped_kge["mask_fraction"] * 100,
            grouped_kge["mean"],
            yerr=grouped_kge["std"].fillna(0.0),
            fmt="-o",
            color=color,
            capsize=2,
            label=f"{horizon} days",
        )
    axes[0].set_xlabel("Masked input observations (%)")
    axes[0].set_ylabel("NSE mean")
    axes[0].set_title("NSE under missing inputs", fontsize=9)
    axes[1].set_xlabel("Masked input observations (%)")
    axes[1].set_ylabel("KGE mean")
    axes[1].set_title("KGE under missing inputs", fontsize=9)
    for ax in axes:
        ax.set_xticks([0, 10, 30, 50])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_missingness.png")
    plt.close(fig)


def fig_per_site() -> None:
    panel = pd.read_parquet(
        PROJECT_ROOT / "data/processed/panel_core_clean.parquet",
        columns=["site_no", "state_cd"],
    ).drop_duplicates("site_no")
    state_map = dict(zip(panel["site_no"].astype(str), panel["state_cd"].astype(str)))
    colors = {"ca": "#2563eb", "ne": "#16a34a", "tx": "#ea580c"}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    for row, horizon in enumerate((30, 90)):
        frames = []
        for seed in (1, 2, 3):
            path = (
                PROJECT_ROOT
                / "dist/kaggle_results_seeds"
                / f"gru_h{horizon}_seed{seed}"
                / "test_metrics_per_site.csv"
            )
            frames.append(pd.read_csv(path, dtype={"site_no": str}))
        data = pd.concat(frames, ignore_index=True)
        data = (
            data.groupby("site_no", as_index=False)[
                ["y_pred_nse", "persistence_nse"]
            ]
            .mean()
            .copy()
        )
        data["delta_nse"] = data["y_pred_nse"] - data["persistence_nse"]
        data["state"] = data["site_no"].map(state_map).fillna("other")

        ax = axes[row, 0]
        for state, group in data.groupby("state"):
            ax.scatter(
                group["persistence_nse"],
                group["y_pred_nse"],
                s=18,
                alpha=0.75,
                color=colors.get(state, "gray"),
                edgecolor="white",
                linewidth=0.3,
                label=state.upper(),
            )
        lower = -2.0 if horizon == 30 else -15.0
        ax.plot([lower, 1], [lower, 1], color="black", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xscale("symlog", linthresh=1, linscale=1)
        ax.set_yscale("symlog", linthresh=1, linscale=1)
        ax.set_xlim(lower, 1)
        ax.set_ylim(lower, 1)
        ticks = [-10, -3, -1, 0, 1] if horizon == 90 else [-2, -1, 0, 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(tick) for tick in ticks])
        ax.set_yticklabels([str(tick) for tick in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Persistence NSE")
        ax.set_ylabel("Calibrated GRU NSE")
        ax.set_title(f"{horizon}-day forecasts", fontsize=9, loc="left")
        ax.grid(alpha=0.2)
        if row == 0:
            ax.legend(fontsize=6, frameon=False, loc="upper left")

        outlier = data[data["site_no"] == "413156098591201"]
        if row == 1 and len(outlier):
            x_out = float(outlier["persistence_nse"].iloc[0])
            y_out = float(outlier["y_pred_nse"].iloc[0])
            ax.annotate(
                f"outlier: {y_out:.1f}",
                xy=(x_out, y_out),
                xytext=(x_out + 0.08, y_out * 0.55),
                fontsize=6,
                color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.7),
            )

        ax = axes[row, 1]
        clipped = data["delta_nse"].clip(-1.0, 1.0)
        ax.hist(clipped, bins=30, color="#6b7280", alpha=0.75)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(
            float(data["delta_nse"].median()),
            color="#dc2626",
            linestyle="--",
            linewidth=0.8,
            label=f"median {float(data['delta_nse'].median()):+.3f}",
        )
        ax.set_xlabel("Delta NSE (calibrated - persistence)")
        ax.set_ylabel("Wells")
        ax.set_title("Per-site NSE change", fontsize=9, loc="left")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        n_out = int((data["delta_nse"].abs() > 1.0).sum())
        if n_out:
            ax.text(
                0.98,
                0.70,
                f"{n_out} wells outside ±1",
                transform=ax.transAxes,
                ha="right",
                fontsize=6,
                color="#dc2626",
            )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_per_site.png")
    plt.close(fig)


def fig_outlier() -> None:
    site = "413156098591201"
    panel = pd.read_parquet(
        PROJECT_ROOT / "data/processed/panel_core_clean.parquet"
    )
    group = panel[panel["site_no"].astype(str) == site].sort_values("date")

    pred = pd.read_parquet(
        PROJECT_ROOT
        / "dist/kaggle_results_seeds/gru_h90_seed1/test_predictions.parquet"
    )
    pred = pred[pred["site_no"].astype(str) == site].sort_values("date")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2))
    axes[0].plot(group["date"], group["value"], color="#111827", linewidth=0.8)
    for stamp, label in (("2016-01-01", "validation"), ("2019-01-01", "test")):
        axes[0].axvline(pd.Timestamp(stamp), color="#dc2626", linestyle="--", linewidth=0.7)
        axes[0].text(pd.Timestamp(stamp), axes[0].get_ylim()[1], label, fontsize=7, va="top", ha="left", color="#dc2626")
    axes[0].set_title(f"{site}: full observed record", fontsize=9, loc="left")
    axes[0].set_ylabel("Depth to water (ft)")
    axes[0].grid(alpha=0.2)

    axes[1].plot(pred["date"], pred["y_true"], color="#111827", linewidth=0.9, label="observed")
    axes[1].plot(pred["date"], pred["persistence"], color="#6b7280", linestyle="--", linewidth=0.8, label="persistence")
    axes[1].plot(pred["date"], pred["y_pred_calibrated"], color="#2563eb", linewidth=0.9, label="calibrated GRU")
    axes[1].set_title("90-day forecasts in available test period", fontsize=9, loc="left")
    axes[1].set_ylabel("Depth to water (ft)")
    axes[1].set_xlabel("Year")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_outlier.png")
    plt.close(fig)


def main() -> None:
    fig_study_map()
    fig_examples()
    fig_horizon_skill()
    fig_scatter()
    fig_alpha()
    fig_missingness()
    fig_per_site()
    fig_outlier()
    print(f"Wrote figures to {FIG_DIR}")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(" ", path.name, f"{path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
