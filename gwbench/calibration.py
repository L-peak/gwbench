"""Post-hoc calibration of model forecasts against a persistence baseline.

Groundwater levels are slow-moving, so a model that deviates too far from the
last observation can be much worse than persistence on low-variance wells.
This module fits a per-site blending weight on validation data:

    calibrated = persistence + alpha * (model - persistence)

with ``alpha`` clipped to [0, 1].  A site whose model is worse than persistence
gets ``alpha = 0``; a site with a useful model gets a positive weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_persistence_blend(
    frame: pd.DataFrame,
    *,
    min_points: int = 30,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
) -> pd.DataFrame:
    """Fit one blending weight per site on validation predictions.

    ``frame`` must contain ``site_no``, ``y_true``, ``y_pred``, and
    ``persistence``.  The closed-form least-squares weight is clipped to
    ``[alpha_min, alpha_max]``.
    """
    rows: list[dict] = []
    for site, group in frame.groupby("site_no", sort=True):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        baseline = group["persistence"].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(baseline)

        if mask.sum() < min_points:
            alpha = alpha_min
        else:
            delta = y_pred[mask] - baseline[mask]
            target = y_true[mask] - baseline[mask]
            denominator = float(np.dot(delta, delta))
            if denominator <= 1e-12:
                alpha = alpha_min
            else:
                alpha = float(np.dot(delta, target) / denominator)
                alpha = float(np.clip(alpha, alpha_min, alpha_max))

        rows.append(
            {
                "site_no": site,
                "alpha": alpha,
                "n_val": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def apply_persistence_blend(
    frame: pd.DataFrame,
    alphas: pd.DataFrame,
    *,
    alpha_col: str = "alpha",
) -> pd.DataFrame:
    """Apply per-site blending weights and add ``y_pred_calibrated``."""
    out = frame.merge(alphas[["site_no", alpha_col]], on="site_no", how="left")
    out[alpha_col] = out[alpha_col].fillna(0.0)
    out["y_pred_calibrated"] = out["persistence"] + out[alpha_col] * (
        out["y_pred"] - out["persistence"]
    )
    return out
