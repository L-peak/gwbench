import numpy as np
import pandas as pd

from gwbench.calibration import apply_persistence_blend, fit_persistence_blend


def make_frame():
    rng = np.random.default_rng(0)
    rows = []
    # The model deviation is scaled by alpha_true; the calibration fit
    # recovers 1 / alpha_true (clipped to [0, 1]), so 2.0 -> 0.5.
    for site, alpha_true in (("A", 2.0), ("B", 0.0)):
        baseline = rng.normal(10, 1, size=100)
        target = baseline + rng.normal(0, 0.2, size=100)
        prediction = baseline + alpha_true * (target - baseline)
        for t, b, p, y in zip(range(100), baseline, prediction, target):
            rows.append(
                {
                    "site_no": site,
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=t),
                    "y_true": y,
                    "y_pred": p,
                    "persistence": b,
                    "seasonal_naive": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_fit_alpha():
    alphas = fit_persistence_blend(make_frame())
    values = alphas.set_index("site_no")["alpha"]
    assert abs(values["A"] - 0.5) < 0.05
    assert values["B"] == 0.0


def test_apply_blend():
    frame = make_frame()
    alphas = fit_persistence_blend(frame)
    calibrated = apply_persistence_blend(frame, alphas)
    assert "y_pred_calibrated" in calibrated.columns
    site_a = calibrated[calibrated["site_no"] == "A"]
    site_b = calibrated[calibrated["site_no"] == "B"]
    assert np.allclose(
        site_b["y_pred_calibrated"], site_b["persistence"], equal_nan=True
    )
    assert not np.allclose(
        site_a["y_pred_calibrated"], site_a["persistence"], equal_nan=True
    )
