import numpy as np
import pandas as pd

from gwbench.windows import (
    WindowDataset,
    build_anchors,
    build_prediction_frame,
    compute_site_stats,
    panel_to_arrays,
    split_anchors_temporal,
)


def make_panel():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    rows = []
    for site, offset in (("A", 0.0), ("B", 100.0)):
        for index, date in enumerate(dates):
            value = np.nan if (site == "B" and index in (3, 4)) else offset + index
            rows.append(
                {
                    "site_no": site,
                    "date": date,
                    "value": value,
                    "observed": not np.isnan(value),
                }
            )
    return pd.DataFrame(rows)


def test_panel_to_arrays_shape():
    arrays = panel_to_arrays(make_panel())
    assert arrays.values.shape == (2, 10)
    assert arrays.observed.shape == (2, 10)
    assert arrays.sites == ["A", "B"]


def test_build_anchors_respects_target_observation():
    arrays = panel_to_arrays(make_panel())
    anchors = build_anchors(
        arrays, lookback=3, horizon=2, stride=1, min_history_obs=1
    )
    assert len(anchors) > 0
    for site, anchor in anchors:
        assert arrays.observed[site, anchor + 2]


def test_compute_site_stats_shapes():
    arrays = panel_to_arrays(make_panel())
    means, stds = compute_site_stats(arrays)
    assert means.shape == (2,)
    assert stds.shape == (2,)
    assert np.all(np.isfinite(means))
    assert np.all(stds > 0)


def test_window_dataset_item():
    arrays = panel_to_arrays(make_panel())
    anchors = build_anchors(
        arrays, lookback=3, horizon=2, stride=1, min_history_obs=1
    )
    means, stds = compute_site_stats(arrays)
    dataset = WindowDataset(
        arrays,
        anchors[:2],
        lookback=3,
        horizon=2,
        means=means,
        stds=stds,
    )
    x, y, site_id, anchor = dataset[0]
    assert x.shape == (3, 4)
    assert y.ndim == 0
    assert site_id.ndim == 0
    assert anchor.ndim == 0


def test_build_prediction_frame_columns():
    arrays = panel_to_arrays(make_panel())
    anchors = build_anchors(
        arrays, lookback=3, horizon=2, stride=1, min_history_obs=1
    )[:3]
    means, stds = compute_site_stats(arrays)
    frame = build_prediction_frame(
        arrays,
        anchors,
        np.zeros(len(anchors), dtype=np.float32),
        means,
        stds,
        lookback=3,
        horizon=2,
    )
    assert set(frame.columns) == {
        "site_no",
        "date",
        "y_true",
        "y_pred",
        "persistence",
        "seasonal_naive",
    }
    assert len(frame) == len(anchors)


def test_residual_zero_prediction_is_persistence():
    arrays = panel_to_arrays(make_panel())
    anchors = build_anchors(
        arrays, lookback=3, horizon=2, stride=1, min_history_obs=1
    )
    means, stds = compute_site_stats(arrays)
    frame = build_prediction_frame(
        arrays,
        anchors,
        np.zeros(len(anchors), dtype=np.float32),
        means,
        stds,
        lookback=3,
        horizon=2,
        residual=True,
    )
    assert np.allclose(
        frame["y_pred"].to_numpy(),
        frame["persistence"].to_numpy(),
        equal_nan=True,
    )


def test_split_anchors_temporal():
    arrays = panel_to_arrays(make_panel())
    anchors = build_anchors(
        arrays, lookback=3, horizon=2, stride=1, min_history_obs=1
    )
    splits = split_anchors_temporal(
        anchors, arrays.dates, train_end="2020-01-05", val_end="2020-01-07"
    )
    assert all(len(splits[key]) > 0 for key in ("train", "val", "test"))
