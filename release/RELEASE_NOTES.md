# gwbench v0.1.0 release notes

This release accompanies the manuscript:

> When Does Deep Learning Beat Persistence for Groundwater-Level Forecasting?
> A Reproducible Benchmark and a Persistence-Anchored Residual Model

## Contents

- `gwbench/` - Python package (data access, QC, panels, windows, metrics,
  residual GRU, calibration)
- `scripts/` - numbered pipeline and experiment scripts
- `paper/` - LaTeX manuscript, bibliography, and figures
- `data/processed/` - processed daily panels used in the paper
- `data/processed/persistence_anchor_protocol.csv` - persistence skill table
- `data/processed/missing_h30_mask{1..5}.csv` and
  `data/processed/missing_h90_mask{1..5}.csv` - repeated missing-input stress
  results; `missing_summary.csv` aggregates them
- `data/processed/fusion_ablation_h30.csv` and `_h90.csv` - raw/fixed/per-site
  calibration ablation
- `data/processed/ridge_aligned.csv`, `ridge_aligned_core.csv`, and
  `ridge_aligned_full.csv` - ridge on the common GRU anchors
- `data/processed/panel_sensitivity.csv` - cross-panel persistence and
  seasonal-naive skill
- `data/processed/bootstrap_summary.csv` - paired site-level bootstrap
  intervals
- `dist/kaggle_results_seeds/seed_runs.json` - multi-seed result summary
- `docs/paper_outline.md` - paper outline and novelty statement
- `notebooks/KAGGLE.md` - Kaggle reproduction instructions

The revised manuscript adds five-mask missingness experiments, the fusion
ablation, per-site and cross-panel analyses, bootstrap intervals, and a
reproducibility appendix. New analysis scripts are
`scripts/13_ridge_aligned_fast.py`, `scripts/14_panel_sensitivity.py`,
`scripts/15_bootstrap.py`, and `scripts/16_aggregate_missing.py`.

Raw USGS observations are not redistributed; `scripts/01_fetch_sites.py` and
`scripts/02_fetch_levels.py` reproduce the download. See `data/README.md`.

## Reproduce the main results

```bash
pip install -r requirements.txt
python scripts/03_build_panel.py \
  --raw data/raw/levels_daily_full.parquet \
  --out data/processed/panel_full.parquet
python scripts/09_filter_qc.py
python scripts/04_baselines.py --panel data/processed/panel_core_clean.parquet
python scripts/07_train_gru.py \
  --panel data/processed/panel_core_clean.parquet \
  --horizon 90 --lookback 365 --stride-train 3 \
  --epochs 40 --out-dir results/gru_h90
```

The exact Kaggle configuration used for the paper is in
`notebooks/KAGGLE.md`.

## License

Code: MIT (see `LICENSE`). Derived panels: CC BY 4.0. Raw USGS data: public
domain. If you use this release, cite the manuscript and the Zenodo DOI:
https://doi.org/10.5281/zenodo.23749304
