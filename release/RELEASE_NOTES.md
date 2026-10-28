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
- `data/processed/missing_stress_h30.csv` and `_h90.csv` - robustness results
- `dist/kaggle_results_seeds/seed_runs.json` - multi-seed result summary
- `docs/paper_outline.md` - paper outline and novelty statement
- `notebooks/KAGGLE.md` - Kaggle reproduction instructions

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
domain. If you use this release, cite the manuscript and the Zenodo DOI.
