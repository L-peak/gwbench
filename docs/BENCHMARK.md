# gwbench benchmark protocol and data dictionary

This document records the exact panel construction, quality-control (QC)
rules, anchor protocol, and file layout used in the manuscript.

## 1. Data source

- USGS National Water Information System (NWIS), Water Data OGC API
- parameter code `72019` (depth to water below land surface, ft)
- statistic code `00001` (daily maximum)
- states: CA, NE, TX
- requested period: 2000-01-01 through 2024-12-31
- retrieval: October 2026; legacy daily-values service used as a batched
  fallback when the OGC endpoint rate-limited

Well selection criteria:

1. at least 3,650 observed daily values;
2. first-to-last span of at least 10 years;
3. record extending to at least 2015-01-01.

Result: 279 wells and 1,581,469 observed daily values.

## 2. Panels

| Panel | Wells | Rule |
|---|---:|---|
| Full | 279 | all selected wells |
| Core | 89 | observed coverage >= 70% |
| Clean core | 74 | core and <= 5 day-to-day jumps > 5 ft |

The clean core is the primary analysis set. The full and core panels are
provided for sensitivity analysis and extension.

## 3. QC and exclusion flow

```
USGS daily values
      |
      v
remove sentinel values (100000.x)
      |
      v
build complete daily panel + observed/missing mask
      |
      v
flag day-to-day jumps > 5 ft with gap <= 2 days
      |
      +--> coverage < 70% ----------> excluded from core
      |
      +--> > 5 flagged jumps -------> excluded from clean core
      |
      v
core (89 wells) / clean core (74 wells)
```

Jumps are flagged, not silently corrected, because they may reflect real
datum or sensor changes. The per-well manifest is `docs/site_manifest.csv`.

Manifest columns:

| Column | Meaning |
|---|---|
| `site_no` | USGS site identifier |
| `state_cd` | CA, NE, or TX |
| `n_days` | days in the complete panel |
| `n_observed` | observed daily values |
| `coverage` | observed fraction |
| `n_jumps` | day-to-day jumps > 5 ft with gap <= 2 days |
| `max_abs_jump` | largest absolute day-to-day change |
| `n_gaps_gt_7d` | number of gaps longer than 7 days |
| `longest_gap_days` | longest gap |
| `value_min`, `value_max`, `value_std` | observed value range and standard deviation |
| `in_full`, `in_core`, `in_clean` | panel membership flags |
| `exclusion_reason` | reason for exclusion from core/clean core |

## 4. Anchor protocol

- input window: 365 days;
- anchor t: last day of the input window;
- target: day t+h, where h is 30 or 90 days;
- valid anchor: target observed and at least 30 observed input values;
- missing inputs are represented by the mask channel;
- anchors with missing targets are excluded, not imputed;
- training anchors: stride 3;
- validation and test anchors: stride 1.

### Purged temporal split

To prevent target leakage across boundaries:

- training: anchor date <= 2015-12-31 and target date <= 2015-12-31;
- validation: anchor date in 2016-2018 and target date <= 2018-12-31;
- test: anchor date >= 2019-01-01 (target therefore in the test period).

After purging, the 30-day model uses 116,415 training and 76,493 validation
anchors; the 90-day model uses 114,941 and 72,462. Test anchors: 103,956
(30 days) and 99,750 (90 days) before per-site filtering.

## 5. Models and baselines

- persistence: last observed value in the input window;
- seasonal naive: value 365 days before the target;
- per-site ridge: last value, 7/30/90/365-day rolling means and standard
  deviations, day-of-year sine/cosine; no linear trend;
- global residual GRU: two layers, hidden 64, dropout 0.2, site embedding,
  residual target relative to the last observation;
- fixed-alpha fusion: one global alpha = median validation-fitted per-site
  weight;
- per-site fusion: alpha_s fitted on validation only and clipped to [0,1].

## 6. File layout

| Path | Contents |
|---|---|
| `data/processed/panel_full.parquet` | 279-well complete daily panel |
| `data/processed/panel_core.parquet` | 89-well screened panel |
| `data/processed/panel_core_clean.parquet` | 74-well primary panel |
| `docs/site_manifest.csv` | per-well QC, coverage, jumps, panel membership |
| `data/processed/persistence_anchor_protocol.csv` | persistence skill by horizon |
| `data/processed/bootstrap_summary_purged.csv` | paired site-level bootstrap |
| `data/processed/fusion_ablation_h*_purged.csv` | fusion ablation |
| `data/processed/missing_summary_purged.csv` | random point-masking summary |
| `data/processed/missing_summary_block.csv` | contiguous-block masking summary |
| `data/processed/calibration_windows/` | calibration-window sensitivity |
| `paper/figures/` | manuscript figures 1-8 |

## 7. Reproduce the tables and figures

```bash
pip install -r requirements.txt

# panels and QC
python scripts/03_build_panel.py --raw data/raw/levels_daily_full.parquet \
  --out data/processed/panel_full.parquet
python scripts/09_filter_qc.py
python scripts/06_qc.py --panel data/processed/panel_full.parquet \
  --out data/processed/qc_full.csv

# baselines and model evaluation
python scripts/13_ridge_aligned_fast.py \
  --panel data/processed/panel_core_clean.parquet --horizons 30,90
python scripts/15_bootstrap.py --results-dir dist/kaggle_results_purged \
  --out data/processed/bootstrap_summary_purged.csv
python scripts/17_fusion_ablation.py --results-dir dist/kaggle_results_purged \
  --tag _purged

# missing-input stress
python scripts/11_missing_stress.py ...   # see README
python scripts/16_aggregate_missing.py --prefix missing_purged \
  --out data/processed/missing_summary_purged.csv

# manuscript figures
GWBENCH_RESULTS_DIR=dist/kaggle_results_purged \
GWBENCH_MISSING_PREFIX=missing_purged python scripts/12_make_figures.py
```

The exact training configuration is in `notebooks/KAGGLE.md` and
`notebooks/kaggle_purged/`. Each saved model directory contains
`config.json`, `best_model.pt`, `persistence_blend_alpha.csv`,
`test_predictions.parquet`, and per-site metrics.
