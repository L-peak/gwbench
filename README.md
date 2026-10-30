# gwbench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23749304.svg)](https://doi.org/10.5281/zenodo.23749304)

Reproducible benchmarks for groundwater-level machine learning, built on
public USGS Water Data.

**Repository:** https://github.com/L-peak/gwbench
**Archived release:** https://doi.org/10.5281/zenodo.23749304

## Status

The pipeline runs end to end on the public USGS panel:

- downloads NWIS site/series catalogs for seven states;
- selects wells with long daily records;
- fetches daily values from the modern USGS OGC API;
- builds a complete daily panel with an observed/missing mask;
- evaluates persistence, seasonal-naive, per-site ridge, and global residual
  GRU baselines on aligned anchors;
- runs jump/gap QC checks, missing-data stress tests, paired site-level
  bootstrap intervals, a fusion ablation, and cross-panel sensitivity
  analyses.

The benchmark contains 279 wells (CA, NE, TX) for parameter 72019 (depth to
water) with statistic 00001, at least 3,650 observations, a 10-year span, and
a record extending past 2015; 74 wells form the clean core used for the main
GRU experiments.

## Important API note

The legacy NWIS `gwlevels` endpoint was decommissioned in November 2025.
`gwbench.usgs` uses the modernized USGS Water Data OGC API:

- `field-measurements` for discrete field measurements;
- `daily` for daily values;
- the legacy NWIS Site Service for site metadata / series catalogs.

The OGC API rate-limits unauthenticated requests (HTTP 429). Two options:

1. Request a free API key at <https://api.waterdata.usgs.gov/signup/> and set
   `USGS_API_KEY` before running;
2. Use `--backend legacy`, which fetches daily values from the legacy `dv`
   endpoint in batches of 20 sites. This is much friendlier to the rate limit
   and is the default for the full download. The legacy endpoint is still
   active as of late 2026 but is not the long-term replacement.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Daily-values series catalog for candidate states
python scripts/01_fetch_sites.py \
  --states CA,TX,NE,KS,FL,AZ,WA \
  --data-type dv \
  --out data/raw/daily_catalog_multi.parquet

# 2. Pilot: 2 wells per state with long daily 72019/00001 records
python scripts/02_fetch_levels.py \
  --sites data/raw/daily_catalog_multi.parquet \
  --limit 10 --per-state 2 \
  --start 2000-01-01 --end 2024-12-31 \
  --parameter-cd 72019 --statistic-id 00001 \
  --min-obs 3650 --min-span-days 3650 --min-last-date 2015-01-01 \
  --skip-gwlevels --backend legacy \
  --cache-dir data/raw/cache_daily_legacy \
  --out data/raw/levels_daily_pilot.parquet

# 3. Complete daily panel with missing mask
python scripts/03_build_panel.py \
  --raw data/raw/levels_daily_pilot.parquet \
  --start 2000-01-01 --end 2024-12-31 \
  --out data/processed/panel_pilot.parquet

# 4. Temporal split + baselines
python scripts/04_baselines.py \
  --panel data/processed/panel_pilot.parquet \
  --train-end 2015-12-31 --val-end 2018-12-31 --split test

# 5. QC: jumps, gaps, value ranges
python scripts/06_qc.py \
  --panel data/processed/panel_pilot.parquet \
  --threshold-ft 5 --out data/processed/qc_pilot.csv

# 6. Quick visual audit
python scripts/05_plot_series.py \
  --panel data/processed/panel_pilot.parquet \
  --out data/processed/panel_pilot.png

# 7. Persistence skill vs forecast horizon
python scripts/08_horizon_baselines.py \
  --panel data/processed/panel_pilot.parquet \
  --horizons 1,7,30,90,180 \
  --out data/processed/horizon_baselines_pilot.csv

# 8. Global GRU smoke test (CPU is enough for the pilot)
python scripts/07_train_gru.py \
  --panel data/processed/panel_pilot.parquet \
  --split-mode temporal --lookback 60 --horizon 7 \
  --stride-train 30 --stride-eval 30 \
  --epochs 2 --batch-size 128 --hidden 16 --layers 1 \
  --out-dir data/processed/gru_smoke

# 9. Ridge evaluated on the exact GRU anchors (memory-bounded for 279 wells)
python scripts/13_ridge_aligned_fast.py \
  --panel data/processed/panel_full.parquet \
  --horizons 30,90 --out data/processed/ridge_aligned_full.csv

# 10. Missing-input stress test, repeated mask draws
for s in 1 2 3 4 5; do
  python scripts/11_missing_stress.py \
    --panel data/processed/panel_core_clean.parquet \
    --model-dir dist/kaggle_results_seeds/gru_h30_seed1 \
    --fractions 0,0.1,0.3,0.5 --stride 7 --mask-seed "$s" \
    --out "data/processed/missing_h30_mask${s}.csv"
done
python scripts/16_aggregate_missing.py

# 11. Paired site-level bootstrap
python scripts/15_bootstrap.py

# 12. Cross-panel persistence / seasonal-naive sensitivity
python scripts/14_panel_sensitivity.py \
  --panel data/processed/panel_full.parquet \
  --out data/processed/panel_sensitivity.csv

# 13. Figures
python scripts/12_make_figures.py
```

USGS requests need network access. In this environment, run the fetch scripts
with the shell sandbox disabled.

## Current datasets (2026-10-27)

| Dataset | Wells | Panel rows | Notes |
|---|---:|---:|---|
| `panel_full.parquet` | 279 | 2,547,828 | all selected daily 72019/00001 wells |
| `panel_core.parquet` | 89 | 812,748 | observed coverage >= 70% |
| `panel_core_clean.parquet` | 74 | 675,768 | core minus wells with >5 large jumps |

The full download contains 1,581,853 observed daily values from 2000-2024
across California, Nebraska, and Texas. Sentinel values (100000.x) were removed
before panel construction.

See `docs/BENCHMARK.md` for the full protocol, data dictionary, QC flow, and
reproduction commands, and `docs/site_manifest.csv` for the per-well manifest
(state, coverage, jumps, gaps, and panel membership).

## Final benchmark results (clean core, seeds 1-3)

| Horizon | Model | NSE mean | NSE median | KGE mean | KGE median |
|---|---|---:|---:|---:|---:|
| 30 d | Persistence | 0.573 | 0.729 | 0.784 | 0.823 |
| 30 d | Ridge | 0.280 | 0.773 | 0.701 | 0.853 |
| 30 d | Calibrated GRU | 0.587 | 0.768 | 0.766 | 0.833 |
| 90 d | Persistence | -0.218 | -0.181 | 0.456 | 0.463 |
| 90 d | Ridge | -1.713 | 0.339 | 0.402 | 0.632 |
| 90 d | Calibrated GRU | -0.244 | 0.210 | 0.519 | 0.573 |

Key findings: persistence is statistically tied with the calibrated GRU on
mean NSE at 30 days; at 90 days the GRU improves median NSE but ridge has an
even higher median; ridge has much heavier mean-error tails, so the GRU's
independent contribution is tail robustness, not median supremacy. A fixed
global blend is worse than persistence; site-specific calibration is
essential. See `paper/main.pdf` for the complete analysis.

> The historical pilot notes below predate the final 279-well benchmark and
> are retained only as a development log. Use the final results above and in
> the manuscript.

## Pilot findings (2026-10-27)

Persistence skill decays sharply with the forecast horizon (six pilot wells):

| Horizon | Persistence NSE (mean) | Persistence KGE (mean) |
|---:|---:|---:|
| 1 day | 0.987 | 0.993 |
| 7 days | 0.888 | 0.941 |
| 30 days | 0.557 | 0.779 |
| 90 days | -0.362 | 0.373 |
| 180 days | -1.462 | -0.018 |

This is the core benchmark motivation: 1-day forecasting is nearly solved by
persistence, while 30-90 day forecasting is where learned models can add value.
Seasonal naive is worse than persistence at every horizon.

The same pattern holds on the 89-well core panel:

| Horizon | Persistence NSE (mean) | Persistence KGE (mean) |
|---:|---:|---:|
| 1 day | 0.973 | 0.985 |
| 7 days | 0.857 | 0.923 |
| 30 days | 0.571 | 0.771 |
| 90 days | -0.083 | 0.460 |
| 180 days | -1.066 | 0.134 |

### Learned baseline

A per-site ridge regression with lag/rolling features (no linear time trend,
predictions unclipped) gives a more nuanced picture on the 74-well clean core:

| Horizon | Ridge NSE median | Persistence NSE median | Ridge NSE mean | Persistence NSE mean |
|---:|---:|---:|---:|---:|
| 30 days | 0.823 | 0.739 | 0.236 | 0.608 |
| 90 days | 0.381 | -0.051 | -1.486 | -0.026 |

The median site improves at both horizons, but a few sites with strong trends
produce large errors that dominate the mean. This is a useful benchmark lesson:
report both mean and median, and treat outlier behavior explicitly. The GRU
experiments should be judged against both.

### 30-day GRU result (Kaggle, v5)

The global GRU uses a residual target (prediction relative to the last
observation) and a per-site persistence blending weight fitted on validation
(2016-2018) and applied to test (2019-2024). 74 clean-core wells, 30-day
horizon:

| Model | NSE mean | NSE median | KGE mean | KGE median | RMSE mean |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.573 | 0.729 | 0.784 | 0.822 | 1.759 |
| Raw residual GRU (seeds 1-3) | -3.155 | 0.564 | 0.443 | 0.761 | 2.164 |
| Calibrated GRU (seeds 1-3) | **0.575** | **0.765** | 0.761 | **0.838** | **1.702** |

The raw GRU still fails catastrophically on a few low-variance wells. The
validation-fitted blending weight fixes this, and the seed-42 run looked
positive. Three additional seeds show that the 30-day gain is fragile:

| Across seeds 1-3 | Calibrated GRU | Persistence |
|---|---:|---:|
| NSE mean | 0.575 ± 0.012 | 0.573 |
| NSE median | 0.765 ± 0.010 | 0.729 |
| KGE mean | 0.761 ± 0.007 | 0.784 |
| KGE median | 0.838 ± 0.009 | 0.822 |
| NSE win rate | 43.3 ± 1.5 / 74 | — |
| KGE win rate | 29.0 ± 3.6 / 74 | — |

A paired bootstrap over sites gives a 95% CI for mean ΔNSE of [-0.077, 0.052]:
at 30 days the calibrated GRU is essentially tied with persistence on mean
NSE, and its mean KGE is slightly worse. The median improves, but the 30-day
headline should be "no robust mean improvement". The failure of the
uncalibrated model remains a useful benchmark finding.

### 90-day GRU result (Kaggle)

Same configuration, horizon 90. Persistence is much weaker at this horizon,
and the calibrated residual model improves on every aggregate metric:

| Model | NSE mean | NSE median | KGE mean | KGE median | RMSE mean |
|---|---:|---:|---:|---:|---:|
| Persistence | -0.218 | -0.181 | 0.456 | 0.462 | 3.772 |
| Raw residual GRU (seeds 1-3) | -14.169 | -0.013 | 0.086 | 0.492 | 4.102 |
| Calibrated GRU (seeds 1-3) | **-0.189** | **0.267** | **0.505** | **0.585** | **3.491** |

- Calibrated NSE improves at about **50/73** wells and KGE at about **41/73**
  (seed means).
- The blending weight is positive for about **63/74** wells on average (range
  58-67); the median alpha ranges from 0.67 to 0.80 across seeds.
- One site (`413156098591201`) dominates the mean: its calibrated NSE ranges
  from -13.27 to -17.33 across seeds while persistence is -0.71. Excluding
  that single site, the seed-averaged calibrated NSE mean is **+0.013** vs
  -0.211 for persistence (individual seeds range -0.057 to +0.068), and KGE
  mean is 0.524 vs 0.457. This is a validation-to-test distribution-shift case
  worth a discussion paragraph.
- Absolute NSE near zero reflects how hard 90-day groundwater forecasting is;
  the benchmark's value is the relative improvement over strong baselines, not
  a high absolute score.

Seed variance for the 90-day result:

| Across seeds 1-3 | Calibrated GRU | Persistence |
|---|---:|---:|
| NSE mean | -0.189 ± 0.093 | -0.218 |
| NSE median | 0.267 ± 0.022 | -0.181 |
| KGE mean | 0.505 ± 0.014 | 0.456 |
| KGE median | 0.585 ± 0.021 | 0.462 |
| NSE win rate | 49.7 ± 3.8 / 73 | — |
| KGE win rate | 41.0 ± 3.6 / 73 | — |

The mean-NSE bootstrap CI crosses zero because one site dominates the mean;
the median NSE and both KGE aggregates are the robust 90-day result. This is
the headline for the paper: at 90 days, a residual GRU with persistence
blending improves the median site and KGE over persistence, but mean NSE is
sensitive to a small number of distribution-shift wells.

### Missing-data robustness

We evaluated the purged models with 0/10/30/50% of the input observations
randomly hidden (no retraining), using five mask seeds:

| Horizon | Masked input | NSE mean | NSE median | KGE mean |
|---|---:|---:|---:|---:|
| 30 days | 0% | 0.637 | 0.780 | 0.774 |
| 30 days | 50% | 0.572 | 0.727 | 0.775 |
| 90 days | 0% | -0.031 | 0.207 | 0.535 |
| 90 days | 50% | 0.035 | 0.265 | 0.559 |

The stress test uses every 7th test anchor, so the absolute values differ
from the full test-set metrics. The response is horizon-dependent: the 30-day
model degrades at 50% point masking and falls slightly below persistence, while
the 90-day model does not degrade. A separate contiguous-block test removes
one 30/60/90-day block per input window: at 30 days mean NSE falls from 0.637
to 0.613 at a 90-day block (median stays near 0.779); at 90 days mean NSE
falls from -0.031 to -0.091 but remains above persistence (-0.221), and KGE is
stable. The random-masking experiment is not a substitute for real multi-week
operational gaps.

- One well (`315712106361803`) has 1,095 day-to-day jumps > 5 ft, consistent
  with pumping or a noisy sensor. It should be excluded or analyzed separately.
- Several series show step-like changes; the QC module flags large short-gap
  jumps without silently correcting them.
- A 2-epoch GRU smoke test is far below persistence at 7 days, as expected; the
  full training run needs the complete panel, longer lookback, and tuning.
- Leave-site-out runs currently use a learned site embedding, so held-out sites
  receive an untrained embedding. Treat those results as a lower bound and use
  `--no-site-embedding` (plus future static features) for transfer experiments.

## Manuscript draft

- Outline and novelty statement: `docs/paper_outline.md`
- LaTeX source: `paper/main.tex`, bibliography: `paper/refs.bib`
- Compiled draft: `paper/main.pdf`
- Figures: `paper/figures/`
- GitHub/Zenodo/Kaggle release guide: `docs/release_guide.md`
- Zenodo-ready archive: `dist/gwbench-v0.1.0.zip`

## Layout

```
gwbench/
  usgs.py          NWIS catalogs + OGC field-measurements/daily clients
  select.py        Long-record well selection
  preprocess.py    QC, daily aggregation, missing masks
  panel.py         Daily panel, temporal/site splits, baselines, evaluation
  qc.py            Jump/gap/flat-period checks
  metrics.py       RMSE, MAE, NSE, KGE, MASE, naive forecasts
scripts/           Numbered pipeline steps
data/raw/          Downloaded data (ignored by git)
data/interim/      Intermediate tables (ignored by git)
data/processed/    Model-ready panels and results (ignored by git)
```

## Next milestone

1. Run the cross-site (leave-well-out) experiment with `--no-site-embedding`
   and static site features.
2. Add the ridge baseline to the same tables and finalize the benchmark
   comparison.
3. Write the C&G paper around the 90-day median/KGE result, the persistence
   baseline, the uncalibrated-GRU failure, and the missing-data robustness.
