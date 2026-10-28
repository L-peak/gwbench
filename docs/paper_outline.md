# gwbench 论文提纲

**Working title:** When Does Deep Learning Beat Persistence for Groundwater-Level Forecasting? A Reproducible Benchmark and a Persistence-Anchored Residual Model

**Target journal:** Computers & Geosciences (Elsevier, subscription route, no APC)
**Backup:** Environmental Modelling & Software; Journal of Hydrology

## 0. Novelty statement (honest positioning)

**What is genuinely new**

1. A reproducible, public-data benchmark for daily groundwater-level
   forecasting with temporal splits, QC, missing-data stress tests, and a
   strong baseline ladder. Most comparable papers use private data or random
   splits and report only one horizon.
2. An empirical finding that a global residual GRU with site embeddings still
   fails catastrophically on low-variance wells, and that a per-site
   validation-fitted persistence blending weight fixes this failure.
3. A horizon-dependent conclusion that is usually missing from the literature:
   at 30 days the calibrated model is statistically tied with persistence on
   mean NSE; at 90 days it improves median NSE, KGE mean, and KGE median across
   seeds. The negative 30-day result is itself a contribution.
4. A missing-data robustness result: the residual + blending model does not
   collapse when 10-50% of input observations are hidden.

**What is not new**

- Residual/delta forecasting, persistence blending, GRU architectures, site
  embeddings, NSE/KGE, and USGS NWIS data itself.
- The per-site calibration is a simple closed-form blend, not a new learning
  algorithm.
- The benchmark covers only three states and one parameter/statistic; it is not
  a universal groundwater benchmark.

**Positioning**

This is a benchmark / reproducibility / empirical-insight paper, not a new
deep-learning method paper. The contribution is the combination of an open
toolkit, honest evaluation, and practical guidance ("anchor to persistence,
calibrate per site, report median and KGE"). It fits C&G and EMS; it is not
positioned for WRR/HESS without cross-site transfer, climate covariates, and
stronger physical baselines.

## 1. Motivation

- Groundwater-level forecasting papers increasingly report deep-learning gains,
  but evaluation protocols vary: random splits, different wells, different
  horizons, weak baselines.
- Persistence is extremely strong at short horizons (1 day NSE ~0.97), so many
  reported gains may be baseline artifacts.
- Missing data and datum/sensor jumps are common in USGS records and are rarely
  handled explicitly.
- Need: a reproducible, public-data benchmark with strong baselines, honest
  splits, and robustness tests.

## 2. Contributions

1. **Open benchmark**: 279 USGS wells with daily depth-to-water (parameter
   72019, statistic 00001), 2000-2024, plus a 74-well clean core; standardized
   temporal and missingness splits.
2. **Strong baseline ladder**: persistence, seasonal naive, per-site ridge with
   lag/rolling features, and a global residual GRU.
3. **Residual + persistence blending**: a per-site validation-fitted blending
   weight that prevents catastrophic failures on low-variance wells.
4. **Honest horizon-dependent conclusion**: at 30 days the calibrated model is
   statistically tied with persistence on mean NSE; at 90 days it improves
   median NSE, KGE mean, and KGE median across seeds.
5. **Robustness**: model performance is stable under 10-50% random input
   missingness, and the benchmark exposes datum/sensor jumps and sentinel values.
6. **Open-source toolkit**: `gwbench` package, cached download pipeline, Kaggle
   notebooks, seeds, and evaluation scripts.

## 3. Data

### 3.1 Source

- USGS NWIS site/series catalog; daily values from the legacy `dv` endpoint
  (batched) with support for the modern OGC `daily` API.
- Parameter 72019 (depth to water below land surface), statistic 00001
  (daily maximum).
- States: California, Nebraska, Texas (the states with the most long daily
  records in the initial catalog scan).
- 279 wells, 1,581,469 observed daily values, 2000-2024.

### 3.2 Panels

| Panel | Wells | Rows | Rule |
|---|---:|---:|---|
| full | 279 | 2,547,828 | all selected wells |
| core | 89 | 812,748 | observed coverage >= 70% |
| clean core | 74 | 675,768 | core minus wells with >5 jumps >5 ft |

### 3.3 QC

- Remove sentinel values (100000.x).
- Flag day-to-day jumps >5 ft when the gap is <=2 days.
- Flag long gaps and flat periods.
- Do not silently correct jumps; exclude the 15 worst wells from the clean core.

## 4. Methods

### 4.1 Splits

- Temporal: train <=2015, validation 2016-2018, test 2019-2024.
- Site split (future work): leave wells/states out; current site-embedding model
  is not a fair transfer test, so report it as a limitation.

### 4.2 Baselines

- Persistence: last observation before the anchor.
- Seasonal naive: value 365 days before the target.
- Per-site ridge: last value, rolling means (7/30/90/365), rolling std,
  day-of-year sin/cos; no linear trend (reduces extrapolation).

### 4.3 Global residual GRU

- Input window 365 days; channels: normalized value, observation mask,
  day-of-year sin/cos; learned site embedding.
- Target: residual from the last observed value in the window.
- 2-layer GRU, hidden 64, dropout 0.2, MSE, Adam, early stopping.
- Horizons 30 and 90 days; stride 3 for training, every day for test.

### 4.4 Persistence blending

- Fit per-site alpha on validation: calibrated = persistence + alpha *
  (GRU - persistence), alpha in [0, 1].
- alpha = 0 means pure persistence; alpha = 1 means pure GRU.
- No test leakage; the same alpha is applied to all missingness levels.

### 4.5 Metrics

- RMSE, MAE, NSE, KGE; report mean and median across wells.
- Win rate: fraction of wells where the model beats persistence.
- Paired bootstrap over wells for mean delta NSE/KGE.
- Multi-seed mean ± std (seeds 1-3; seed 42 exploratory).

## 5. Results

### 5.1 Persistence horizon curve

Core panel: NSE mean 0.973 (1d), 0.857 (7d), 0.571 (30d), -0.083 (90d),
-1.066 (180d). This motivates the 30/90-day focus.

### 5.2 30-day result (negative/marginal)

- Single seed 42: calibrated NSE mean 0.603 vs 0.573.
- Seeds 1-3: 0.575 ± 0.012 vs 0.573; KGE mean 0.761 ± 0.007 vs 0.784.
- Bootstrap 95% CI for mean delta NSE [-0.077, 0.052] -> no robust mean gain.
- Median NSE 0.765 vs 0.729; median KGE 0.838 vs 0.822.
- Conclusion: at 30 days, persistence is essentially unbeaten on mean NSE.

### 5.3 90-day result (main positive result)

- Seeds 1-3: calibrated NSE mean -0.189 ± 0.093 vs -0.218; median 0.267 ±
  0.022 vs -0.181; KGE mean 0.505 ± 0.014 vs 0.456; KGE median 0.585 ± 0.021
  vs 0.462.
- Win rates: 49.7 ± 3.8 / 73 (NSE), 41.0 ± 3.6 / 73 (KGE).
- Single seed 42: NSE mean -0.139, median 0.276; KGE mean 0.520, median 0.569.
- Outlier: site 413156098591201; excluding it, NSE mean +0.048 vs -0.211 and
  KGE mean 0.538 vs 0.457.
- Conclusion: 90-day median and KGE gains are robust; mean NSE is sensitive to
  one distribution-shift well.

### 5.4 Raw vs calibrated GRU

- Raw residual GRU: 30d NSE median 0.480; 90d NSE median 0.009; catastrophic
  failures on low-variance wells (e.g., 30d NSE -198).
- Calibration is essential; alpha >0 for 53/74 (30d) and 60/74 (90d) wells.

### 5.5 Missing-data robustness

- 30d: NSE mean 0.644 -> 0.627 from 0% to 50% masked input; KGE mean 0.777 ->
  0.787.
- 90d: NSE mean 0.044 -> 0.067; KGE mean 0.539 -> 0.562.
- Evaluated every 7th test anchor, no retraining.

### 5.6 QC / data insights

- 279 wells selected; 74 clean-core after coverage and jump filters.
- 15 wells with >5 jumps; one well 1,095 jumps.
- Sentinel values 100000.x in two wells.

## 6. Figures and tables

**Figures**
1. Study map with 279 wells and 74 clean-core wells (color by state).
2. Example hydrographs: clean vs jumpy vs missing-data well.
3. Persistence skill vs horizon (NSE/KGE with confidence bands).
4. 30d/90d predicted-vs-observed scatter for calibrated GRU and persistence.
5. Alpha distribution (30d vs 90d) and per-site delta NSE.
6. Missing-data robustness curves.

**Tables**
1. Dataset summary and panels.
2. Persistence horizon curve.
3. 30d model comparison (raw/calibrated/persistence, mean/median).
4. 90d model comparison.
5. Multi-seed variance and bootstrap CIs.
6. Missing-data stress test.
7. Ridge baseline comparison (optional supplementary).

## 7. Discussion

- Persistence is a strong baseline at short horizons; papers claiming large
  30-day gains should report it and use temporal splits.
- Global deep models without a persistence anchor fail on low-variance wells;
  residual targets + validation blending are a simple fix.
- Median/KGE are more robust than mean NSE under distribution shift; report both.
- Missing-data robustness is a genuine strength of the residual + blending
  approach.
- Benchmark limitations: only three states, one parameter/statistic, no climate
  covariates, no site-transfer solution yet.

## 8. Limitations

- No static site features (well depth, aquifer, land use) in the model yet.
- Site split is not a fair transfer test with site embeddings; future work.
- Calibration alpha is fitted on 2016-2018; one test site shows drift.
- Only daily max depth (72019/00001); other statistics/parameters may behave
  differently.
- USGS records have datum/sensor changes that are flagged, not corrected.

## 9. Reproducibility and data availability

- Code: `gwbench` package + scripts + Kaggle notebooks.
- Data: public USGS NWIS; processed panels and baseline outputs archived with
  the paper (Zenodo DOI planned).
- Seeds, configs, and exact commands included.
- No proprietary data; no APC.

## 10. Reviewer concerns and responses

- **"Why not compare to MODFLOW/physical models?"** The paper is a data-driven
  benchmark; persistence is the relevant strong baseline. Physical models are
  future work.
- **"Only three states."** Selected because they have the longest public daily
  records; the pipeline generalizes to other states.
- **"Mean NSE is negative at 90 days."** Report median/KGE and the outlier
  analysis; the benchmark measures relative skill, not absolute performance.
- **"Calibration is just persistence."** Alpha is fitted per site and positive
  for 60/74 wells at 90 days; the raw GRU is also reported.
- **"Data leakage?"** Temporal split; calibration uses validation only; test
  2019-2024 untouched.

## 11. Timeline

1. Outline + figure plan: done.
2. Draft Methods/Data: 1 week.
3. Draft Results/Discussion: 1 week.
4. Internal review + code/data cleanup: 1 week.
5. Submit to Computers & Geosciences.
