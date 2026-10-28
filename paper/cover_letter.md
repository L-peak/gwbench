Dear Editors,

Please consider our manuscript, "When Does Deep Learning Beat Persistence
for Groundwater-Level Forecasting? A Reproducible Benchmark and a
Persistence-Anchored Residual Model," for publication in Computers &
Geosciences.

The manuscript contributes a reproducible, public-data benchmark for daily
groundwater-level forecasting built from USGS NWIS records (279 wells,
2000-2024), a strong baseline ladder, and a careful evaluation of when a
global residual GRU actually beats persistence. The main findings are:

- Persistence is extremely strong at short horizons (mean NSE 0.97 at one
  day); a calibrated residual GRU is statistically tied with persistence at
  30 days.
- At 90 days, the calibrated model improves median NSE from -0.18 to 0.27
  and mean KGE from 0.46 to 0.51 across three seeds.
- A per-site validation-fitted persistence blending weight is essential:
  the uncalibrated GRU fails catastrophically on low-variance wells.
- The calibrated model is stable when 10-50% of input observations are
  masked.

We believe this fits Computers & Geosciences because it is a benchmark and
software contribution rather than another architecture paper: it provides an
open pipeline, standardized temporal splits, quality-control code, and an
honest negative result about short-horizon deep learning. The code and
processed data are available at https://github.com/L-peak/gwbench and archived
at https://doi.org/10.5281/zenodo.23749304. Raw observations are public USGS
data.

The manuscript is original, is not under consideration elsewhere, and all
authors have approved the submission. We declare no competing interests.

Sincerely,
Zhaoyang Li
Henan Polytechnic University
3286533389@qq.com
