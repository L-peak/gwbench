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
- At 90 days, the calibrated model improves median NSE from -0.18 to 0.21
  (paired median gain 0.107, 95% CI 0.048-0.225) and mean KGE from 0.46 to
  0.52 (95% CI 0.010-0.109); mean NSE is -0.24 vs -0.22 and its interval
  includes zero because one well dominates the mean.
- A simple per-site ridge baseline has a higher median NSE than the GRU at
  both horizons, but much heavier mean-error tails. The GRU's independent
  contribution is fewer severe failures than the raw GRU and ridge, not
  median supremacy; persistence remains the most conservative tail baseline.
- A fixed global blending weight is worse than persistence; a per-site
  validation-fitted weight is essential.
- Missing-input robustness is horizon-dependent: at 30 days, 50% masking
  pushes mean NSE below persistence; at 90 days, masking does not degrade
  performance (five repeated mask draws).

We believe this fits Computers & Geosciences because it is a benchmark and
software contribution rather than another architecture paper: it provides an
open pipeline, purged temporal splits, quality-control code, and an
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
