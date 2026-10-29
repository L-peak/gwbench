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
  (paired median gain 0.115, 95% CI 0.033-0.207) and mean KGE from 0.46 to
  0.51; mean NSE and mean KGE intervals include zero because one well
  dominates the mean.
- A simple per-site ridge baseline has a higher median NSE than the GRU at
  both horizons, but much heavier mean-error tails. The GRU's independent
  contribution is tail robustness, not median supremacy.
- A fixed global blending weight is worse than persistence; a per-site
  validation-fitted weight is essential.
- Missing-input results are stable at 0/10/30/50% masking with five repeated
  mask draws.

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
