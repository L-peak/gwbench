# Response to Second-Round Review

We thank the reviewer for the detailed and demanding report. The revision
addresses every P0 item with new experiments and every P1 item with new
analyses and documentation. The main changes are: (i) a single statistical
narrative anchored on the 90-day median NSE, (ii) a five-seed missing-data
experiment at 0/10/30/50% masking, (iii) a controlled fusion ablation with
raw, fixed-weight, and per-site calibration, (iv) per-site and cross-panel
analyses, (v) a reproducibility appendix, and (vi) a purged temporal split
with full retraining of all six models. We also report a result that
strengthens the benchmark's honesty: a simple per-site ridge baseline has a
higher median NSE than the GRU at both horizons, while the GRU has much better
mean errors. The GRU's independent contribution is a reduction in the most
severe failures relative to the raw GRU and ridge, not median supremacy;
persistence remains the most conservative tail baseline.

Table and figure numbers below refer to the revised manuscript.

## P0. Data leakage at the temporal split boundaries

We agree with the reviewer that the previous anchor-date split was not
sufficient. The last h validation anchors had targets in the test period, and
those targets were used for early stopping and for fitting the per-site
blending weights. The test predictions themselves were generated only on test
anchors, but model selection and calibration could still be influenced by
early test-period targets. This was a genuine leakage risk.

The revision now uses a purged temporal split:

- a training anchor is retained only if its target date satisfies
  t + h <= 2015-12-31;
- a validation anchor is retained only if t + h <= 2018-12-31;
- test anchors satisfy t >= 2019-01-01, so every test target is in the test
  period.

This removed 2,017 training and 2,122 validation anchors at 30 days, and 6,048
training and 6,328 validation anchors at 90 days. We retrained all six models
(30/90 days x seeds 1-3) on a Kaggle GPU with the purged protocol, then reran
early stopping, per-site calibration, test evaluation, missing-data stress,
fusion ablation, bootstrap intervals, state-level analysis, and all figures.
The purged run is scripted in `scripts/07_train_gru.py --purge-targets` and
`notebooks/kaggle_purged/`.

The main conclusions survive, with one important correction: at 90 days the
mean NSE is slightly worse than persistence, not slightly better.

- 30 days: calibrated mean NSE 0.587 +/- 0.007 vs 0.573 for persistence;
  paired mean Delta-NSE +0.014 (95% CI -0.056 to +0.059); paired median
  Delta-NSE +0.006 (95% CI 0.000 to 0.022). This remains a statistical tie.
- 90 days: median NSE 0.210 +/- 0.039 vs -0.181; paired median Delta-NSE
  +0.107 (95% CI 0.048 to 0.225). Mean KGE 0.519 +/- 0.004 vs 0.456; paired
  mean Delta-KGE +0.063 (95% CI 0.010 to 0.109) and median +0.023 (95% CI
  0.010 to 0.076). Mean NSE is -0.244 +/- 0.008 vs -0.218; paired mean
  Delta-NSE -0.027 (95% CI -0.531 to +0.262). Excluding the single
  distribution-shift well, mean NSE is -0.025 vs -0.211.
- Fusion ablation: at 30 days fixed alpha gives mean/median NSE -0.325/0.658
  vs per-site 0.587/0.768; at 90 days fixed alpha gives -4.661/0.070 vs
  per-site -0.244/0.210. A global weight still fails.
- Ridge: median NSE 0.773/0.339 vs calibrated GRU 0.768/0.210; mean NSE
  0.280/-1.713 vs 0.587/-0.244. Ridge remains better in the median and much
  worse in the mean.

The boundary-overlap limitation has been removed from the manuscript and
replaced by the purged protocol in Section 3.1 and Appendix A.

## P0.1 Unified statistical claims and abstract

The abstract, Results, Discussion, and Conclusions now use the same endpoint
and the same wording:

- **30 days:** the calibrated GRU is statistically tied with persistence on
  mean NSE. The paired site-level mean Delta-NSE is +0.014 (95% CI -0.056 to
  +0.059); the paired median is +0.006 (95% CI 0.000 to 0.022). Mean KGE is
  slightly lower (-0.018, 95% CI -0.043 to 0.000). We no longer describe this
  as an ML win.
- **90 days:** the robust results are median NSE and KGE. The paired median
  Delta-NSE is +0.107 (95% CI 0.048 to 0.225), with 51/73 wells improving, 18
  worsening, and 4 unchanged. Mean NSE is -0.244 vs -0.218 for persistence;
  paired mean Delta-NSE is -0.027 (95% CI -0.531 to +0.262). Mean Delta-KGE
  is +0.063 (95% CI 0.010 to 0.109), and median Delta-KGE is +0.023 (95% CI
  0.010 to 0.076). The mean NSE interval includes zero and one
  distribution-shift well dominates it; the text says so explicitly. Excluding
  that well, mean NSE is -0.025 vs -0.211.
- **Ridge comparison:** a per-site ridge baseline has higher median NSE and
  KGE than the GRU at both horizons, but much worse mean errors. This is now
  stated in the abstract, Section 4.3, the Discussion, and the Conclusions.
  The benchmark's conclusion is not "the GRU wins"; it is that the calibrated
  GRU reduces the most severe failures relative to the raw GRU and ridge,
  while persistence remains the most conservative tail baseline. Reporting
  only a mean or only a median conceals the trade-off. Quantitatively, at 90
  days the number of wells with NSE < -1 is 9 (calibrated GRU), 7
  (persistence), 11 (ridge), and 25 (raw GRU); minimum NSE is -16.0, -3.5,
  -37.9, and -228.4, respectively. At 30 days the counts are 2, 1, 6, and 9.

The confidence intervals are regenerated by a documented paired site-level
bootstrap (wells resampled, 10,000 replicates, fixed seed 20261029). Metrics
are averaged over seeds before resampling. The Methods now state that the
median of paired differences is not the difference of marginal medians.

## P0.2 Complete missing-data experiment

We added the requested 0/10/30/50% experiment with repeated mask draws
(Table 5, Figure 6). Each fraction is evaluated with five mask seeds (1-5) on
the stride-7 test subset, using the seed-1 model without retraining. Values
are mean +/- standard deviation across mask draws.

- 30 days: mean NSE 0.637 -> 0.572; median NSE 0.780 -> 0.727; mean KGE
  0.774 -> 0.775. At 50% masking the model is slightly below persistence
  (0.615/0.744/0.786), so the 30-day result is not robust to severe input
  missingness.
- 90 days: mean NSE -0.031 -> 0.035; median NSE 0.207 -> 0.265; mean KGE
  0.535 -> 0.559. Masking does not degrade performance at this horizon.
- The mask-to-mask standard deviation is at most 0.004 in mean NSE and 0.002
  in mean KGE.

The table caption states that the stress test uses the seed-1 model and that
the stride-7 subset is not directly comparable to the full test set. The text
also distinguishes random isolated masking from real multi-week gaps; we do
not claim robustness to all missing-data mechanisms.

## P0.3 Baselines and fusion ablation

**Baselines.** Table 3 now includes persistence, seasonal naive, per-site
ridge, raw GRU, and calibrated GRU on exactly the same anchors. Seasonal naive
is far worse than persistence (30-day mean NSE -2.995; 90-day -3.437). Ridge
has median NSE 0.773/0.339 at 30/90 days, higher than the calibrated GRU's
0.768/0.210, but its mean NSE is 0.280/-1.713, far worse than the GRU's
0.587/-0.244. This is a central benchmark finding, not a footnote.

**Fusion ablation.** Table 4 compares four strategies on identical anchors:
persistence (alpha=0), raw GRU (alpha=1), a fixed global alpha equal to the
median validation-fitted per-site weight for each seed, and the per-site
alpha. At 30 days, fixed alpha gives mean/median NSE -0.325/0.658, both worse
than persistence (0.573/0.729) and far worse than per-site alpha
(0.587/0.768). At 90 days, fixed alpha gives -4.661/0.070, versus -0.244/0.210
for per-site alpha. A single global weight cannot down-weight the low-variance
wells that dominate the mean; site-specific calibration is the essential
ingredient. The fixed weights themselves range from 0.62 to 0.69 at 30 days
and 0.64 to 0.73 at 90 days across seeds.

## P1.1 Per-site and cross-panel analyses

**Per-site.** Figure 8 shows persistence NSE versus calibrated-GRU NSE for
30 and 90 days (symlog axes, colored by state) and the distribution of
Delta-NSE. At 30 days, 47 wells improve, 25 worsen, and 2 are unchanged. At
90 days, 51 improve, 18 worsen, and 4 are unchanged; one well has NSE -16.0
and is annotated. The figure makes clear that the 90-day result is a
distributional improvement, not a uniform gain.

**Cross-panel.** Table 7 re-evaluates persistence and ridge on the full
(246/245 wells with sufficient anchors), screened-core (89/88), and clean-core
(74/73) panels. At 90 days, persistence mean NSE is -3.051 (full), -0.271
(core), and -0.218 (clean); median NSE is -0.264, -0.418, and -0.181. Ridge
median NSE is 0.092 (full), 0.263 (core), and 0.339 (clean), while its mean is
-26.491, -1.579, and -1.713. The clean core has the strongest persistence
baseline, so the clean-core comparison is conservative for a learned model,
not an artificially easy subset. The GRU was trained only on the clean core;
we explicitly do not claim cross-panel GRU comparability.

## P1.2 Reproducibility

Appendix A now documents:

- data retrieval (USGS OGC endpoints, parameter 72019, statistic 00001,
  period, selection thresholds, QC rules, panel counts);
- anchor construction (365-day window, target index t+h, minimum 30 observed
  inputs, missing targets excluded, stride 3 for training and 1 for
  validation/test, purged target boundaries, 116,415/76,493 training and
  validation anchors at 30 days and 114,941/72,462 at 90 days, and 103,956/
  99,750 test anchors at 30/90 days);
- model hyperparameters (two GRU layers, hidden 64, dropout 0.2, Adam 1e-3,
  batch 256, MSE, gradient clipping 1.0, 40 epochs, patience 5);
- calibration (validation-only, clipped least squares, alpha=0 fallback);
- bootstrap protocol (10,000 well-level resamples, seed 20261029);
- missing-mask protocol (five mask seeds, stride 7, no retraining);
- software and dependency versions.

`requirements.txt` now includes PyTorch and scikit-learn. The new analysis
scripts are in the repository: `scripts/13_ridge_aligned_fast.py`,
`scripts/14_panel_sensitivity.py`, `scripts/15_bootstrap.py`,
`scripts/16_aggregate_missing.py`, `scripts/17_fusion_ablation.py`,
`scripts/18_tail_analysis.py`, and `scripts/19_purged_calibration.py`. The
purged retraining kernel is in `notebooks/kaggle_purged/`.

The boundary-overlap limitation has been removed. Training and validation
anchors are purged at the target-date boundary, so no early-stopping or
calibration target belongs to a later period.

## P2 Contributions and related work

The Introduction now has a related-work paragraph positioning the work against
groundwater ML reviews and large-sample hydrology benchmarks, and a rewritten
four-contribution list: (i) open benchmark and panels, (ii) aligned
persistence/seasonal/ridge baselines, (iii) controlled fusion ablation
isolating site-specific calibration, and (iv) robust statistics, repeated
missingness, and cross-panel sensitivity. The text explicitly says that
residual forecasting and persistence blending are known techniques; the
novelty is the benchmark protocol and the empirical characterization of
failure modes.

## Specific concerns

- **Anchor definitions:** now explicit in Section 3.1 (window, target index,
  minimum history, missing-target rule, strides, counts).
- **Missing-data table:** now 0/10/30/50% with five repeats and uncertainty.
- **Baselines:** seasonal naive and ridge added to Table 3.
- **Statistical inference:** paired well-level bootstrap, fixed seed, both
  mean and median; marginal-versus-paired median distinction stated.
- **NSE/KGE interpretation:** Methods now explain that NSE is sensitive to
  low-variance wells and KGE balances correlation, bias, and variability.
- **Formula clarity:** y_last is defined; the calibration equation and
  alpha=0 fallback are stated.
- **Per-site figure:** added as Figure 8.
- **Site embeddings / unseen wells:** unchanged limitation, now stated more
  prominently in Section 5.2 and Section 5.4.
- **References:** the related-work citations are retained and all DOIs were
  checked previously.

## Note on new experiments

No new USGS data were downloaded. All new analyses use the existing processed
panels and the six purged retrained models (30/90 days x seeds 1-3). The
missing stress test adds five mask seeds; the cross-panel ridge uses a
vectorized per-site implementation. The full-panel ridge result is reported
with 246/245 wells because the remaining wells lack sufficient training or
test anchors; this is stated in the table caption.
