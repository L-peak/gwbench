# Response to Reviewers

We thank the reviewers for their careful and constructive reports. The revision
substantially sharpens the positioning of the paper, aligns the baselines,
adds statistical uncertainty and heterogeneity analyses, and improves the
presentation. Below we respond point by point. Section, table, and figure
numbers refer to the revised manuscript.

## Summary of major changes

1. **Positioning.** The manuscript is now explicitly framed as a benchmark and
   reproducibility contribution, not a new architecture. The Introduction states
   that residual forecasting and persistence blending are known techniques and
   that the novelty is the benchmark protocol plus the empirical characterization
   of failure modes. Section 5.1 adds a concrete protocol for future models.
2. **Evaluation alignment.** The ridge baseline is now evaluated on exactly the
   same anchors as the GRU (Section 4.3). We also quantify the effect of the
   training/test stride mismatch (Section 4.6) and report bootstrap confidence
   intervals for mean and median differences (Sections 4.2 and 4.4).
3. **Coverage and heterogeneity.** A state-level table (Table 5, Section 4.7)
   shows that gains are not uniform across California, Nebraska, and Texas.
   Per-site win/loss counts and delta-NSE quantiles are reported in Sections 4.2
   and 4.4.
4. **Outlier analysis.** A new figure (Figure 7, Section 4.8) shows the
   distribution-shift well, its seasonal pumping-like spikes, and the failure of
   the calibrated model in the available test period. We explicitly state that
   pumping, drought, and sensor/datum changes cannot be separated without
   pumping records.
5. **Presentation.** The abstract and conclusions now state that mean NSE and
   mean KGE confidence intervals include zero and that the median NSE gain is
   the robust result. Ligatures were disabled (microtype) so that words such as
   "difficult" and "different" extract correctly. Figure 4 now uses density
   hexbin panels. Table 1 distinguishes the full, screened-core, and clean-core
   panels, and the main text states that the clean core is the primary set.
6. **References.** We added two recent groundwater ML references
   \citep{khan2023,dey2025} and verified all DOIs against Crossref.

## Reviewer 1

**1. Limited novelty relative to journal expectations.**
We agree that the main contribution is not a new algorithm. The revision
therefore (i) states this explicitly in the Introduction, (ii) adds Section 5.1
with a recommended evaluation protocol for future models, and (iii) positions
the paper as a community benchmark resource. We also added state-level and
per-site heterogeneity analyses to make the empirical contribution more
substantial.

**2. Limited benchmark coverage.**
We acknowledge the three-state, single-parameter scope and now state it more
prominently in the Limitations. To partially address it, Table 5 and Section
4.7 report performance separately for California (32 wells), Nebraska (37/36
wells at 30/90 days), and Texas (5 wells). The results show that the aggregate
finding is not driven by a single state, but also that gains are heterogeneous.
Cross-site transfer and climate covariates are explicitly discussed in Sections
5.2 and 5.3 and left for future work.

**3. Evaluation design.**
- *Anchor protocol:* Section 4.6 quantifies the effect of the stride mismatch.
  Re-evaluating at stride 3 changes mean NSE from 0.575 to 0.606 at 30 days and
  from -0.189 to -0.197 at 90 days; median and KGE differences are below 0.01.
- *Ridge alignment:* The ridge baseline is now evaluated on exactly the same
  anchors as the GRU (Section 4.3). Its median NSE is 0.780 at 30 days and 0.304
  at 90 days; its mean NSE is 0.264 and -1.688, respectively.
- *90-day outlier:* We now report bootstrap intervals for mean and median
  differences (Section 4.4) and state that the median NSE gain is the robust
  result. We also report the outlier well's forecasts in Figure 7 and discuss
  the ambiguity of the physical cause.
- *Win rates:* Section 4.4 reports 55/73 positive, 13 negative, and 5 zero
  delta-NSE wells, with the median delta +0.115 (95% CI 0.033-0.207) and the
  mean +0.029 (95% CI -0.431-0.299). The state table provides further
  heterogeneity.

**4. Statistical reporting.**
We added paired bootstrap confidence intervals for mean and median delta-NSE
and delta-KGE at 30 and 90 days (Sections 4.2 and 4.4). We retained three seeds
because of the GPU budget but now report seed variance, per-site distributions,
and confidence intervals; additional seeds are a natural extension.

**5. Presentation and writing.**
The abstract and conclusions were rewritten to avoid overemphasizing the
90-day point estimates. Figure 4 now uses density hexbins; Figure 7 adds the
outlier case. Ligatures were disabled, and the manuscript was recompiled and
checked for text-extraction artifacts. Table 1 and the Data section now make
the hierarchy between the 279-well full panel, the 89-well screened core, and
the 74-well clean core explicit.

**Specific comments.**
- Table 1 now labels the panels "Full", "Core (screened)", and "Clean core
  (main)", and the caption states that the clean core is the primary analysis
  set.
- Equation (1) now explains why site-wise z-scoring is preferred to a global
  scale.
- The missing-data table caption and the first sentence of Section 4.6 state
  that the results use a stride-7 subset and are not directly comparable to
  Table 3.
- The Zenodo DOI is displayed as `10.5281/zenodo.23749304` without a line break.
- Two recent references were added and all DOIs were checked.

## Reviewer 2

**1. Site embeddings and unseen wells.**
Section 5.2 now explicitly states that the current model is evaluated on wells
seen during training, that the embedding is undefined for unseen wells, and
that the present results are not a leave-well-out transfer test. A proper
cross-site experiment requires static predictors (well depth, aquifer, land
cover, climate) and is left for future work; the repository includes the split
machinery.

**2. Climate covariates.**
Section 5.3 now explains that the univariate model cannot represent recharge,
pumping, evapotranspiration, or storage changes, that these omissions likely
limit 90-day absolute skill, and that precipitation, PET, and GRACE-derived
storage are the most promising next steps. We also state that the reported
gains come from persistence anchoring and validation calibration, not from
external drivers.

**3. Outlier case.**
Figure 7 and Section 4.8 present the outlier well's full record and 90-day
forecasts. The well shows strong seasonal spikes consistent with pumping and a
multi-year decline. We state that without pumping/abstraction records we cannot
attribute the shift to a specific mechanism; pumping changes, drought, and
sensor or datum changes are all plausible.

**4. Typography and figures.**
Ligatures were disabled with microtype, Figure 4 was changed to hexbin density
plots, and the revised PDF was checked for legibility. Figure 7 was added at
300 dpi.

## Reviewer 3

The revision addresses the same major points: explicit benchmark positioning,
state-level and per-site heterogeneity, aligned ridge evaluation, stride
sensitivity, bootstrap confidence intervals, cautious abstract/conclusion
wording, and the outlier case. The manuscript now reports 90-day median NSE
0.267 (95% CI for the median gain 0.033-0.207) and explains why the mean is
sensitive to one well. The new Section 5.1 gives a concrete protocol for future
benchmark users.

## Note on new experiments

No new USGS data were downloaded. All new analyses use the existing processed
panels and saved predictions: the aligned ridge baseline, the stride
sensitivity analysis, the state-level table, the bootstrap intervals, and the
outlier figure. The three-seed experiments remain the main multi-seed evidence;
additional seeds are left for future work.
