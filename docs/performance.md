# ** Performance Report **

# M3D_EN - v0.2.5 - 25 October 2026

This report evaluates G-unit detection on selected intervals from five
M3D_EN presentation videos, focusing on shots
with clearly visible hands and upper body. These are exploratory results,
not an independent validation of the release.

The figures below were recomputed from the five reprocessed runs using saved
LLM estimates and this release's candidate-entry and final-grouping rules.
The batch and single-run evaluators now use the same grouped LLM decisions.
No new model calls were made; this is not a new end-to-end evaluation of the
release.

On this selection, the default **LLM-only scoring mode** reaches
**97.7% recall, 91.7% F1 and 65.3% balanced accuracy**, with a **67.1%
false-positive rate**. The always-G-unit baseline reaches 89.7% F1. The system
detects most reference G-unit time and shows discrimination beyond simple
baselines; rejection of non-G-unit time remains the main limitation.

## What was evaluated

Selected intervals from five distinct English presentation videos in M3D_EN
were compared with their reference EAFs. The assessed selection totals
**19 min 8.6 s**. Media and annotations are not distributed with this project. See the
[method and attribution](pipeline.md#method-origins).

All five runs used the visual-language model `Qwen 36-27b`, through Open WebUI.
Phase1 was enabled. Phase2 requested three target samples (300 ms), up to 47
preceding samples (4.7 s) and ten following samples (1 s), with context shortened
where unavailable.

**LLM-only scoring** (`probability_score_source = "llm"`).
The LLM receives both labelled images and motion data. Its non-G-unit estimate
`L` supplies the candidate score directly: `Q = L`. Thus LLM-only describes
score selection, not removal of motion from model input. Local temporal rules
then produce the final intervals, without another model pass.

The metrics below were calculated by replaying local candidate creation and final
grouping from saved LLM estimates, without repeating model calls. No reference-fitted thresholds
were introduced for this report. The data informed development, so the results
remain exploratory. See [the pipeline description](pipeline.md) for the method.

## Assessment scope and metrics

A reference-positive sample has its center inside a `Manual_GUnit` annotation
with value `G-unit`. Scores use the 100 ms analysis grid, not source video
frames. Unlabelled or blank `Spker_vis` intervals are excluded. Detection also
excludes the saved Phase1 exclusions and anything outside the processing range.
Thus detection metrics are conditional on admission by Phase1, not full-video
end-to-end sensitivity.

| Detection scope | Eligible samples | Evaluated time | Reference G-unit prevalence |
| --- | ---: | ---: | ---: |
| M3D_EN selected visible-speaker shots | 11,486 | 19 min 8.6 s | 81.4% |

The visibility selection uses exact, case-sensitive matches in `Spker_vis` to `Half Body Shot`,
`Full Body Shot`, `Close Shot`, `3/4 Body shot`, `Full Body shot`, `Half Body shot`,
`Full shot` and `Mid shot`. These reference shot labels define the reported
scope; for example, `Close shot` does not match `Close Shot`. Selected intervals
can still contain camera cuts or brief occlusions. This report covers detection
within this selection; it does not report Phase1 scene-exclusion performance.

Report **balanced accuracy together with precision, recall, F1 and false-positive
rate**. Balanced accuracy is `(recall + specificity) / 2`; false-positive rate
is `FP / (FP + TN)`, the fraction of reference-negative time incorrectly marked
positive. It is not the fraction of predictions that are wrong. Ordinary
accuracy and F1 alone can look favorable when most eligible time is positive.
All table metrics are percentages, pooled from sample counts unless stated
otherwise. These are time-sample metrics, not counts of correctly detected
complete gesture units or measurements of boundary accuracy.

## Final detection results

These results include local candidate creation and final grouping: rest thresholds
0.50/0.65, a 300 ms minimum departure, and final joining across gaps of at most
300 ms. Weak rest can continue supported activity but cannot establish a phrase
by itself; final joining respects exclusions, continuity barriers and unscored
gaps. The policies are `contextual-score-candidates-v0.3` and
`probability-finalization-v0.4`. These are not simple score-threshold results.

| Score selection | Balanced accuracy | Precision | Recall | F1 | False-positive rate | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM-only (default) | 65.29 | 86.40 | 97.67 | 91.69 | 67.09 | 85.59 |

TP/FN/FP/TN counts are **9,126 / 218 / 1,437 / 705**. This corresponds to
912.6 s of correctly detected G-unit time, 21.8 s of missed G-unit time,
143.7 s of false alarms and 70.5 s of correctly rejected non-G-unit time.

### Variation across videos

The following are final LLM-only results for the selected intervals in each
video. They show the variation behind the pooled result.

| Experiment | Recording/reference identifier | Evaluated time (s) | Balanced accuracy | F1 |
| --- | --- | ---: | ---: | ---: |
| 001 | M3D_TED_AS2010 | 171.3 | 74.78 | 87.67 |
| 002 | M3D_TED_EG2009 | 275.7 | 55.89 | 95.07 |
| 003 | M3D_TED_ES2016 | 352.6 | 57.68 | 88.35 |
| 004 | M3D_TED_MS2010 | 134.4 | 69.79 | 90.97 |
| 005 | M3D_TED_SJ2010 | 214.6 | 71.88 | 95.39 |

Giving each video equal weight, mean balanced accuracy is 66.00% and mean F1 is
91.49%. Experiment 002 illustrates why F1 needs a companion metric: F1 is
95.07%, while balanced accuracy is only 55.89%.

## Simple baselines and uncertainty

Use both an always-G-unit baseline and independent 50:50 random assignment on
the same eligible samples. Both have 50% balanced accuracy when both reference
classes occur. Random metrics below use analytic expected confusion counts;
the displayed random F1 is a ratio of expected counts, not the expectation of
F1 over finite random realizations.

| Baseline | Balanced accuracy | Accuracy | F1 |
| --- | ---: | ---: | ---: |
| Always G-unit | 50.00 | 81.35 | 89.72 |
| Random 50:50 | 50.00 | 50.00 | 61.93 |

LLM-only detection exceeds these baselines in pooled balanced accuracy,
accuracy and F1. Relative to always-G-unit, its advantage is **15.29 percentage
points in balanced accuracy, 4.24 points in accuracy and 1.97 points in F1**.
Its false-positive rate is worse than random assignment's 50%, compensated by
much higher recall.

### Bootstrap confidence intervals and adjusted p-values

Uncertainty uses **5,000 paired circular-block bootstrap resamples**, stratified
by video, with **10-second blocks** as the primary analysis and 5- and 20-second
blocks as sensitivity checks. Each block carries the paired detector/reference
outcomes and exclusion mask. Gaps retain their time spacing; excluded samples
contribute no scores. Resampling stays within each video and retains its spanned
timeline length. The 100 ms observations are not sampled independently.

The table reports detector-minus-baseline differences in percentage points,
with **95% basic bootstrap confidence intervals** and approximate two-sided
centered-bootstrap **Holm-adjusted p-values**. The correction covers 12 pooled
comparisons: two metrics × two baselines × three block lengths. Confidence
intervals are pointwise, not adjusted for multiple comparisons. F1 is reported
descriptively; the tested metrics are balanced accuracy and accuracy.

| Metric | Baseline | Improvement (pp) | 95% bootstrap CI (pp) | Holm-adjusted p |
| --- | --- | ---: | ---: | ---: |
| Balanced accuracy | Always G-unit | +15.29 | +11.52 to +18.93 | 0.0048 |
| Balanced accuracy | Random 50:50 | +15.29 | +11.52 to +18.93 | 0.0048 |
| Accuracy | Always G-unit | +4.24 | +2.49 to +5.88 | 0.0048 |
| Accuracy | Random 50:50 | +35.59 | +32.99 to +38.41 | 0.0048 |

The two balanced-accuracy comparisons are identical because both baselines equal
50%; retaining both in the correction is conservative. Baseline scores are
recomputed from each resample's reference prevalence. Random assignment is
represented by its analytic expected scores, not a separately sampled random
prediction sequence. Equal displayed p-values reflect the finite resolution of
5,000 resamples, not identical effect sizes.

All four comparisons remain positive, with intervals above zero and adjusted
p-values below 0.05, at both sensitivity block lengths. For balanced accuracy,
the improvement interval is +11.66 to +18.86 points with 5-second blocks and
+11.37 to +18.95 points with 20-second blocks.

These results support better discrimination than the baselines **within these
five recordings and this selection policy**. They do not establish performance
on new speakers or videos. The bootstrap assumes that time blocks adequately
represent dependence within each recording; longer dependence or strongly
changing error patterns can make these intervals too narrow. Circular sampling
joins the end to the beginning of the same recording, never different videos.
The development-informed sample remains an exploratory evaluation.

## Evaluation workflow and provenance

The source was the regenerated filtered evaluation of five completed, reprocessed
runs. Their hashes and final decisions were verified against the release
code. For this report, the saved raw LLM estimate was selected as `Q = L`, and
local candidate creation and final grouping were recomputed. The reference
mapping, time alignment, Phase1 admission and shot-value selection stayed fixed.
Bootstrap comparisons were then recomputed from those final LLM-only decisions.
No preprocessing, prompt, model answer or reference annotation was changed.

The [evaluation provenance](evaluation-provenance.json) records source/reference
and reprocessed-result hashes, per-video confusion counts, the selection and
grouping settings, and the bootstrap seed and results. The base seed is 1729;
each block length uses the first eight SHA-256 bytes of
`1729|all-selected-recordings|SECONDS`, interpreted as an unsigned big-endian integer.
The derived seeds are recorded as well. This fixed rule is shared by batch
reports; identical original and reprocessed timelines
produce the same pooled bootstrap result.

The public `evaluate-run` command scores one saved final result. It does not
implement this report's shot-value filter, pooled aggregation or bootstrap
workflow. Those calculations use the local batch evaluation script, which is not bundled
with this release; the command alone cannot reproduce these tables. The metadata
documents the calculation and its inputs without redistributing videos,
reference annotations or model requests. Repeating inference can produce
different estimates even with the same client and configuration.
