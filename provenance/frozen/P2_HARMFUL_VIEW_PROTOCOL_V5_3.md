# ROBird-Bench v5.3 P2 Harmful-View Mechanism Pilot

> Date: 2026-08-22 | Status: `FROZEN_P2_HARMFUL_VIEW_V5_3`

## Question

When a second natural photo from the same observation hurts a frozen
single-photo classifier, can an unlabeled quality/consensus trimmed aggregator
avoid the harmful view without sacrificing the single-photo endpoint?

## Fixed inputs

- v5.3 canonical manifest and observer-disjoint splits;
- frozen DINOv2 ViT-S/14 224-pixel features;
- the already trained `mean-feature-k1` classifier, used only to produce
  per-photo logits;
- development-test split only. No unseen-holdout or confirmatory claim.

## Frozen quality score

For each RGB photo, resize grayscale to 128 x 128 and compute:

`q = z(log(1 + LaplacianVariance)) + 0.25*z(contrast) + 0.25*z(exposure)`

where contrast is grayscale standard deviation and exposure is
`1 - 2*abs(mean-0.5)`. Z-scores are computed within each observation only.
For a one-photo observation, `q=0`.

## Frozen aggregators

`mean_logit`, `median_logit`, `quality_top_half`, `consensus_top_half`, and the
primary `quality_consensus_trim`. The latter scores each view by the mean of
its within-observation z-scored quality and cosine similarity to the centered
logit consensus, then averages the highest `ceil(V/2)` logits. Ties are broken
by ascending `photo_id`. No label, split outcome or test metric enters a score.

All methods evaluate budgets 1-5, use every eligible development-test
observation at each budget and report the fixed common cohort for the budget
1-to-2 gate. Metrics include macro Top-1, micro Top-1, NLL, Brier, ECE,
corrections/regressions and species-stratified bootstrap intervals.

## P2 continuation gate

The primary method passes only if, on the 750-observation budget 1-to-2 common
cohort:

1. its budget-2 macro Top-1 is at least 0.5 percentage points above
   `mean_logit` budget 2;
2. its budget-2 minus budget-1 macro delta is at least -0.003;
3. corrections exceed regressions relative to its own budget-1 predictions.

Failure stops this mechanism pilot and is recorded as negative evidence. It
does not authorize another unregistered method family.
