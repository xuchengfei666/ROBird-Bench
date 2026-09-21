# ROBird-Bench v5.3 Development Benchmark P1 Protocol

> Date: 2026-08-22 | Status: `FROZEN_BENCHMARK_P1_V5_3`

## Scope

This is a single-seed development pilot on the frozen v5.3 observer-disjoint
split and frozen DINOv2 ViT-S/14 224-pixel features. It is not a final
multi-seed paper table and contains no unseen-holdout result.

## Ordered runs

1. `mean-feature-k1`: mean feature classifier trained with one photo.
2. `mean-feature-all`: mean feature classifier trained with all available
   photos.
3. If the task-signal gate passes, continue with `deepsets-all`,
   `set-transformer-all` and `rcca-all` under the identical feature, split,
   optimizer, early-stopping and evaluation contracts.

All evaluation is observation-equal on `development_test`, covers budgets
1-5, uses exhaustive/content-independent subsets capped at 32 per
observation-budget, and reports common-cohort macro Top-1, NLL, Brier, ECE,
normalized AUBC and species-stratified observation bootstrap intervals.

## P1 task-signal gate

Continue to the three learned set aggregators only if the all-photo mean-feature
run has both:

- common-cohort macro Top-1 at budget 5 at least 0.5 percentage points above
  budget 1; and
- positive first-to-last net corrections.

Failure stops the method pilot for diagnosis; it does not invalidate the
v5.3 dataset. Passing only authorizes the remaining single-seed development
baselines. No superiority or external-generalization claim is allowed from P1.
