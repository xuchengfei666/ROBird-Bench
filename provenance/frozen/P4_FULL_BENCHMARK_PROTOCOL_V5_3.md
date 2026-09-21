# ROBird-Bench v5.3 P4 Full Comparative Suite

> Date: 2026-08-22 | Status: `FROZEN_P4_FULL_BENCHMARK_V5_3`

## Purpose

P4 tests whether the promising but non-confirmatory P3 RLRA result is specific
to risk-aware reliability or is shared by standard learned set aggregators.
P3's checkpoint and development-test result are fixed inputs; P4 does not
retrain or retune RLRA.

## Fixed comparison

Train and validation use all photos in each observation under the v5.3
observer-disjoint split. The runner evaluates four separately named methods:

- `deepsets-all`;
- `set-transformer-all`;
- `rcca-all`;
- `max-feature-all`.

Each checkpoint is selected only on train/validation. Development-test is
evaluated once per checkpoint with the existing exhaustive/content-independent
budget-1..5 evaluator. The P3 `rlra` output is copied into the consolidated
summary as a fixed prior result, not recomputed or selected against the P4
baselines.

## Outputs

The suite records per-method metrics and predictions, then writes one long
summary table and JSON with Macro/Micro Top-1, NLL, Brier, ECE, normalized
Macro AUBC, common-cohort budget curves and paired corrections/regressions.
All outputs remain `CONTINUE_DEVELOPMENT_ONLY`; G6 is not manufactured by
reusing the development-test split.

## Boundary

P4 is a comparative evidence stage. It has no post-hoc winner gate and cannot
trigger additional tuning, data changes or a new method family. A baseline
that beats RLRA is reported as evidence, not silently adopted. Any final
method claim requires an unseen G6 holdout under a separately frozen protocol.
