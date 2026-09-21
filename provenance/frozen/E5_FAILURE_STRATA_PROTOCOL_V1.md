# E5 Failure-Stratum Analysis

Date: 2026-09-05  
Status: frozen

## Question

Which observation, observer and image strata drive the non-monotonic budget
behavior and the cross-source transfer gap?

## Frozen inputs

- ROBird-v5.3 manifest and observer-disjoint split.
- P4.1 all-photo predictions for Deep Sets, Set Transformer, RCCA and
  MaxFeature on `development_test`.
- G6X-v1.1 manifest and external predictions for source-level comparison.
- No new training, calibration or test-driven threshold selection.

## Strata

For ROBird observation groups, report exact photo cardinality (2/3/4/5),
development-test observer frequency (1/2--3/4+ observations), and mean group
pixel area (`<0.5M`, `0.5--1.5M`, `>=1.5M`). Also report a per-taxon table.
For each stratum and budget 1, 2 and 5, report group count, accuracy, mean
confidence, NLL, and budget-1-to-2 correction/regression counts.

For G6X, report per-taxon single-photo accuracy, confidence and true-class
probability for Deep Sets, Set Transformer, RCCA and MaxFeature. G6X has no
observation-level grouping and is not used for multi-photo strata.

## Interpretation boundary

This analysis describes where the benchmark and transfer models fail. It does
not infer biological traits, optical causes, observer identity equivalence or
causality from metadata.
