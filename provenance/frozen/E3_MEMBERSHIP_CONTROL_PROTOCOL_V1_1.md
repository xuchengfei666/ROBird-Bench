# E3 Same-Observation Membership Negative Control v1.1

Date: 2026-09-05  
Status: frozen continuation after v1 design defect

## Correction

E3-v1 circularly shifted complete observation groups. That preserved the
collection of evaluated sets and could not isolate membership effects. The v1
result is retained as a design-defect record. This v1.1 continuation changes
only the shuffle operator and keeps all data, models, metrics and split rules
fixed.

## Frozen design

- Evaluation split: ROBird-v5.3 `development_test` only.
- Models: the four frozen P4.1 all-photo checkpoints; no retraining.
- Real condition: original observation groups with the frozen exhaustive
  budget-subset protocol for budgets 1--5.
- Negative-control condition: within each `(taxon_id, photo_count)` stratum,
  flatten photos in deterministic `(observation_id, photo_id)` order and apply
  a one-photo circular shift. Repartition the shifted stream into the original
  group boundaries. This preserves taxon, group cardinality and the marginal
  photo pool, but breaks same-observation membership for every eligible
  multi-view stratum.
- Single-view groups are excluded from the primary control because a photo-level
  permutation cannot change their membership semantics; the excluded count is
  recorded.
- The same checkpoints, DINOv2 features, subset enumeration and metrics are
  used for both conditions.

## Interpretation boundary

A performance change on the photo-level shuffled condition is diagnostic of
membership-sensitive set composition. It is not evidence of biological
relatedness, causal optical complementarity or observer identity.
