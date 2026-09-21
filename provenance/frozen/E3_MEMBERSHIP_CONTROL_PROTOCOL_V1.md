# E3 Same-Observation Membership Negative Control

Date: 2026-09-05  
Status: frozen

## Question

Does a learned set model benefit from the fact that photos belong to the same
natural observation, rather than merely receiving more images of the same
taxon?

## Frozen design

- Evaluation split: ROBird-v5.3 `development_test` only.
- Models: the four frozen P4.1 all-photo checkpoints (`Deep Sets`, `Set
  Transformer`, `RCCA`, `MaxFeature`). No retraining and no test-driven tuning.
- Real condition: the original observation groups and the frozen exhaustive
  budget-subset protocol for budgets 1--5.
- Negative-control condition: within each `(taxon_id, photo_count)` stratum,
  sort observation IDs and circularly shift the complete photo groups by one
  position. The destination keeps its original taxon label and group size but
  receives the source observation's photos. Strata with one observation are
  excluded and counted explicitly.
- The permutation is one-to-one inside every eligible stratum, so marginal
  photo counts and image-source distribution are preserved and no shuffled
  photo is reused across destination groups.
- The same model checkpoint, feature matrix, budget enumeration and metric
  implementation are used for both conditions.

## Interpretation boundary

The experiment is a negative control, not a new training benchmark. A drop on
the shuffled condition is evidence that observation membership carries useful
within-set structure for the evaluated model; it is not proof of biological
relatedness, view complementarity, or an optical cause. A null result would
mean that the current benchmark gain may be explainable by same-taxon image
pooling alone.

## Stop conditions

- Stop if any eligible shuffled group changes cardinality, taxon label or
  photo uniqueness.
- Stop if the shuffle creates overlap between source groups or if the input
  checkpoint provenance cannot be verified.
- Do not modify v5.3, P4.1 or G6X artifacts based on the outcome.
