# E3 Same-Observation Membership Negative Control v1.2

Date: 2026-09-05  
Status: frozen continuation after v1 and v1.1 control limitations

## Correction

E3-v1 permuted complete sets and E3-v1.1 retained most original photos in each
destination. Both are preserved as audit records but are not the strict primary
control. This continuation requires at least three observations per
taxon/cardinality stratum and uses a per-position derangement of source
observations.

## Frozen design

- Evaluation split: ROBird-v5.3 `development_test` only.
- Models and checkpoints: unchanged P4.1 all-photo checkpoints.
- Real condition: original groups with the frozen exhaustive budget protocol.
- Negative-control condition: for each eligible `(taxon_id, photo_count)`
  stratum, sorted observations are arranged as equal-length photo lists. For
  photo position `j`, destination index `d` receives the position-`j` photo
  from source index `d + 1 + (j mod (n-1))` modulo `n`. Thus every destination
  photo comes from a different source observation, every source photo is used
  once, and taxon/cardinality marginals are preserved.
- Strata with fewer than three observations or fewer than two photos are
  excluded and counted explicitly.

## Interpretation boundary

Any difference between real and strict-shuffled sets is diagnostic of natural
observation membership and within-observation correlation. It is not a causal
biological or optical claim, and the mixed sets are not a deployment protocol.
