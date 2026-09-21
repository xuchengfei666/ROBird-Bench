# ROBird-Bench Dataset P0-v5.2.6 Observation Cleanup Protocol

> Date: 2026-08-22 | Status: `FROZEN_P0_V5_2_6_CLEANUP`

## Purpose

v5.2.5 passed metadata selection but its byte audit found one exact duplicate
observation and 14 cross-observer dHash candidates. This version creates a new
metadata manifest without changing v5.2.5 or any earlier artifact.

## Fixed rule

1. Every observation containing a repeated SHA-256 is forced into quarantine as
   a whole observation. This prevents retaining two metadata rows for the same
   byte and keeps observation groups internally intact.
2. Build an observation graph from the 14 frozen v5.2.5 dHash pairs. Solve a
   minimum-cost binary vertex cover. Costs are 1000 for ROQ seed observations,
   100 for v5.2.4 expansion observations and 1 for v5.2.5 expansion
   observations, with a stable hash tie term. The result quarantines only
   observations; no near-duplicate is declared `distinct`.
3. Keep all other observations and reselect the remaining expansion groups from
   the complete v5.1 census pool using the existing bounded-total MILP,
   observer-capacity one, exact 5,000 groups and 48-52 groups per taxon.

If metadata gates pass, only v5.2.6 byte download is authorized. A fresh
non-empty exact or near-duplicate result remains a Dataset P0 STOP and requires
another version or explicit human review. No split, feature or model output is
authorized by this protocol.
