# ROBird-Bench Dataset P0-v5.2.3 Joint Replacement Audit Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_3_REPLACEMENT_AUDIT` | v5.2-v5.2.2 and v1-v5.1 artifacts remain immutable

## 1. Purpose

The v5.2.2 launch reached seed preparation but exposed a historical class-table boundary: the v4 table already removed `Geothlypis philadelphia` (`taxon_id=145224`), while the original seed manifest still contains its 26 groups. The new replacement table also removes `Cardellina canadensis` (`taxon_id=145275`). v5.2.3 explicitly excludes both historical removed seed taxa from retained seed rows while keeping all of their observations and observers in the exclusion sets.

## 2. Scientific Contract

The v5.2 reserve order, source, seed, target bounds, exact 2,972 new groups, one-new-observer capacity, candidate rule and metadata gates are unchanged. The resulting temporary table has 78 retained seed taxa and 22 expansion taxa. A candidate passes only if the existing bounded-total MILP and every metadata gate pass.

## 3. Engineering Correction

Both removed IDs are required to have exactly 26 seed observation groups. Their rows are removed only from the retained seed manifest; their observation and observer IDs remain excluded from all new selections. Any remaining seed taxon absent from the temporary class map fails closed.

## 4. Output Boundary

The v5.2.3 script may write only `replacement_attempts_v5_2_3.json` and `replacement_audit_v5_2_3.json`. It cannot write a class table, canonical manifest, image bytes, feature cache or model result. A metadata PASS only authorizes a separately frozen class-table stage.

## 5. Freeze Contract

`FROZEN_DATASET_P0_V5_2_3_REPLACEMENT.sha256` binds this protocol, config and corrected script, the immutable v5.2.2/v5.2.1/v5.2 records and all v5.1/v4 evidence.
