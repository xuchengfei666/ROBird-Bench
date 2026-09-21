# ROBird-Bench Dataset P0-v5.2.1 Joint Replacement Audit Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_1_REPLACEMENT_AUDIT` | v5.2 and v1-v5.1 artifacts remain immutable

## 1. Purpose

The v5.2 launch stopped before candidate enumeration because its frozen script read `expected_v5_1_reserve_individually_feasible` while its frozen configuration declared only the semantically equivalent `expected_v5_1_reserve_feasible`. v5.2.1 is an engineering-only continuation. It preserves the v5.2 replacement rule, source, seed, thresholds, candidate pool, MILP, metadata gates and output boundary, while adding the missing configuration alias and using isolated output paths.

## 2. Scientific Contract

The failed seed class remains `Cardellina canadensis` (`taxon_id=145275`). All 41 reserve taxa remain in completed v5.1 census order. Each temporary table has 78 seed taxa and 22 expansion taxa, final group bounds 48-52, exact 2,972 new groups, one selected group per new observer, and the unchanged metadata gates. The first candidate passing every gate remains the only permitted replacement; otherwise the audit stops.

## 3. Engineering Correction

The new configuration contains both reserve-feasibility key spellings, with value `41`, so the unchanged validation semantics are explicit and the corrected v5.2.1 script can read the long spelling. No source query, API window, threshold, seed or selection rule is changed. The original v5.2 files and failed launch evidence remain read-only.

## 4. Output Boundary

The v5.2.1 script may write only `replacement_attempts_v5_2_1.json` and `replacement_audit_v5_2_1.json`. It cannot write a class table, canonical manifest, image bytes, feature cache or model result. Even a metadata PASS only authorizes a separately frozen class-table stage; it does not authorize image download.

## 5. Freeze Contract

`FROZEN_DATASET_P0_V5_2_1_REPLACEMENT.sha256` binds this protocol, config and corrected script, the immutable v5.2 freeze record and all v5.1/v4 evidence. Any failure at the metadata gate remains a scientific STOP.
