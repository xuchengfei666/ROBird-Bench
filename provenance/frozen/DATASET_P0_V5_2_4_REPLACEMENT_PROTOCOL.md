# ROBird-Bench Dataset P0-v5.2.4 Joint Replacement Audit Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_4_REPLACEMENT_AUDIT` | v5.2-v5.2.3 and v1-v5.1 artifacts remain immutable

## 1. Purpose

The completed v5.2.3 run showed that all feasible candidates were rejected by an implementation error in the `species_balance` metadata gate. The bounded MILP returns new-group counts (22-26 for retained seed taxa and 48-52 for expansion taxa), but the gate compared those bounds directly with final group counts (48-52). v5.2.4 corrects this accounting.

## 2. Scientific Contract

The 41-reserve order, failed seed removal, historical seed exclusions, source, seed, exact 2,972 new groups, observer capacity, replacement rule and all metadata thresholds are unchanged. The corrected gate checks both final groups per taxon (48-52) and new groups per taxon against the frozen lower/upper targets.

## 3. Output Boundary

The v5.2.4 script may write only `replacement_attempts_v5_2_4.json` and `replacement_audit_v5_2_4.json`. It cannot write a class table, canonical manifest, image bytes, feature cache or model result. A metadata PASS only authorizes a separately frozen class-table stage.

## 4. Freeze Contract

`FROZEN_DATASET_P0_V5_2_4_REPLACEMENT.sha256` binds this protocol, config and corrected script, the immutable v5.2.3/v5.2.2/v5.2.1/v5.2 records and all v5.1/v4 evidence.
