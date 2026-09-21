# ROBird-Bench Dataset P0-v5.2.2 Joint Replacement Audit Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_2_REPLACEMENT_AUDIT` | v5.2/v5.2.1 and v1-v5.1 artifacts remain immutable

## 1. Purpose

The v5.2.1 launch reached the frozen feasibility audit but stopped before MILP because two historical candidate rows have `taxon_id: null` (`Corvus caurinus` and `Corvus monedula`) after unresolved taxonomy. v5.2.2 is an engineering-only continuation that filters those non-resolved rows when constructing the lookup used by the already-frozen 41-reserve pool. It does not alter the reserve pool or any scientific gate.

## 2. Scientific Contract

The failed seed class, 41 reserve order, source, seed, target bounds, exact 2,972 new groups, one-new-observer capacity and all metadata gates are inherited unchanged from v5.2. The first candidate passing every gate remains the only permitted replacement; otherwise the audit stops.

## 3. Engineering Correction

Only feasibility rows with a non-null `taxon_id` are indexed. The v5.1 census already supplies the authoritative reserve IDs and the script still fails closed if any reserve ID is absent from this filtered lookup. The two unresolved historical rows are not reserve taxa and are not eligible for selection.

## 4. Output Boundary

The v5.2.2 script may write only `replacement_attempts_v5_2_2.json` and `replacement_audit_v5_2_2.json`. It cannot write a class table, canonical manifest, image bytes, feature cache or model result. A metadata PASS only authorizes a separately frozen class-table stage.

## 5. Freeze Contract

`FROZEN_DATASET_P0_V5_2_2_REPLACEMENT.sha256` binds this protocol, config and corrected script, the immutable v5.2.1/v5.2 records and all v5.1/v4 evidence.
