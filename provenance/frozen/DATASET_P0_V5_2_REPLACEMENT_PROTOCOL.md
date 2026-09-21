# ROBird-Bench Dataset P0-v5.2 Joint Replacement Audit Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_REPLACEMENT_AUDIT` | v1-v5.1 artifacts remain immutable

## 1. Rationale

The complete v5.1 metadata census found one current class below its common lower bound: `Cardellina canadensis` (`taxon_id=145275`) retained 20 independent candidates against 22. It found 41 reserve taxa with no individual shortfall at the expansion lower bound of 48. An isolated first-reserve replacement could still fail the global observer-capacity constraint, so v5.2 evaluates the complete fixed reserve pool once.

## 2. Fixed Replacement Rule

The failed seed class is removed and no other current class changes. Reserve candidates are enumerated in the exact order of the completed v5.1 census pool, inherited from the immutable feasibility-audit order. For each candidate, the script:

1. replaces the failed class at its original class position;
2. derives 22-26 new-group bounds for 78 retained seed taxa and 48-52 bounds for the 22 expansion taxa;
3. sets exact new groups to `2,972` (`5,000 - 78 x 26`);
4. runs the deterministic one-new-observer-per-selected-group bounded MILP;
5. checks all metadata gates listed below.

The first candidate passing all gates is the sole permitted replacement. If no candidate passes, the audit is a terminal STOP and no class table is generated.

## 3. Immutable Inputs and Metadata Gates

The audit uses only the v5.1 retained candidate groups, the v5.1 audit, the v4 class table, the frozen seed manifests and the v1-v5.1 freeze chain. No API query, source-window change, seed change or new taxonomy search is allowed.

The candidate must satisfy: exactly 100 taxa, exactly 5,000 observation groups, at least 12,000 photos, final 48-52 groups per taxon, zero duplicate photo/observation IDs, 2-5 eligible photos per group, only CC0/CC-BY/CC-BY-SA licenses, one selected group per new observer, zero selected new observer/observation overlap with all inspected seed IDs, and canonical manifest schema validity.

These gates are metadata evidence only. They do not verify image bytes, near duplicates or holdout splits.

## 4. Outputs and Boundary

The script may write only an atomic attempt ledger and final replacement audit under v5.2 paths. The audit records every candidate's MILP status, metadata gates and the first passing candidate with its selected groups. It must not write a class table, canonical manifest, image bytes, feature cache or model output. A PASS does not itself authorize image download; it authorizes creation of a separately frozen v5.2 class table/config.

## 5. Freeze Contract

`FROZEN_DATASET_P0_V5_2_REPLACEMENT.sha256` covers this protocol, the config and script, the complete v5.1 census snapshot/audit and freeze record, the v4 class table/config/protocol and all prior immutable freeze evidence. All v1-v5.1 artifacts are read-only.
