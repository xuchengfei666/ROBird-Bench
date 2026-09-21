# ROBird-Bench Dataset P0-v2 Protocol

> Date: 2026-08-19 | Status: `FROZEN_DATASET_P0_V2` | Supersedes no prior artifact

## 1. Rationale

Dataset P0-v1 remains `STOP_SCALEUP_PER_SPECIES_INFEASIBLE`. In its frozen five-page API window, `Larus occidentalis` retained 18 independent candidates and `Larus californicus` retained 15, below the fixed target of 24. P0-v2 tests whether this was caused by insufficient metadata window depth after excluding all 1,504 inspected seed observers.

## 2. Only Design Change

The maximum iNaturalist observation query window changes from 5 pages x 200 records to 20 pages x 200 records per taxon. Results remain ordered by descending `created_at`. Collection may stop early only after the metadata-only independent-observer candidate cap is reached.

No other scientific field changes:

- exact 100-taxon table SHA-256: `421eb906908c6ccdfa183b6bd093001255b8befa59cff65a33f5e6d31b9a5ba7`;
- API cutoff: `2026-08-18T16:00:00Z`;
- quality grade: research;
- photo licenses: CC0, CC-BY and CC-BY-SA;
- 2-5 non-hidden photos per observation;
- deterministic metadata-only photo and observation selection;
- 80 seed taxa retain 26 groups and target 24 new groups each;
- 20 expansion taxa target 50 groups each;
- every newly selected observer has global capacity one;
- exact endpoint: 100 taxa, 5,000 groups and at least 12,000 photos.

## 3. Isolation

P0-v1 protocol, configs, freeze record, candidate snapshot and STOP audit are immutable. P0-v2 uses new configs, hash record, candidate/audit/manifests and `E:/Datasets/ROBird-Bench/p0-v2/` for future image bytes. A v2 result cannot overwrite or relabel v1.

## 4. Gates

The collector stops immediately when any taxon has fewer independent candidates than its target. If every taxon passes this precheck, the global one-observer MILP must select exactly 2,920 new groups. The combined manifest must then contain exactly 100 taxa and 5,000 groups, at least 12,000 photos, zero new-to-seed observation overlap and zero new-to-seed observer overlap.

Only `PASS_METADATA_TO_DOWNLOAD` permits image download. Every image must subsequently pass full decoding, local-path, attribution, license and SHA-256 verification. Exact/near-duplicate audit and observer-disjoint split gates remain mandatory before features or training.

## 5. Freeze Contract

`FROZEN_DATASET_P0_V2.sha256` covers this protocol, both v2 configs, the unchanged exact 100-taxon table, the completed taxonomy feasibility audit and the v1 STOP audit. Both v1 and v2 hash chains must verify before v2 execution.
