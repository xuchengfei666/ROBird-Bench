# ROBird-Bench Dataset P0-v4 Protocol

> Date: 2026-08-20 | Status: `FROZEN_DATASET_P0_V4` | Supersedes no prior artifact

## 1. Rationale

Dataset P0-v3 remains an immutable negative result. Its common 48-52 final-group interval repaired the v2 `Gavia pacifica` deficit, but the frozen 20-page source window reached `Geothlypis philadelphia` with only 16 independent eligible observers, below the common lower bound of 22. The failure is a class-table feasibility failure, not a reason to weaken the common balance rule.

## 2. Deterministic Class-Table Repair

P0-v4 changes exactly one dataset-design object: the 100-taxon class table.

- remove the failed seed taxon `Geothlypis philadelphia` (`taxon_id=145224`);
- scan the already frozen taxonomy-feasibility candidate order;
- select the first candidate not already in the v3 class table whose frozen feasibility audit has at least 50 eligible groups and 50 independent eligible observers;
- under the recorded audit, this deterministic replacement is `Sterna paradisaea` (`taxon_id=4449`), with 153 eligible groups and 87 independent observers;
- retain the other 99 v3 taxa and do not inspect a new candidate window.

The replacement rule is determined from the pre-existing feasibility audit, not from the v3 failure output or image content. No class receives a special lower bound and no source-query condition changes.

## 3. Balance and Source Contract

All v3 source and selection conditions remain unchanged: seed `20260819`, exact API cutoff `2026-08-18T16:00:00Z`, research-grade observations, CC0/CC-BY/CC-BY-SA licenses, 2-5 non-hidden photos, 20 x 200 pages, deterministic ordering and one selected group per new observer globally.

The repaired table contains 79 retained seed taxa and 21 expansion taxa:

- each seed taxon ends with 48-52 groups and receives 22-26 new groups;
- each expansion taxon ends with 48-52 groups and receives 48-52 new groups;
- the exact global endpoint remains 100 taxa and 5,000 groups;
- the exact new-group total is `2,946` (`5,000 - 79 x 26`).

The bounded-total MILP and all provenance, duplicate, byte and split gates remain mandatory. A lower-bound or global-MILP failure is a STOP.

## 4. Isolation and Claim Boundary

P0-v1, P0-v2 and P0-v3 protocols, class tables, candidates and STOP audits remain read-only. P0-v4 uses independent protocol, class-table, candidate, audit, manifest, duplicate-review, split and `E:/Datasets/ROBird-Bench/p0-v4/` paths. No image download is allowed unless the v4 collector writes `PASS_METADATA_TO_DOWNLOAD`.

## 5. Freeze Contract

`FROZEN_DATASET_P0_V4.sha256` covers this protocol, both v4 configs, the deterministic repaired class table, the taxonomy feasibility PASS and all three immutable v1/v2/v3 STOP audits. The v1-v4 hash chains must verify before metadata collection.
