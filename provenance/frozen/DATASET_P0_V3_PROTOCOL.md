# ROBird-Bench Dataset P0-v3 Protocol

> Date: 2026-08-19 | Status: `FROZEN_DATASET_P0_V3` | Supersedes no prior artifact

## 1. Rationale

Dataset P0-v2 remains `STOP_SCALEUP_PER_SPECIES_INFEASIBLE`. Its 20-page window repaired both known P0-v1 gull deficits, but `Gavia pacifica` supplied only 22 independent eligible observers for a fixed target of 24. Because the window contained only 24 eligible groups in total, further pagination is not an evidence-based repair. The exact value of 50 groups per taxon is an engineering convention rather than a biological threshold.

## 2. Only Scientific Design Change

P0-v3 replaces exact per-taxon equality at 50 groups with a common final interval of 48-52 groups while preserving exactly 5,000 groups overall:

- all 100 frozen taxa must end with 48-52 groups;
- each of the 80 seed taxa starts with 26 groups and receives 22-26 new groups;
- each of the 20 expansion taxa receives 48-52 new groups;
- exactly 2,920 new groups are selected;
- every newly selected observer has global capacity one.

The exact taxon table, seed `20260819`, API cutoff `2026-08-18T16:00:00Z`, 20 x 200 query window, descending `created_at` order, research grade, CC0/CC-BY/CC-BY-SA licenses, 2-5 non-hidden photos and deterministic metadata-only selection remain unchanged from v2. No taxon receives an individual exception and no taxon is removed.

## 3. Bounded-Total Gate

For each taxon, the metadata precheck requires at least its lower-bound number of independent eligible observers. Candidate retention uses a deterministic cap equal to four times its upper bound. If every lower bound passes, one binary MILP must enforce all per-taxon intervals, exactly 2,920 selected groups and at most one selected group per new observer globally.

The combined metadata manifest must contain exactly 100 taxa and 5,000 groups, 48-52 groups per taxon, at least 12,000 photos, zero new-to-seed observation overlap and zero new-to-seed observer overlap. Solver failure or any gate failure is a STOP.

## 4. Isolation and Claim Boundary

P0-v1 and P0-v2 protocols, candidates and STOP audits remain read-only negative results. P0-v3 uses separate config, candidate, audit, manifest, duplicate-review, split and `E:/Datasets/ROBird-Bench/p0-v3/` paths. It does not relabel either earlier result.

Only `PASS_METADATA_TO_DOWNLOAD` permits image download. Every downloaded image must later pass decoding, path, attribution, license and SHA-256 checks. Exact/near-duplicate review and observer-disjoint split gates remain mandatory before feature extraction or training.

## 5. Freeze Contract

`FROZEN_DATASET_P0_V3.sha256` covers this protocol, both v3 configs, the unchanged exact 100-taxon table, the completed taxonomy feasibility audit and both immutable v1/v2 STOP audits. All three versioned hash chains must verify before P0-v3 collection.
