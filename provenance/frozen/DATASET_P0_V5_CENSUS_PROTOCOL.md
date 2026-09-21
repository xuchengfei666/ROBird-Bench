# ROBird-Bench Dataset P0-v5 Preselection Census Protocol

> Date: 2026-08-20 | Status: `FROZEN_P0_V5_CENSUS` | Dataset P0-v4 remains terminally stopped

## 1. Rationale

Dataset P0-v4 stopped at taxon 71/100 when `Cardellina canadensis` retained 20 independent candidates against its frozen seed lower bound of 22. The remaining 29 current taxa were not observed. Replacing only the first failed class would therefore repeat class-table selection from incomplete evidence.

This protocol defines a bounded metadata census that observes the complete current and reserve pools before any P0-v5 class table is proposed. It is not a Dataset P0 gate and cannot produce a download authorization.

## 2. Frozen Taxon Universe

The census order is fixed before execution:

1. all 100 taxa in `frozen_100_taxa_v4.csv`, ordered by `class_index`;
2. all resolved taxa in the pre-existing `scaleup_feasibility/audit.json` order that are absent from the v4 table and had at least 50 eligible groups and 50 eligible observers in that frozen audit.

The second rule yields exactly 41 reserve taxa, for an exact census total of 141 unique taxa. No new taxonomy window, API-driven taxon search or result-dependent pool expansion is allowed.

## 3. Source and Eligibility Contract

The census copies the complete v4 collection contract: seed `20260819`, API cutoff `2026-08-18T16:00:00Z`, research grade, CC0/CC-BY/CC-BY-SA photos, 2-5 eligible photos per observation, 20 pages of 200 observations, deterministic descending creation order and exclusion of every originally inspected observation and observer ID.

The first 71 current taxa are bootstrapped only from the exact terminal v4 candidate snapshot. Its SHA-256, matching v4 audit reference, status, contract and completed-prefix order must verify before reuse. The remaining taxa are queried under the unchanged source contract. Atomic snapshots are resumable only when their census contract hash matches.

## 4. Per-Taxon Diagnostic

- current seed taxa have 26 existing groups and require 22-26 new groups;
- current expansion taxa have zero existing groups and require 48-52 new groups;
- reserve taxa have zero existing groups and require 48-52 new groups;
- candidate compaction retains at most four times the upper target and at most one group per observer within each taxon.

The script records a shortfall whenever retained independent candidates are below the applicable lower target, but it continues through all 141 taxa. Per-taxon success does not prove global observer-capacity feasibility.

## 5. Output and Stop Boundary

The census may write only an atomic candidate snapshot, a final census audit and runtime logs. The audit reports current shortfalls, individually feasible reserves, counts, hashes and the explicit boundary `dataset_p0_decision_authorized=false`.

It must not write a v5 class table, run a global selection MILP, create a metadata or canonical manifest, download image bytes, extract features or train models. A later P0-v5 proposal requires a separately justified class-selection rule, protocol, configs, tests and freeze record.

## 6. Freeze Contract

Before collection, a dedicated census hash record must cover this protocol, the census config, the v4 class table, the frozen feasibility audit, the v4 config and freeze record, and all v1-v4 terminal selection evidence needed to prove the continuation chain. All prior Dataset P0 artifacts remain read-only.
