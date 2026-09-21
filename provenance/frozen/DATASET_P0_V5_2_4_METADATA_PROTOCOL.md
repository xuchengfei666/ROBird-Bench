# ROBird-Bench Dataset P0-v5.2.4 Metadata Manifest Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_4_METADATA` | class table and replacement audit are immutable

## 1. Purpose

This stage converts the PASS replacement audit's selected groups plus the frozen inspected seed manifests into the canonical metadata manifest for the v5.2.4 collection. It does not query a new source window or select a different candidate.

## 2. Fixed Inputs and Provenance

The class table is `frozen_100_taxa_v5_2_4.csv`. The expansion groups are exactly `selected_groups` from the PASS replacement audit for `Sterna striata`. Retained seed rows come from the original and confirmation manifests after excluding both historical removed seed taxa (`Geothlypis philadelphia`, `Cardellina canadensis`) from retained rows; their observation and observer IDs remain in the exclusion sets. Source URLs, licenses, attribution and observation provenance are preserved verbatim.

## 3. Metadata Gates

The manifest must pass canonical schema validation, exactly 100 taxa and 5,000 observation groups, at least 12,000 photos, final 48-52 groups per taxon, 2-5 photos per observation, allowed CC0/CC-BY/CC-BY-SA licenses, globally unique photo IDs, one selected group per new observer, and zero selected observer/observation overlap with all inspected seed IDs. These are metadata gates; image bytes, exact file hashes and near-duplicate status are evaluated after download.

## 4. Download and Audit Boundary

The resulting audit status is `PASS_METADATA_TO_DOWNLOAD`. It authorizes the existing byte downloader under this frozen config, but Dataset P0 is not considered complete until every image is decoded and hashed, provenance/byte gates pass, and near-duplicate candidates receive the required review. No features or models may run before those gates.

## 5. Freeze Contract

`FROZEN_DATASET_P0_V5_2_4_METADATA.sha256` binds this protocol, config, metadata builder, existing downloader, the frozen class-table chain and the v5.2.4 replacement chain.
