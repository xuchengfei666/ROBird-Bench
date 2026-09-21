# ROBird-Bench Dataset P0-v5.2.4 Class-Table Freeze Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_4_CLASS_TABLE` | replacement audit is immutable

## 1. Purpose

The v5.2.4 replacement audit passed all metadata gates for `Sterna striata` (`taxon_id=4478`). This stage writes the exact 100-class table embedded in that PASS audit and verifies that the table cannot drift before metadata-manifest construction.

## 2. Fixed Table Rule

The table is copied only from `selected_table_preview` in `replacement_audit_v5_2_4.json`. It must contain 100 unique contiguous classes, 78 retained inspected-seed taxa and 22 iNaturalist 2021 expansion taxa. `Cardellina canadensis` and the historical v4-removed `Geothlypis philadelphia` are absent from retained seed data; `Sterna striata` occupies the original failed class position. Seed rows retain 26 existing groups with a 24-group target; expansion rows have zero existing groups with a 50-group target.

## 3. Output Boundary

This stage writes only `frozen_100_taxa_v5_2_4.csv` and a small class-table audit. It does not query the API, download images, build a photo manifest, extract features or train models.

## 4. Freeze Contract

`FROZEN_DATASET_P0_V5_2_4_CLASS_TABLE.sha256` binds this protocol, config and writer, the complete v5.2.4 replacement audit/ledger and the v5.2.4 replacement freeze chain. The next metadata-manifest stage requires this class-table freeze.
