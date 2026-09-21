# ROBird-Bench Dataset P0-v5.1 Parser-Correction Census Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_1_CENSUS` | P0-v5 remains an immutable failed checkpoint

## 1. Rationale

The frozen P0-v5 metadata-only census stopped after 86 of 141 taxa because its shared URL parser rejected the valid iNaturalist Open Data URL `https://inaturalist-open-data.s3.amazonaws.com/photos/1273149/square.`. A read-only HTTP HEAD check returned status 200 and `image/jpeg` for both the source object and its `large.` variant. This protocol corrects only that deterministic parser-compatibility defect.

## 2. Immutable Continuation Checkpoint

The v5 snapshot `code/results/dataset_p0/census_candidates_v5.json` is read-only evidence. Before resuming, v5.1 must verify its SHA-256, status `RUNNING`, contract hash, 86 completed taxa, 7,532 retained groups and last completed taxon `Anas georgica` (`taxon_id=6988`). The v5.1 snapshot is a new artifact and may not overwrite or mutate the v5 checkpoint.

## 3. Parser Correction Contract

The v5.1 parser accepts only the existing iNaturalist size stems `square`, `small`, `medium` and `large`. A URL ending in `stem.` with an empty extension is valid and is normalized to `large.` while preserving the query string. URLs with no size separator, an unknown stem or any other malformed path remain rejected. No photo is silently substituted, downloaded or inferred during this metadata census.

## 4. Unchanged Scientific Contract

The v5.1 run preserves the exact v5 taxon pool of 100 current taxa plus 41 frozen-order reserves; the v4 source cutoff, research-grade filter, allowed licenses, 20-page window, photo-count rule, deterministic seed, inspected observation/observer exclusions, target derivation and candidate cap are unchanged. Only the parser compatibility rule is newly versioned.

## 5. Output and Stop Boundary

The continuation writes only `census_candidates_v5_1.json`, `census_audit_v5_1.json` and runtime logs. Its audit remains non-authorizing with `dataset_p0_decision_authorized=false`, `download_authorized=false`, `class_table_written=false` and `global_milp_evaluated=false`. It must not create a v5.1 class table, run selection, download image bytes, extract features or train models.

## 6. Freeze Contract

The v5.1 freeze record binds this protocol, config, wrapper, unchanged v5 census script, v5 protocol/config/freeze record, the immutable v5 checkpoint and all prior v4 evidence. A completed v5.1 census is evidence for a later separately designed P0 class-selection protocol, not a Dataset P0 PASS.
