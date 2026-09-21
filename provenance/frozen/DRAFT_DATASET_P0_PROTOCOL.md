# ROBird-Bench Dataset-and-Task P0 Protocol

> Date: 2026-08-19 | Status: `FROZEN_DATASET_P0` | Execution: allowed only through verified gates

## 1. Purpose

P0 tests whether a scalable, legally traceable and leakage-resistant benchmark can be built from public iNaturalist bird observations containing multiple real photographs. It does not test a learned method and does not reinterpret photographs as high/low-quality pairs.

## 2. Data Unit

- One sample is one iNaturalist observation with one taxon label and at least two eligible photos.
- Photos form an unordered set. Any budgeted subset is selected by content-independent IDs or exhaustive subset enumeration.
- Observation and observer identities, not individual photos, define split boundaries.
- Existing ROQ and Confirmation-P0 cohorts are development data because their outcomes have already been inspected.
- A future confirmation cohort must be collected after the benchmark protocol and models are frozen.

## 3. Required Manifest Fields

`cohort`, `observation_id`, `observer_id`, `taxon_id`, `scientific_name`, `common_name`, `observed_on`, `created_at`, `place_ids`, `photo_id`, `license_code`, `attribution`, `url`, `local_path`, `sha256`, `width`, `height`.

Exact coordinates are not stored. Empty optional metadata must remain explicit rather than being fabricated.

## 4. Frozen Gates

These values are final Dataset P0 thresholds for version `p0-20260819`.

| Gate | Frozen requirement | Failure consequence |
|---|---|---|
| P0-G0 Provenance | Every photo has an allowed open license, attribution, stable photo ID and source URL | Stop collection release |
| P0-G1 Scale | At least 100 species, 5,000 observation groups and 12,000 photos | Retain as pilot only |
| P0-G2 Per-species support | At least 30 groups and 15 observers per retained species; no observer contributes over 20% of a species | Drop ineligible species before image/model inspection |
| P0-G3 Group integrity | One observer and one taxon per observation; 2-5 retained photos per group | Stop and repair manifest builder |
| P0-G4 Leakage | Zero observation, observer or exact-file overlap across splits | Stop before feature extraction |
| P0-G5 Near duplicates | Perceptual-near-duplicate candidates are quarantined and audited before final split release | No confirmatory test until resolved |
| P0-G6 Holdout | A new, never-inspected observer/observation-disjoint cohort is reserved after protocol freeze | Development-only paper claims |

## 5. Development Splits

The development pool is divided by observer with a deterministic mixed-integer assignment. Targets are 70% train, 15% validation and 15% development test at the observation-group level, balanced per species as closely as feasible. Existing historical split labels are retained as provenance fields but do not determine the new benchmark split.

## 6. Benchmark Tasks

1. Observation-level classification from an unordered set of `k` photos.
2. Accuracy/calibration versus photo budget, using all subsets for groups of at most five photos when feasible.
3. Observer and optional geography/time stress evaluation.
4. Optional calibrated early stopping after the common classification benchmark is established.

## 7. Claim Boundary

P0 may establish dataset scale, traceability and split feasibility. It cannot establish classification improvement, optical cause, image quality, same-individual identity, human identifiability or biological mechanism.

## 8. Freeze Procedure

Before Dataset P0 execution, replace every draft threshold with a final value, record exact input/API cutoffs, and create `FROZEN_DATASET_P0.sha256` at the project root. The file uses standard `<64-char SHA-256><two spaces><path relative to project root>` lines. The Dataset freeze covers this protocol, `code/configs/dataset_p0.yaml`, `code/configs/scaleup_p0.yaml`, the exact 100-taxon table and the completed feasibility audit. Change the two dataset YAML statuses to `frozen` and regenerate all hashes as the final freeze action. Runtime code must verify the record and refuse execution or a PASS/STOP decision when the status/hash contract is incomplete.

`code/configs/benchmark.yaml` remains draft during dataset construction. It is added to a regenerated freeze record only after the feature-extraction and experiment controls are complete; this later addition cannot alter the already hashed Dataset P0 files or taxon table.

## 9. Preregistered Scale-Up Feasibility

This stage is metadata-only and cannot produce a Dataset P0 PASS. Its fixed taxonomy source is `E:/Datasets/iNaturalist-2021/metadata/train_mini.json`, SHA-256 `481b14dcabe945181e79c2cb8a65cf257e767468db8d94a9559b0ce639086060`. Only category records whose class is `Aves` are candidates.

The 80 taxa common to both inspected cohorts are retained. Candidate additions exclude those taxa and are ordered without image inspection: candidates sharing a genus with the seed set first, then candidates sharing a family, then all other Aves; ties use the iNaturalist 2021 category ID. A candidate resolves only when the current API returns exactly one active Aves taxon at rank `species` with an exact scientific-name match.

The metadata cutoff is `2026-08-18T16:00:00Z`. Queries use research-grade observations, explicit CC0/CC-BY/CC-BY-SA photo licenses, up to five pages of 200 observations per taxon, and only non-hidden photos. A query may stop early after 75 eligible distinct observers; this condition uses metadata IDs only. Existing observation and observer IDs are excluded. The first 120 ranked candidates are audited; the first 20 candidates with at least 50 independently eligible observations from at least 50 eligible observers form the proposed expansion. Failure to obtain 20 taxa stops this design without changing the cutoff, prefix, page count or thresholds.

## 10. Frozen Scale-Up Design

If Section 9 passes, freeze the exact 100-taxon table before final metadata selection. Retain the existing 80 species x 26 groups, add exactly 24 groups to each existing species, and add exactly 50 groups to each new species. The result must contain exactly 100 species and 5,000 observations. Every newly selected observation must use a previously unseen observer and observation; each new observer may contribute at most one selected group globally. Selection is a deterministic metadata-only MILP. Image content, sharpness, bird size and model outputs are forbidden selection variables.

The frozen record must additionally cover `code/configs/scaleup_p0.yaml` and the exact 100-taxon table. Image download may begin only after the frozen metadata collector satisfies the exact species/group counts and observer constraints.
