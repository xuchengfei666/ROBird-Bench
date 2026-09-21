# ROBird-Bench Dataset P0-v5.2.4 Byte/Provenance/Duplicate Audit Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_4_BYTE_AUDIT` | no feature/model execution

## 1. Purpose

Audit every downloaded v5.2.4 photo against the canonical metadata manifest, including file existence, exact SHA-256, decodability, schema/provenance, license, group integrity, exact duplicate hashes and dHash near-duplicate candidates.

## 2. Fixed Inputs

The canonical manifest and E: download ledger are outputs of the frozen v5.2.4 metadata/downloader protocol. The audit config uses only the frozen manifest path, file hashes, allowed licenses and duplicate threshold; it does not alter source selection or class membership.

## 3. Stop Rules

Any missing/undecodable file, hash mismatch, invalid provenance, exact duplicate hash, failed group/support gate or unresolved near-duplicate candidate prevents a Dataset P0 PASS. A non-empty near-duplicate candidate list is emitted for human review; no automatic `distinct` decision is allowed. Holdout registration remains a separate G6 requirement.

## 4. Boundary

This stage may write only audit and near-duplicate candidate artifacts. It must not extract features or train models. Observer-disjoint split construction remains locked until the audit gates pass and any required duplicate review is resolved.

## 5. Freeze Contract

`FROZEN_DATASET_P0_V5_2_4_BYTE_AUDIT.sha256` binds this protocol/config, the frozen audit scripts, canonical manifest/metadata audit, E: ledger and class/replacement freeze chains.
