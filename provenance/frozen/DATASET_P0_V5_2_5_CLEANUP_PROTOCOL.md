# ROBird-Bench Dataset P0-v5.2.5 Cleanup and Reselection Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_5_CLEANUP`

## Purpose

v5.2.4 passed metadata gates but stopped at the byte audit because 34 exact
hashes and 327 dHash near-duplicate candidates were present. This version is a
separate, deterministic metadata cleanup. It does not edit any v5.2.4 file,
does not label a near-duplicate `distinct`, and does not train a model.

## Fixed cleanup rule

1. For each repeated SHA-256 in the v5.2.4 canonical manifest, retain the
   smallest `photo_id` and quarantine the other rows.
2. Build the graph from the frozen 327 near-duplicate pairs. Ignore an edge
   already covered by step 1. Solve a binary minimum-cost vertex cover with
   cost 100 for a legacy ROQ photo and cost 1 for a scale-up photo. A stable
   SHA-256 tie term makes the objective deterministic. Covered photos are
   quarantined; no pair is declared distinct.
3. Remove quarantined photos. An observation with fewer than two remaining
   photos is quarantined as a whole group.
4. Keep all remaining seed groups. Treat remaining v5.2.4 scale-up groups as
   selectable candidates after cleanup, and add unused groups from the complete
   frozen v5.1 census pool. Exclude old observation IDs, seed observers and
   groups quarantined in step 3 from the added pool.
5. Run the existing bounded-total MILP with observer capacity one globally.
   The final endpoint remains exactly 100 taxa, 5,000 groups, 48-52 groups per
   taxon and at least 12,000 photos. The candidate count is derived from the
   number of retained seed groups; it is not hand-tuned.

The resulting metadata manifest is an input to the existing byte downloader.
Only after the new bytes are downloaded may the normal exact-hash and dHash
audits decide whether Dataset P0 passes. A new near-duplicate set requires a
new version; it is never auto-approved in this version.

## Immutable inputs and outputs

Inputs are the v5.2.4 canonical manifest, its STOP audit, its candidate CSV
and sidecar, the frozen v5.2.4 class table, and the completed v5.1 census
snapshot. Outputs are `p0_metadata_photos_v5_2_5.csv`,
`cleanup_audit_v5_2_5.json`, and runtime logs. No old artifact is overwritten.

## Decision boundary

`PASS_METADATA_TO_DOWNLOAD` authorizes only the byte download. The Dataset P0
decision remains false until G0-G5 are rerun on the v5.2.5 canonical bytes;
G6 unseen holdout remains a separate registration requirement.
