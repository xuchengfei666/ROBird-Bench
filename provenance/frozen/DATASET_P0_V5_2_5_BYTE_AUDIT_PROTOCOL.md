# ROBird-Bench Dataset P0-v5.2.5 Byte Audit Protocol

> Date: 2026-08-21 | Status: `FROZEN_P0_V5_2_5_BYTE_AUDIT`

This audit applies the unchanged ROBird Dataset P0 G0-G6 predicates to the
v5.2.5 canonical manifest after its cleanup download. It verifies file
existence, decodability, byte hashes, licenses, group integrity, exact hashes
and a fresh dHash candidate set. The v5.2.5 cleanup decision is not a P0
pass; it only authorizes this byte audit.

The near-duplicate candidate CSV is generated from the v5.2.5 bytes. A
non-empty candidate set leaves G5 false and does not authorize downstream
features, splits or models. Any later quarantine/reselection must use a new
versioned protocol and preserve this audit unchanged.

G6 remains false until an unseen, observer- and observation-disjoint holdout
is registered with a matching hash. Only a report with all required gates true
may authorize Dataset P0 downstream work.
