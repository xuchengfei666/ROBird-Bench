# ROBird-Bench Dataset P0-v5.3 Byte Audit Protocol

> Date: 2026-08-22 | Status: `FROZEN_P0_V5_3_BYTE_AUDIT`

Apply the unchanged Dataset P0 G0-G6 predicates to the final v5.3 canonical
manifest. Inspect all file paths and SHA-256 values, enforce provenance,
scale, species support, group integrity and zero exact duplicate hashes, and
regenerate the global 8 x 8 dHash screening candidates.

The dHash candidate set must match a complete pair-level visual review. dHash
is a screening signal rather than an automatic duplicate decision. The review
is explicitly recorded as Codex-assisted visual inspection, not independent
human-expert annotation.

If G0-G5 pass and no unseen holdout is registered, the only authorized decision
is `CONTINUE_DEVELOPMENT_ONLY`: build deterministic observer-disjoint
development splits and permit development experiments, while withholding
external-holdout claims.
