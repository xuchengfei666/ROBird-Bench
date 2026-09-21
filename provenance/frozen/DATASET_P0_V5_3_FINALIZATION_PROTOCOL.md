# ROBird-Bench Dataset P0-v5.3 Finalization Protocol

> Date: 2026-08-22 | Status: `FROZEN_P0_V5_3_FINALIZATION`

## Scope

v5.3 uses the immutable v5.2.5 canonical manifest as its base. The 14 v5.2.5
dHash candidates have been inspected side by side and recorded as visually
distinct. The only confirmed byte failure is observation `341743168`
(`Mareca strepera`, taxon `558439`), whose two photo IDs have identical
SHA-256 bytes. This version replaces exactly that observation.

## Replacement preflight

Candidate observations come from the complete frozen v5.1 census and must have
taxon ID `558439`, 2-5 licensed photos, an observation ID absent from v5.2.5,
an observer absent from v5.2.5, and no observation-level exact-byte history in
v5.2.4-v5.2.6. Candidates are ordered by the frozen seed and the key
`v5.3-staging`; the fixed expected order is recorded in the config.

All five eligible observations are downloaded into an E: staging cache before
selection. A candidate fails automatically if any photo byte repeats within
the candidate or matches a retained v5.2.5 byte. Candidate-to-base dHash pairs
are screening candidates only. If present, they require recorded side-by-side
visual decisions. The selected replacement is the first candidate in frozen
order with clean exact bytes and no visually confirmed duplicate pair.

## Final endpoint

The writer removes only observation `341743168`, inserts one accepted
replacement observation, preserves the frozen 100-class table, and requires
exactly 5,000 groups, at least 12,000 photos, 48-52 groups per taxon, unique
photo IDs, zero exact duplicate hashes and observer capacity one. It writes a
new v5.3 canonical manifest and finalization audit without changing older
artifacts.

The resulting manifest must still pass an independently frozen G0-G5 byte
audit. A non-empty dHash set is reviewed rather than automatically
quarantined. With no registered unseen holdout, a passing G0-G5 audit may
authorize development splits and model development only, not external
holdout claims.
