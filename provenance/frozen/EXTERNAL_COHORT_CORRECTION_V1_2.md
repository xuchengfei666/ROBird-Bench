# External cohort v1.2 engineering correction — 2026-09-09

v1.1 stopped in metadata_provenance with TypeError: cannot convert the series to
int. The development manifest contains missing scientific names alongside present
names for the SAME taxon/class pair. Deduplicating three columns did not produce
a unique taxon index. No photo download or evaluation occurred. Old code, freeze,
config and exception artifact remain immutable.

v1.2 constructs the identity map separately: each taxon MUST have exactly one
integer class_index, each class_index one taxon, and one unique NONEMPTY scientific
name. Missing display names may coexist with that unique name; conflicting class
IDs or nonempty names still STOP. No arbitrary first-row class choice. No legacy
manifest is edited. Exactly the same 13 taxa, 339 groups and 970 photo IDs must
survive the real-data integration test, with all exclusion rules unchanged.

New wrapper reuses the unmodified v1.1 pipeline with only its materialize function
replaced by the corrected version. Config/checkpoints/thresholds/analysis remain
unchanged; isolated v1.2 E: and report directories retain the original failure.
Before release: synthetic missing-name/conflict tests, full regression, full real
census materialization, then hidden serial start confirmed by committed photo
ledger(s), not just PID/preflight. No continuous monitoring or false completion.
