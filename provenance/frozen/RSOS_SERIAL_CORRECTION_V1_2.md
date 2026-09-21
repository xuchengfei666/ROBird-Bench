# Serial runner v1.2 engineering correction

2026-09-08. Before any experiment the v1 runner parsed a text hash manifest as JSON.
The first repair was mistakenly applied to v1 in place; it was never executed.
v1 has now been restored byte-for-byte to its recorded hash.
The unused v1.1 freeze record contains stale hashes and is INVALID, never executable.
Use only run_rsos_serial_v1_2.py and FROZEN_RSOS_SERIAL_SUITE_V1_2.json.
No old experimental output was modified.

v1.2 changes the freeze parser and separates mutable continuation logs from immutable
scientific inputs. It binds ALL shared modules and inherited checkpoint/feature inputs.
New exclusion config explicitly enumerates historical observation manifests.
Scientific matrix, seeds, training parameters, metrics, E3 randomizations and
metadata bounds remain as specified in RSOS_SERIAL_SUITE_V1.md.
The process writes stage status; training epoch snapshots and completion markers
support continuation. No repeated model polling is needed.
