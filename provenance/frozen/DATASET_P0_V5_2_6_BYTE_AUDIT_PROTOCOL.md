# ROBird-Bench Dataset P0-v5.2.6 Byte Audit Protocol

> Date: 2026-08-22 | Status: `FROZEN_P0_V5_2_6_BYTE_AUDIT`

Apply the unchanged G0-G6 Dataset P0 predicates to the v5.2.6 canonical bytes.
Inspect every path and SHA-256, decode every image, check group and license
integrity, and generate a fresh dHash candidate set. A non-empty exact hash or
near-duplicate set keeps the decision at `STOP_DATASET_P0`; no candidate is
automatically labeled `distinct`. G6 still requires a separately registered
unseen holdout. No split, feature or model is allowed unless the resulting
audit explicitly authorizes it.
