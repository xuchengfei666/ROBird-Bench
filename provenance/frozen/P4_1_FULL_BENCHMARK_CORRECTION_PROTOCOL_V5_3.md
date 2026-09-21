# ROBird-Bench v5.3 P4.1 Runner Correction

> Date: 2026-08-22 | Status: `FROZEN_P4_1_FULL_BENCHMARK_V5_3`

P4.1 is an engineering-only continuation after the frozen P4 runner failed
before training because it resolved the project root one directory too deep
and attempted `code/code/scripts/train.py`. No P4 output, checkpoint or metric
was created by that failed attempt.

P4.1 changes only the runner's project-root resolution from
`Path(config_path).parents[1]` to `parents[2]`. It preserves the P4 model list,
all-photo training settings, v5.3 data/feature chain, P3 fixed result and
development-only claim scope. The corrected runner writes isolated outputs
under `benchmark_v5_3_p4_1`; no P4 file is overwritten.
