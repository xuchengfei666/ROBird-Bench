# Reproduction levels and fixed records

## 1. Inspect and check the paper's numerical evidence

The Git tree includes the headline tables and manuscript figure sources.
`tools/verify_release.py` validates all present files in FILE_MANIFEST.json,
checks cohort cardinalities, confirms the main-effect signs and checks the
Deep Sets native/pool event identity. It explicitly reports bulk files absent
from a fresh clone. Missing archives are not silently counted as verified.

## 2. Recompute from stored probabilities

The release assets contain actual-subset predictions and regrouping mappings,
not just final accuracy numbers. `tools/recompute_prediction.py` validates
coverage and recomputes per-budget species-macro accuracy without a GPU.
Detailed bootstrap, sign-flip, pooled probabilities, ECE and grouping algorithms
are preserved in `code/src/robird/budget_metrics_v2_1.py`,
`external_completion_v1.py` and `explanatory_suite_v1*.py`. The original
independent checks are in `code/scripts/audit_explanatory_science_v1.py`.

Recorded seeds: training 20260819/20260820/20260821; explanatory randomization
2026091900 through 2026091919; explanatory inference seed 20260919. Natural
observations, coarse-matched pseudo-groups and full external observations are
different supports. Repeat ranges are not confidence intervals. Holm128 and
Holm16 are separate fixed families. No new seed search was performed for release.

## 3. Re-evaluate or retrain the original models

Frozen study scripts and configs, learned head checkpoints, cached features,
class order and split are provided. The historical runner scripts contain
original workstation paths and hash guards. They cannot simply be launched
on a new workstation without a separately documented path-only adaptation.
Do not edit a frozen hash file and present a changed run as the original.
For a new replay use the models/data/training modules, the published split,
the recorded architecture/preprocessing and a distinct output directory.

The public release has been checked for arithmetic and artifact integrity;
it does not claim a clean-machine end-to-end retraining test. GPU nondeterminism,
package builds and changes to upstream source images can affect a new replay.
The environment is Python 3.10.18, torch 2.5.1, torchvision 0.20.1 and the pinned
analysis packages. Select the appropriate official torch CPU/CUDA distribution
for the host; an environment name such as `pytorch1.0` is not a torch version.

## Figure/table map

- Figures 3 and S1(a): gain/fixed-five/training-policy source CSVs.
- Figure 4: grouping.csv and attrition.csv.
- Figure 5: error_means.csv (474 matched-support groups).
- Figure 6: pooling.csv and curves.csv (1,121 full external groups).
- Figure S1(b): matched_unmatched_natural_descriptive.csv; group taxon supports differ.
- Figures 1, 2 and 7: protocol diagram, support counts and the fixed illustrative
  case selection. They do not add experimental results.

No third-party photo is replaced with a generated or visually enhanced image.
No unsuccessful historical gate is promoted to a pass in this release.
