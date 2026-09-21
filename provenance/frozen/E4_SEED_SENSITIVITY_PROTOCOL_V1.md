# E4 Training-Seed Sensitivity

Date: 2026-09-05  
Status: frozen

## Question

Are the P4.1 conclusions about Deep Sets and Set Transformer stable across
independent training seeds?

## Frozen design

- Dataset, manifest, observer-disjoint split and DINOv2 features: unchanged
  ROBird-v5.3 development protocol.
- Models: `Deep Sets` and `Set Transformer` only. P4.1 hyperparameters are
  copied exactly; no architecture or optimizer change is allowed.
- Training seeds: `20260819`, `20260820`, `20260821`, predeclared before
  evaluation.
- Each seed trains from a fresh initialization on `train`, selects the best
  checkpoint by `validation` loss and is evaluated once on `development_test`
  after all six checkpoints have been written.
- Evaluation budgets, exhaustive subset cap, metrics and bootstrap settings are
  identical to P4.1. No development-test result is used for model selection.

## Reported quantities

For every model and budget report each seed's Macro Top-1, Micro Top-1, Top-5,
NLL, Brier and ECE, followed by mean, sample standard deviation and the range.
The result is a stability analysis, not a new winner gate.

## Stop conditions

- Stop if any seed fails training, provenance, checkpoint or evaluation
  integrity.
- Do not discard an underperforming seed or rerun with a new seed.
- Do not modify P4.1 or tune on development-test.
