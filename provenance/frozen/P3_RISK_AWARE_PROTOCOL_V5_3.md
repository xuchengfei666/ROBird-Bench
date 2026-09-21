# ROBird-Bench v5.3 P3 Risk-aware Latent-view Reliability Aggregation

> Date: 2026-08-22 | Status: `FROZEN_P3_RISK_AWARE_V5_3`

## Question

Can a learned, permutation-invariant reliability model reduce harmful-view
degradation without hard-coding image sharpness, selecting views from labels,
or forcing a fixed fraction of photos to be discarded?

## Fixed inputs and claim scope

- v5.3 canonical manifest, observer-disjoint splits and DINOv2 ViT-S/14 features;
- frozen `mean-feature-k1` classifier used to initialize the photo-logit head;
- P1 task-signal STOP and P2 harmful-view STOP are immutable negative evidence;
- train observations are used for optimization, validation for checkpoint selection;
- only `development_test` is evaluated in this P3 run; no G6 unseen-holdout claim.

## RLRA method

For each photo feature `f_i`, a shared classifier produces logit evidence `z_i`.
The reliability head receives `f_i`, scaled `z_i`, top-two margin, normalized
entropy and centered-logit disagreement with the observation consensus. Its
scalar output is converted to masked softmax view weights. The weighted logits
are interpolated with masked mean logits by a learned conflict gate. All inputs
are observed photos; no image quality cache or test outcome is used at
inference.

Training minimizes:

`CE(full) + lambda_cvar * CVaR_q(CE(stochastic dropout subsets))`
`+ lambda_consistency * KL(full || dropout predictions)`.

Stochastic masks retain each real view with the frozen keep probability, always
retain at least one view, and use a fixed single-view anchor probability. The
CVaR term is computed per observation over the configured stochastic passes.

## Comparisons and evaluation

RLRA is compared with the frozen mean-logit/mean-feature baseline on exactly
the same deterministic exhaustive subsets for budgets 1-5. Metrics include
macro/micro Top-1, NLL, Brier, ECE, paired corrections/regressions and a
species-stratified bootstrap interval for the budget-1-to-2 delta.

## Frozen continuation gate

On the fixed common cohort of 750 observations eligible at budgets 1 and 2,
all conditions must hold:

1. RLRA budget-2 macro Top-1 is at least `0.5` percentage points above the
   frozen mean baseline;
2. RLRA budget-2 minus budget-1 macro delta is at least `-0.3` percentage
   points;
3. RLRA corrections exceed regressions relative to its own budget-1 output.

If any condition fails, the formal status is `STOP_P3_RISK_AWARE_GATE` and no
development-test hyperparameter search or unregistered method family is
authorized. A pass remains development-only until G6 is registered.
