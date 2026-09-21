# ROBird-Bench v5.3 DINOv2 Feature Protocol

> Date: 2026-08-22 | Status: `FROZEN_FEATURE_P0_V5_3`

## Claim scope

This is a development-only, frozen feature extraction stage downstream of the
v5.3 G0-G5 Dataset P0 pass. It does not create an unseen holdout or authorize
confirmatory external-generalization claims.

## Encoder and preprocessing

- Encoder: timm `vit_small_patch14_dinov2.lvd142m`, pretrained on LVD-142M.
- Output: 384-dimensional pre-logit embedding, converted to float32 and L2
  normalized per photo.
- Input: RGB, bicubic resize of the shorter side to 256, center crop 224 x 224,
  ImageNet mean/std normalization.
- Inference: evaluation mode, no gradients, CUDA autocast float16 when CUDA is
  available; saved values are float32.
- Ordering: the v5.3 canonical manifest sorted by `row_id`; the feature index
  binds every `feature_row` to exactly one `photo_id`.

The 224-pixel encoder is the first development baseline. It is not represented
as a high-resolution final model; resolution/backbone comparisons require
separate frozen feature versions.

## Gates

Extraction requires the exact v5.3 canonical manifest, Dataset P0 audit with
decision `CONTINUE_DEVELOPMENT_ONLY` or `CONTINUE_TO_COMMON_BENCHMARK`, and the
passing v5.3 split audit. The output must contain 14,092 finite rows, unique
photo IDs, dimension 384 and nonzero L2 norm before normalization. The audit
records the loaded model-state SHA-256, timm/torch/torchvision versions,
preprocessing contract and hashes of the feature matrix and index.

Feature bytes remain on E:. No classifier or aggregator is trained by this
protocol.
