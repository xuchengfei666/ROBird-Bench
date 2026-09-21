# Third-party dependencies

This repository distributes original study code, not copies of the pretrained
encoder source trees. Dependencies are installed separately and retain their
upstream licenses. See their own distributions for license text and notices.

- DINOv2: https://github.com/facebookresearch/dinov2 . The study used ViT-S/14
  with the recorded checkpoint and preprocessing hashes in the frozen records.
- PyTorch: https://github.com/pytorch/pytorch .
- torchvision / ResNet-50: https://github.com/pytorch/vision . The study used
  `ResNet50_Weights.IMAGENET1K_V2`; torchvision's recorded URL and transform
  information are retained in the feature audit records.
- NumPy, pandas, SciPy, scikit-learn, Pillow, PyYAML, tqdm, pytest and matplotlib
  retain their own package licenses. No installed environments are bundled.
- iNaturalist observations/photographs: https://www.inaturalist.org . Per-photo
  source identifiers, license codes and attribution govern image reuse.

The architecture adaptations in `code/src/robird/models.py` and `rsos_suite_v1.py`
are the study implementations. Deep Sets, Set Transformer and the cited prior
literature should still be cited when these implementations are used.

The private authoring runtime and journal template are not part of the research
software license. Scientific figures are reproducible with the supplied Python
plotting code and numerical sources; editable diagram files are separate artwork.
