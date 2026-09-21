# Release preparation checks (22 September 2026)

These checks test the release tooling and previously frozen evidence, not new
experiments or clean-machine end-to-end training.

- Portable test selection: **215 passed, 20 deselected, 2 warnings**. The wrapper
  explicitly lists the historical fixture-dependent exclusions.
- Unfiltered historical suite on the minimal release: **215 passed, 12 failed,
  8 errors, 2 warnings**. Missing historical census/review fixtures are not
  included merely to turn the test count green. See `tools/test_portable.py`.
- CPU replay of all actual subsets for the DINOv2 Deep Sets full-set-trained
  checkpoint, seed 20260819, passed coverage/probability checks. External
  species-macro accuracy for budgets 1--5 was 0.582327, 0.658712, 0.668625,
  0.641097 and 0.595390. Supports change across these budgets; these are not
  fixed-five curves and not three-seed manuscript means.
- The five regenerated quantitative figure PNGs are byte-identical to the
  manuscript v6 figures on the recorded Windows analysis environment.
- The editable diagrams use Times New Roman with integer point sizes. Embedded
  case photographs retain original source RGB pixels. These are illustrative
  examples selected by the documented rule, not independently relabelled data.
- Live source retrieval smoke test: the first two unique external-manifest
  photographs downloaded successfully and matched their recorded SHA-256 hashes.
  This does not establish future availability of every upstream photograph.
- `tools/verify_release.py` distinguishes present checked files from absent
  bulk files. A fresh clone without release archives is not a full-evidence
  validation. Asset sizes and SHA-256 hashes are in `release_assets.json`.

An unauthenticated Windows clone exposed automatic LF-to-CRLF conversion by
the user's global Git configuration. Code v1.0.2 adds `.gitattributes` with
`* -text` to preserve the recorded bytes on every platform. Source hashes and
scientific files are unchanged; earlier published tags are not rewritten.

Use code **v1.0.2** with evidence assets **v1.0.0**. The code patch adds portable
plotting, retrieval, test selection and documentation; it changes no frozen
experimental output. The original v1.0.0 source tag has not been rewritten.
