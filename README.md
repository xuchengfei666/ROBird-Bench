# ROBird-Bench

Code and research data for **Natural photo grouping and aggregation trade-offs in multi-photo bird recognition**, by Chengfei Xu and Yunrui Jiang.

This is an empirical observation-set study, not a claim of a new state-of-the-art aggregation architecture. The frozen development cohort contains 100 taxa and 5,000 observations. The expanded, same-platform external cohort contains 56 taxa, 1,121 observations, 3,090 photographs and 759 observer IDs. Coarse difficulty matching retains 474 observations across 53 taxa. The original complete 100-taxon external gate was not met.

## Main findings and boundaries

- A second photograph improves species-macro accuracy for the tested encoders and aggregators.
- Regrouping a fixed photo pool changes recognition, including after the specified coarse difficulty matching. This label-informed control is not a deployable recognition method or a uniquely identified causal mechanism.
- For DINOv2 Deep Sets, greater rescue is largely cancelled by greater destruction. Its native-versus-pool net difference is +0.145 percentage points, not statistically significant after Holm16 correction.
- Mean gains coexist with harmful additions. The study does not establish a universal two-photo optimum, absence of all leakage, or cross-platform generalisation.

## Release layout

Use code tag **v1.0.1** (portable tooling and figure documentation corrections).
The evidence archives remain under release **v1.0.0**, with unchanged archive
hashes. The v1.0.0 auto-generated source archive predates these tooling fixes;
use the v1.0.1 code snapshot with the v1.0.0 evidence files. This version split
does not change any frozen experiment, probability, support or result.

| Location | Contents |
|---|---|
| `code/src`, `code/scripts`, `code/configs`, `code/tests` | Study implementations, historical versioned runners, fixed configurations and unit tests |
| `data/manifests` | Development/external photo retrieval manifests, class order, split and matched-support membership |
| `results` | Compact main-result tables available without downloading bulk assets |
| `figure_source_data` | Numerical sources for manuscript figures |
| `provenance/frozen` | Original frozen protocols and hash records, including unsuccessful historical gates |
| `release_assets.json` | Independently extractable release ZIP volumes and their SHA-256 hashes |
| `FILE_MANIFEST.json` | Per-file hashes and original-source hashes; records transformed public manifests |
| GitHub release `v1.0.0` | Detailed result tables, subset probabilities, regrouping maps, learned heads and feature caches |

Raw iNaturalist image bytes are **not bundled**. Retrieval URLs, identifiers, original byte hashes, license codes and attribution are provided. Each photograph remains under its own source license; it is not covered by the code license. Some original images may later become unavailable or change. The retrieval helper fails on byte mismatches rather than silently replacing images. No raw API user profiles or precise coordinates are included. Public source IDs and photo attribution remain necessary for provenance; these records are not claimed to be anonymous.

## Quick verification

Python 3.10 is the recorded study version. Install the portable analysis dependencies:

```bash
python -m pip install -r requirements-analysis.txt
python tools/verify_release.py
```

This verifies local file hashes, support sizes, fixed class order, positive one-to-two effects, grouping effects and the native/pool result. It does not train models or download the complete evidence archive.

To download release archives with hash checks:

```bash
python tools/fetch_release.py --group results
python tools/fetch_release.py --group external-predictions-and-controls
python tools/fetch_release.py --group development-models-and-predictions
```

Archives are independently extractable; all extracted paths are relative to this repository. Downloads require approximately 5.4 GB in total, plus space for extraction. Use `--extract` to extract safely after hash verification. Existing verified archives are reused. No Git LFS subscription is required.

To validate and recompute actual-budget accuracy from one released prediction file:

```bash
python tools/recompute_prediction.py --predictions artifacts/external_expansion_v2_2/runs/dinov2-deepsets-all-seed20260819/predictions.csv --manifest data/manifests/external_primary.csv --output recomputed.csv
```

All non-empty photo subsets must be present; the check rejects foreign photos, invalid probabilities and incomplete coverage. Accuracy is averaged within observations and then taxa; probability-quality metrics give observations equal weight. Three-seed results require the three specified seeds; one run is not the manuscript mean. Run `python tools/test_portable.py` after installing the full environment. This explicitly deselects 20 historical fixture-dependent cases, while retaining them in the source. The unfiltered historical suite produced 215 passes, 12 failures and 8 errors on the minimal release clone because historical census/review fixtures are absent. Frozen historical runner entry points are **not advertised as portable one-command experiments**. See `REPRODUCIBILITY.md`.

For licensed source-photo retrieval, use `python tools/retrieve_photos.py --manifest data/manifests/external_primary.csv --output downloaded_photos --limit 2` to check two images before retrieving the full cohort. A changed or missing source is reported as a failure, never silently substituted.

## Citation and licenses

To regenerate the five quantitative figures without the original workstation:

```bash
python tools/plot_figures.py
```

Outputs are written to `generated_figures/`. Editable protocol/case diagrams
and licensed-photo provenance are in `figures/`. See `VALIDATION.md` for the
exact checks performed and their limitations.

See `CITATION.cff`, `LICENSE` and `DATA_LICENSE.md`. This is a versioned GitHub software/data release, not a claimed journal acceptance or a DOI archive. Do not invent a DOI when citing it. Author-created code is MIT licensed; author-created derived tables are CC BY 4.0. Third-party photos, dependencies and pretrained encoders retain their own rights.

The study received no specific funding and the authors declare no competing interests. Generative AI assisted planning, code, analysis interpretation, duplicate screening and manuscript/figure preparation; it is not an independent annotator or author. Releasing evidence does not by itself establish journal-policy compliance or replace author responsibility.
