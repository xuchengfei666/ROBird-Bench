# Real-case photographs and provenance

The six photographs embedded in `figure_cases.pdf` and `diagrams_editable.pptx`
retain the RGB pixels of the previously downloaded source photographs.
Standalone `.image` files are not included in this public repository; the
recorded source URLs and byte hashes in `case_selection.json` support retrieval.
No enhancement, cropping, synthetic replacement or retouching was applied.
Figures scale photographs while retaining aspect ratio and field of view.
The frozen model preprocessing, including its centre crop, is distinct from
this full-frame display.

| Observation | Photo IDs | Recorded attribution | Recorded license |
|---|---|---|---|
| 6718263 | 8526241, 8526294 | Stosh Morency | CC BY |
| 13807886 | 20368359, 20379432 | no rights reserved | CC0 |
| 22037163 | 34143218, 34143544 | no rights reserved | CC0 |

Source observation pages:

- https://www.inaturalist.org/observations/6718263
- https://www.inaturalist.org/observations/13807886
- https://www.inaturalist.org/observations/22037163

`case_selection.json` retains individual image URLs, license codes, attribution,
original byte hashes and all displayed prediction probabilities. License
versions absent from the source manifest have not been invented. Third-party
photo rights remain with their rights holders, separately from the manuscript.

Selection used the lowest numeric observation ID in each specified event
among 611 exactly-two-photo groups for DINOv2 Deep Sets, full-set training,
seed 20260819. The three eligible event counts were 71 (same wrong class),
45 (destruction) and 9 (rescue). The rule was fixed before viewing candidate
photos but after aggregate analysis; these are not representative samples
or a preregistered study. All selected native and pooled classes agree.

Species text is the recorded source label, not independent expert adjudication.
