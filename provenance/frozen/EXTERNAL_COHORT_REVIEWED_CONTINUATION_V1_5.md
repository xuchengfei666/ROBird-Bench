# External cohort reviewed continuation v1.5

## Correction and scope (2026-09-09, before new predictions)

用户要求完成剩余62对候选复核并启动实验。v1.4下载已完成；970关系与v1.3逐条
SHA256一致。新入口没有下载器，不重新查询来源或下载照片。历史v1.1--v1.4
代码、freeze和STOP全部保留。v1.4曾声称whole-cluster exclusion，但实际沿用
v1.3的single-photo exclusion；该功能当时未实现，不能引用为已完成证据。

## Adjudication

65个固定编号来自outcome-blinded contact-sheet index，逐对绑定source/observation/
reference identity、两侧路径、SHA256及dHash距离。AI-assisted visual review，不冒充
独立人工金标准。50个history匹配是可见不同主体/姿态/场景；12个同observation
候选保留为自然近重复冗余（不保证提供不同信息）；3个已审exact pairs组成同一
共享场景。无未决候选方可进入特征。结论只覆盖冻结筛查标记的候选，不宣称排除
所有可能泄漏。任一身份、字节、pair清单或既有freeze不匹配都失败关闭。

## Preserve all source data

full primary: 970 source relations / 969 photo IDs / 967 byte hashes / 339 groups /
13 availability-selected taxa。保留源标签，13类由100类census后可用性筛选得到，
87类短缺保留；同iNaturalist平台，不是原100类的无偏G6。original_g6_pass=false。

shared observations 377824923 and 377824924 (observer2132783) contain the same
three image-byte sets under two taxon labels. Only the sensitivity analysis omits
both complete observations: 964 relations / 337 groups / 13 taxa. No file/row is
deleted from the primary input. Taxon4328 observer count becomes9 from10 in this
conditional sensitivity; do not claim it independently passes the original gate.
Record same-input/single-label ambiguity across all subsets, not an optical cause.

## Frozen computation

Reuse v1.4 bytes/ledgers and verified metadata as read-only. DINO features computed
once per unique photo ID, then expanded to source relations with fixed row index.
Unchanged DINO state hash, transform and fp16; reuse local encoder cache if present.
24 existing checkpoints: 4 models x 3 seeds x 2 training policies. No training or
new model selection. Output probabilities remain100-dimensional. Retain all signs
of results; k1/all policies never pooled. Full and exclusion modes run serially:
A/B exhaustive true-k and nested subsets; C12x20 fixed membership controls per
mode; D fixed automatic quality proxies; E frozen5000-repeat statistics and all
model comparisons. Common-observation full/exclusion predictions should match;
whole-cohort score differences are composition sensitivity, not model improvement.
Shared image cache is reused for quality; outcome-blinded human templates remain
unfilled. Motion blur/occlusion ground truth and original G6 remain pending/false.

## Engineering and verification

New files: robird/external_cohort_v1_5.py (review/input gates, masks, comparisons),
run_external_cohort_v1_5.py (prepare/preflight/serial runner/hidden host), explicit
external_duplicate_decisions_v1_5.json and production-function tests. Freeze binds
parent chain, pair index/images, review record, source manifests/ledgers, code,
config and checkpoints. Per-stage atomic markers support resume without repeating
finished stages. OS lock prevents multiple workers. Hidden host records exit code
and local state every15 seconds without AI polling; no automated bypass on failure.
