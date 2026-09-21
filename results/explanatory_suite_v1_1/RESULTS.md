# Explanatory suite v1 — automatic evidence summary

这是post-hoc解释性补实验；完成不等于假设成立。旧v2.2和G6=false不改变。

难度匹配共同支持：474 observations / 53 taxa / 1025 photos。
筛选损耗见matching_attrition_by_taxon.csv；代理为独立ResNet线性k1头，开发validation分箱。

## Real-group paired tests

三seed平均同组差值；百分点评估，observer cluster点态95%CI，新增Holm16家族。

| Encoder | Model | Contrast | Budget | Effect pp | 95% CI pp | Holm16 p |
|---|---|---|---|---:|---|---:|
| dinov2 | deepsets | native_minus_pool | 2 | +0.145 | [-0.648, +0.955] | 1.00000 |
| dinov2 | deepsets | native_minus_pool | 3 | -1.386 | [-2.735, -0.123] | 0.52020 |
| dinov2 | mean_feature | native_minus_pool | 2 | -0.279 | [-0.986, +0.408] | 1.00000 |
| dinov2 | mean_feature | native_minus_pool | 3 | -0.972 | [-1.914, -0.122] | 0.49600 |
| dinov2 | probability_mlp | native_minus_pool | 2 | +0.000 | [+0.000, +0.000] | 1.00000 |
| dinov2 | probability_mlp | native_minus_pool | 3 | +0.000 | [+0.000, +0.000] | 1.00000 |
| dinov2 | set_transformer | native_minus_pool | 2 | +1.155 | [+0.407, +1.892] | 0.02520 |
| dinov2 | set_transformer | native_minus_pool | 3 | +1.961 | [+0.606, +3.243] | 0.05720 |
| resnet50 | deepsets | native_minus_pool | 2 | +0.433 | [-0.360, +1.234] | 1.00000 |
| resnet50 | deepsets | native_minus_pool | 3 | -0.477 | [-1.912, +0.913] | 1.00000 |
| resnet50 | deepsets | k1_to_k2 | 2 | +6.185 | [+5.135, +7.269] | 0.00160 |
| resnet50 | deepsets | k2_to_k3 | 3 | +1.987 | [+0.819, +3.195] | 0.02340 |
| resnet50 | probability_mlp | native_minus_pool | 2 | +0.000 | [+0.000, +0.000] | 1.00000 |
| resnet50 | probability_mlp | native_minus_pool | 3 | +0.000 | [+0.000, +0.000] | 1.00000 |
| resnet50 | probability_mlp | k1_to_k2 | 2 | +5.191 | [+4.273, +6.143] | 0.00160 |
| resnet50 | probability_mlp | k2_to_k3 | 3 | +2.624 | [+1.682, +3.603] | 0.00160 |

非显著不等于等效；Probability MLP原生与概率池化是同义负对照，不算独立方法成功。
完整seed、逐类、nested regression/correction分别见real_seed_curves、real_per_taxon、nested_seed_macro。

## Common-support grouping controls

以下k2 native差值为control minus real，20重复先各平均3seed，再列重复范围；不是CI。

| Encoder | Model | Control | Mean gain pp | Repeat range pp |
|---|---|---|---:|---|
| dinov2 | deepsets | difficulty_matched | +9.379 | [+8.265, +10.391] |
| dinov2 | deepsets | unrestricted_common | +9.899 | [+8.666, +11.136] |
| dinov2 | mean_feature | difficulty_matched | +9.103 | [+7.177, +10.322] |
| dinov2 | mean_feature | unrestricted_common | +11.164 | [+9.351, +13.466] |
| dinov2 | probability_mlp | difficulty_matched | +6.100 | [+5.245, +7.449] |
| dinov2 | probability_mlp | unrestricted_common | +7.848 | [+6.242, +9.115] |
| dinov2 | set_transformer | difficulty_matched | +7.464 | [+6.443, +9.251] |
| dinov2 | set_transformer | unrestricted_common | +9.038 | [+7.078, +11.179] |
| resnet50 | deepsets | difficulty_matched | +7.842 | [+6.961, +9.243] |
| resnet50 | deepsets | unrestricted_common | +9.088 | [+7.495, +10.420] |
| resnet50 | probability_mlp | difficulty_matched | +3.499 | [+2.024, +5.605] |
| resnet50 | probability_mlp | unrestricted_common | +5.231 | [+2.421, +6.936] |

难度匹配后差距保留只能说明超出该粗代理分箱的残余分组效应；
差距缩小支持难度构成贡献，不能将残余全部归因错误依赖。
joint_wrong_pair、same_wrong_pair和set-correction/regression见control_repeat_effects；
残余难度差、同observer来源比例见matching_balance_summary，不得省略。

## Boundaries and interpretation

ResNet两模型四个预定预算对比中，0项未同时满足正点估计与Holm16<.05；完整正负证据均在上表。
Full-support E3编码器对比见full_support_e3_encoder_comparison.csv，不能与common-support数值直接相减。
同平台、可用性选择56类而非无偏100类；AI辅助去重非独立人工gold standard。
未做人工质量、未验证模糊/遮挡/光学因果、未提出或训练新算法。
下一步应结合匹配损耗、连续难度残差、错误共现与聚合对照共同判断主线，不按单个p值改故事。
