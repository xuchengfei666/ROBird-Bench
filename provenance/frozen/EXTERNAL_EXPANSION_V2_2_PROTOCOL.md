# External expansion v2.2 — unique-frame continuation

2026-09-19，用户“继续推进 就这样”批准审计后的独立处理与串行评估。
本版不重抓数据、不删除照片、不改v2/v2.1、旧13类结果、审计及原G6=false。
审计属于AI辅助视觉判读而非独立人工金标准，候选之外没有穷尽无重复保证。

## Primary cohort fixed before new predictions

复用v2.1全部3107条已下载记录，来源规则和100-way类映射不变。只将已封存的
9对exact和4对视觉同帧在各自observation内建连通分量，最小photo ID为代表；
64对同组不同帧正常保留，287history和28cross-observation distinct不删。
不能用图像质量、标签难度或模型结果选代表。不补抓、不从前五张以外补照片。
折叠后仅余1张的4组不参与多图评估；文件全留。新主清单3090照片/1121组/
759 source observer IDs/56taxa>=8，其中50taxa>=10。这些是输入一致性预期，
不是可随结果调节的新门槛；数量不符应STOP查明。

另固定exclude_affected_observations敏感性：从源清单整组排除全部13个有重复的
observation，3063照片/1112组/755observer/56与50taxa。它是主清单完整观察组的
严格子集，各组照片不变，因此可直接过滤同一组的冻结预测，不能复用改变了组内
照片数的预测。敏感性不能取代主结果，不重新训练。

## Gates and provenance

冻结parent 1259输入、review_done/impact_done递归证据、源download/metadata、
reference signature、配置、此协议及新代码/测试。逐图复验原download ledger与
SHA/dHash/尺寸；历史reference bytes复验。新清单严格是源行子集（strict10按
处理后实际observer支持重新计算），继续验证license/attribution/URL/source IDs/
observer与observation隔离、每taxon/observer最多2组、2--5张、原100类索引。
已审392候选的重复边不能双端保留；已知same-frame不计新增photo预算。子集与
字节不变意味着原候选集合已覆盖本版原阈值下的候选，无需换阈值重新判读。
新data_gate_done必须同时绑定主清单、敏感性清单和完整去重映射。它只授权此新
版本模型评估，不把v2.1的duplicate_done伪造为PASS。

## Frozen inference and statistics

- 复用v2.1 resolved config的24个checkpoint：4模型×all/k1×3seed，固定100-way；
  使用原DINOv2 encoder hash、224中心裁剪、384维归一化特征、真正k=1..5穷举子集。
- 不训练，不调参，不修改随机种子、bootstrap=5000、signflip=9999或置信方法。
- 主清单先完成24项实际预测，再进行12个all-trained模型各20次E3同taxon、
  同cardinality跨observation重组（label-informed诊断，不是独立现实观察）。
- 主/整组排除敏感性均报告full8与strict10、逐seed和3seed指标均值、
  macro/micro/top5/NLL/Brier/ECE、逐物种及fixed-n5、paired 1->2及2->3/4/5。
  沿用各分析64项Holm；另附全部128项Holm，不把新增敏感性当独立确认。
- 敏感性只从主预测过滤未改变的完整观察组，复用概率而非重跑网络。
  不另做敏感性E3（重组总体已改变，不可假装过滤旧E3即可复用）。
- 附主/敏感性的逐seed/平均seed准确率和与development匹配物种/组大小的来源差；
  只作描述性差异，不作因果解释。此次不重做已取消的人类质量标签或拓展模型矩阵。
- 空支持不记0、不把不同k可用队列直接相比；n5仅108组，不能代表全部1121组。
- 单worker串行、原子分片恢复；hash错/科学门槛STOP，工程错误记录后有限恢复。
  不进行AI高频轮询。旧缓存不复制图像；新features/predictions在E盘。

## Implementation and completion

入口 code/scripts/run_external_expansion_v2_2.py：
--prepare冻结；--data-only提交并复验数据门槛；--host隐藏后台完整串行续跑。
小产物 code/results/external_expansion_v2_2/；
大产物 E:/Datasets/ROBird-Bench/external_expansion_v2_2/。
完成marker绑定数据、特征、24评估、12 E3、两套统计、128校正和来源差；
称EXPANDED_EXTERNAL_UNIQUE_FRAME_VALIDATION_COMPLETE，不称原无偏100类G6通过。
