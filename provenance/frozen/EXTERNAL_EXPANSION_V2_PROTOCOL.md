# External Expansion v2 — 2026-09-18

## Authorization and estimand

用户授权在独立版本略微降低支持量门槛、扩大外部验证。不按模型准确率挑物种；
不以“面子”作为科学通过门槛。旧13类结果、旧G6=false及所有冻结文件不变。
本版本是在看过development和旧external结果之后设计的追加验证，不能称为整个
研究的事前预注册。新队列在新预测前冻结，并排除旧external的观察者/观察/照片。
这是同平台、metadata-availability-selected cohort，不是100类无偏抽样。

## Frozen sampling rules

- 同一development_v5_3的100 taxa、class_index及精确species taxon_id；不纳入子种。
- 仍用research grade、created_d2=2026-09-08T00:00:00Z、降序observation ID。
- 照片逐张检查cc0/cc-by/cc-by-sa、attribution、合法source URL；每组2–5张，
  按photo ID选前5。重复photo ID、历史photo/observation/observer均排除。
- 最低支持由10降到8个不同source observer IDs；这些ID不证明真实人的独立性。
  同时报告保持旧标准的>=10 observer taxa子集，绝不把8当成10。
- 每个taxon最多100页、每页200条；达到20个新observer IDs时停止处理。
  每个observer在一个taxon最多2组，最多40组/taxon；跨taxon共享observer允许，
  下游使用observer-cluster推断。固定规则不随可行类数/准确率改变。
- 复用旧census哈希核验的前缀响应并重新过滤；若未达目标且旧页未耗尽，使用
  id_below=最后已检索页最小ID继续，不能重复取前五页。新增响应压缩存E盘。
- 明确区分TARGET_REACHED、SOURCE_EXHAUSTED、SEARCH_BUDGET_TRUNCATED。
  最后者不是平台数据短缺；所有100类都写入attrition/coverage表。
- 旧缓存和新响应取得时间不同；固定creation cutoff不等于历史状态快照。
  taxonomy/license/grade以各响应取得时状态为准，记录各自日期及SHA256。
- API单进程每次请求至少间隔2秒；每UTC日最多5000次（重试也计数）；
  429/5xx/网络故障指数退避，尊重Retry-After；达到每日额度由后台程序等待。
  至多5次即时重试，host至多3次续跑工程/网络失败；科学/重复门槛不重试。

## Isolation and decision sequence

1. 冻结本协议、v2代码/config、旧cache链、development类表、历史manifest、
   旧external清单和24个既有checkpoint；不修改旧freeze。
2. metadata census覆盖100类；确定性包含所有>=8支持taxa，无准确率参与。
   若<=13类，记录NOT_EXPANDED并停止，不放宽更多规则。
3. 固定manifest、>=10子集和每类损耗后再下载；字节/来源检查不通过不得丢掉
   失败行后继续。原图不删除。数据、cache、特征及预测都留在E盘。
4. SHA256与64-bit dHash<=4筛查新队列内及全部历史（含旧external/G6X）照片。
   候选对未判读则停在DUPLICATE_REVIEW_REQUIRED；不能自动标成“不同照片”。
   原G6X仅有rights-holder而无可靠observer ID：只能做字节/近重复隔离，
   不能声称对G6X实现了observer-disjoint。已知source ID清单零交集必须通过。
5. 通过字节/来源/重复门槛后用冻结DINOv2特征与4模型×2训练政策×3seed，
   保持100-way输出，真正k=1..5全部子集；不训练、不调参、不重选checkpoint。
6. 复用20次同taxon/同cardinality跨observation的E3 label-informed诊断。
   此次不重做已取消的human-quality ratings，不把自动代理当人工真值。
7. 所有>=8taxa及>=10taxa子集各报每seed、3seed指标均值、逐taxon与fixed-n5；
   macro/micro/top5/NLL/Brier/ECE及5000次observer-cluster CI。
   空高预算记NOT_ESTIMABLE而非0；CI仍受小cluster与availability选择限制。
   在相同observation上检验1->2和2->3/4/5，9999 sign flips；对主/敏感性、
   4模型、2政策、4转换构成的固定64项家族做Holm；缺项用p=1保守占位。
   报>=10子集与主集的实际类数，不用更好的敏感性结果替换主结果。

## Files and completion

入口：`code/scripts/run_external_expansion_v2.py --host`（隐藏pythonw host）。
独立小产物：`code/results/external_expansion_v2/`。
大文件：`E:/Datasets/ROBird-Bench/external_expansion_v2/`。
所有page、taxon、download、feature、eval及statistics阶段均原子提交与hash恢复。
状态在queue_status.json / host_status.json；无需AI定时高频轮询。
完成称EXPANDED_EXTERNAL_VALIDATION_COMPLETE，不称原G6正式PASS。

## Manuscript erratum

旧v3/v3.1图将development的141 taxa/41 reserves/15,392 candidates混入了
external的100->13 census；这是错误链条。它们必须分开，新增队列不能沿用该图。
本轮在submission_for_rsos/notes记录hold notice，待新数据完成后统一重绘。
