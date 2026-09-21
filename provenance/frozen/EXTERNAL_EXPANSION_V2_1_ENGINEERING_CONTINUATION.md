# External expansion v2.1 — display-name normalization only

2026-09-18，v2首次启动停在metadata_census之前，错误信息为
`Expected original contiguous 100-taxon mapping`。诊断：development manifest
仍有100个taxon_id、100个class_index（0..99），但78类部分照片scientific_name为空。
v2误将(taxon_id,class_index,name)去重的178行当成了178类。这是工程读取错误，
不是科学门槛失败，亦没有新API响应/照片/特征/预测。

修正只把冻结development清单归一化为100行taxon_id/class_index/name展示表：
每个ID组合的非空名称必须唯一；任何ID冲突或多个不同非空名称仍拒绝。
原始照片清单和ID/class_index不修改、不换物种、不降低资格门槛。
独立v2.1数据/结果目录、config及freeze；继承v2全部阈值/种子/代码和旧证据hash。
v2 STOP原样保留，新prepare核验parent freeze并绑定归一化表及错误证据。
运行前对真实development清单验证100行，并新增缺名/冲突单元测试。

其余协议完全继承EXTERNAL_EXPANSION_V2_PROTOCOL.md。新版入口
`code/scripts/run_external_expansion_v2_1.py --host`，结果
`code/results/external_expansion_v2_1/`，大文件
`E:/Datasets/ROBird-Bench/external_expansion_v2_1/`。
不得再启动v2旧入口；原G6仍false。
