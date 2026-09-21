# Explanatory suite v1.1 — engineering continuation, 2026-09-20

用户明确授权修复后继续后台串行。v1在首个control之前STOP，真实18头、ResNet特征、
难度代理、40映射已完成；旧STOP/代码/config/freeze/结果均只读。

## Defect and correction

float32 difficulty在common CSV中以短十进制保存，而map通过to_dict变成Python float，
写出其完整float32值。读回float64严格相等误报。不是分箱或图片变化。
新版不使用allclose或放宽容差：两侧转回原float32后要求精确一致，并在全40映射上
核对原始proxy_probabilities.npz重新按旧算法计算出的difficulty。还要核对冻结validation
cutpoints、difficulty_bin以及旧validate_mapping的全部身份/照片池/标签/无自donor约束。
非有限值拒绝；hash改变拒绝。只在独立加载的旧module实例中替换validate_mapping
为明确的精度适配器，旧磁盘源码不改，不改变全局pandas或numpy行为。

## Resume design

新root code/results/explanatory_suite_v1_1及E:/Datasets/ROBird-Bench/explanatory_suite_v1_1。
冻结继承v1的1306输入expected hashes，并绑定旧STOP/host、已完成stage递归产物、
新修复代码/tests/本协议。不覆盖旧输出。已完成的real groups/nested/summary与40map
CSV逐字节复制到新root，生成新contract的wrapper marker，引用旧marker并标明reused；
特征数组只读原路径不复制/重提。没有新增下载/训练/样本筛选/分箱/seed/检验。
模型/统计执行复用旧冻结函数，只有read-back difficulty校验适配。
队列继续720 controls -> 120 ResNet E3 -> 旧Holm16统计/完整报告。

## Tests and gates

回归必须实际写/读CSV，重现旧失败再证新适配通过；真正相邻float32变化、NaN、
bin改变仍拒绝。真实preflight核验原始代理值、40maps、18real markers，冻结后每启动重验。
后台前额外执行一份真实完整control冒烟，结果属于正式首个control并由marker复用，
不是看效果选参数。失败记录诚实保存；科学规则绝不自调。

恢复入口docs/EXPLANATORY_SUITE_V1_1_HANDOFF_20260920.md。无AI高频轮询。
