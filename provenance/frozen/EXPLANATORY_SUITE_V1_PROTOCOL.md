# Explanatory suite v1 — 2026-09-19

## Status and question

用户已批准三组补实验并要求后台串行、不持续轮询。本版是看过expanded external v2.2
结果后的解释性分析，不是新的未触碰holdout，不预设正结果。主问题：自然同观察照片的
错误共现与难度构成，如何影响多图平均收益和局部回退？旧v2.2、STOP、原G6=false、
人工质量取消不变。只读3090照片/1121组/56类，100-way输出，零下载/训练/删除。

## Models and rationale

已有all-trained DINOv2 mean-feature、Probability MLP、Deep Sets、Set Transformer各
三seed(20260819/20/21)，及ResNet50 Probability MLP、Deep Sets同三seed；18个目标头。
分别区分线性logit平均、单图概率平均、非线性特征集聚合、交互式集合聚合；不是新算法。
出处沿用RSOS_SERIAL_SUITE_V1.md及稿件已核验参考文献，本版不声称文献新颖性。
独立ResNet50 mean-feature k1-trained三seed作为难度代理，不作为匹配效果目标模型。

## A. Error structure and difficulty-matched regrouping

代理difficulty=-log(max(三seed平均true-label singleton probability,1e-12))。
分箱边界仅取development validation照片难度的1/3、2/3分位(linear，照片等权)。
不以external分位数、目标模型对错模式或外部效果定边界。代理利用真标签，是
label-informed诊断，不是质量真值或可部署方法；与目标错误相关性仍可能存在。

按taxon、cardinality、排序3-bin难度签名分层，>=3个不同observation才纳入。
其他整组排除、逐组逐类记attrition，不放宽门槛救覆盖。无支持则该项NOT_ESTIMABLE，
其余已授权项完成。三条件完全同组支持/照片池：real、unrestricted_common（仅保持
taxon/cardinality）、difficulty_matched（再保持每组bin签名）。

每层随机排列observation；组内照片在各bin内随机排序。第j位置从循环偏移
1+j%(组数-1)的donor取图；每图一次，无本组donor，每伪组>=2 donor observations。
unrestricted用同一循环规则但不按签名分层，组内任意随机排列。seed=2026091900+r，
r=0..19；保留source observer映射，不强制跨observer，不把伪组slot当真实observer。
这不是从全部可能划分均匀抽样。逐repeat报告同observer来源比例、排序difficulty绝对差、
组平均difficulty绝对差、bin一致率；不筛repeat，粗分箱残余不平衡须保留。

主k2、次k3、k1照片池守恒负对照，k4/5描述性。枚举所有子集，先组内平均，再taxon
等权，再三seed平均：pair jointly wrong、pair same wrong label（无条件率）、
any-singleton-correct/set-wrong、all-singletons-wrong/set-correct，另报singleton期望
正确率、set正确率、NLL/Brier。错误共现是描述性事件率，不把边际错误率乘积当已验证独立零假设。
20重复范围不是CI；只报完整repeat分布/逐类/seed，不做伪组observer bootstrap或p值。
剩余差不能全部归因错误依赖；差消失也不证明机制不存在。

## B. Same weights, same acquired photos

18目标头同照片集，对比原生set概率与同checkpoint各singleton概率平均。
k1是单图正确率期望，不是单图概率集成。Probability MLP数值相等是实现负对照；
mean-feature是线性logit平均，其概率平均不同但不另加同义Mean-logit基线。
主要看Deep Sets/Transformer，不要求原生胜出。所有预算macro/micro、NLL/Brier/ECE、
native-minus-pool、nested regression/correction及错误结构全部保存，k2/3配对检验。

## C. Minimal second encoder replication

本地ResNet50 IMAGENET1K_V2权重，原232->224 transform、L2归一化，external只提一次。
6旧目标头零训练；报告同组1->2、2->3、harmful additions。旧E3规则（taxon/cardinality
>=3组）再20重复，和旧DINO E3对齐。A的common/matched也覆盖ResNet的6头。

## Statistics and integrity

真实组：三seed在同组差值上平均，固定物种macro，observer-cluster linearized 5000
bootstrap、9999双侧cluster signflip，seed20260919。observer独立和cluster符号
可交换是假定而非ID证明；报cluster数/最大权重/跨类observer，不删除离群值。
新增一个Holm16家族：6种encoder/model x native-minus-pool k2/k3=12项（含MLP
同义负对照），加ResNet2模型 x1->2/2->3=4项。点态CI非同时CI，不替换旧Holm128。
全部为post-hoc；重组只做随机化描述，不把20repeat/3seed当独立数据样本。
hash/科学完整性失败STOP；工程异常记日志，不篡改冻结规则，不按显著性停止。

## Implementation and execution

新module explanatory_suite_v1.py：difficulty/匹配/不变量/复用预测/错误分解/统计。
新runner run_explanatory_suite_v1.py：冻结/OS锁/原子marker/单worker隐藏host。
freeze绑定parent freeze 57e47894a193c7f9583660e9830ddb91a7d24746ae16a596885d1b9c6a2c25f2
及parent automatic_done 1a0230712c759987afd92dd09e00d101c3d7574a6474a53af308e922d55363c0，
绑定新code/config/tests、旧checkpoint/权重/开发特征/索引。原已完成文件只读。
顺序verify -> ResNet features -> proxy -> matching maps -> real/native+pool18 ->
common controls18x40 -> ResNet full E3 6x20 -> tables/tests -> final。
所有stage原子done支持恢复，final不覆盖；大文件E:/Datasets/ROBird-Bench/explanatory_suite_v1。
小规模GPU/映射/负对照测试通过后才冻结启动，不设置AI周期轮询。
ResNet缓存一致性使用原始32张batch；小batch可能触发cuDNN TF32数值路径变化，
不放宽2e-6校验容差而是复现原batch shape。正式extract同样batch32。
预计数十分钟至数小时，尚无本版实测耗时/显存，不声称已验证。
