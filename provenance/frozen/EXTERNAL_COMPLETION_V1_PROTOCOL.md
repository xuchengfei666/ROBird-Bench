# External completion v1 — statistical reporting and human-quality boundary

## Scope and retrospective status (2026-09-09)

用户要求先补齐缺项，再分析研究结论。v1.5自动队列已于19:41:16正常完成。
本补充只读其完整与整簇排除两套已保存预测，不下载、不推理、不训练、不改标签、
不改旧代码/结果/freeze。旧48-cell development模型矩阵不扩大。已见过若干结果，
因此本文件是retrospective reporting补充而非未见数据的预注册。没有新成功门槛。
所有结果均输出，不由p值决定保留与否；此轮不撰写性能叙事。

## 1 Complete metric intervals

两个cohort modes，4 models，2 training policies，3 seeds各自报告，以及固定3seed
指标均值报告（不是概率ensemble，不把seed当独立观察）。k1--5在eligible cohort
及固定n=5 cohort上报告macro Top1、micro Top1、Top5、NLL、Brier、15-bin ECE，
并在eligible cohort逐物种提供相同指标。每个原始预测先通过100-way/全部子集
身份检验；点估计对照冻结metrics/per-taxon结果，误差超1e-10则停止。

Bootstrap固定5000次、seed20260909，独立单位为observer，不能把照片/子集当样本。
Macro Top1沿用固定物种的linearized observer bootstrap，不替换旧区间；其他
group-equal指标使用observer multiplicities重新计算比率。ECE每次重新汇总15个
固定bin的correct-confidence差后取绝对值，不能平均单图ECE；3seed ECE先分别
计算，再平均指标。保留跨taxon同observer关联。n_clusters<2只报告点估计，CI为
NOT_ESTIMABLE，不制造零宽“确定性”。零宽经验区间另标finite-sample degenerate，
不证明总体无不确定性。区间为逐项pointwise、条件于现有可用性筛选cohort，不能
解释为同时覆盖或对所有鸟类的无偏推断。既有taxon bootstrap仍独立保留。

## 2 Paired budget tests

使用相同observation交集比较(1,2),(2,3),(2,4),(2,5)，3seed逐组均值、政策分开。
统计量为taxon-macro accuracy difference。每个observer贡献为其组内
sum(delta_i/(T*n_taxon))；同observer下所有taxon/observation共用一个sign。
对observer贡献作双侧sign-flip：<=13clusters穷举，否则9999次Monte Carlo，
以绝对统计量比较且随机p=(1+extreme)/(1+9999)，记录MC标准误。
另报既有taxon-signflip作为假设不同的敏感性，不把两个p取较小值。
这要求独立observer clusters及零假设下cluster符号可交换/对称；是明确假设，
不是数据检查能证明，也不是随机采集/干预证据。预算差值是子集平均后的连续量，
不能硬做把子集视作独立二元样本的McNemar。固定cohort的配对bootstrap区间与
上述检验的假设不同，不承诺它们严格互为反演。

Full/all 16个预算检验为主reporting family；full/k1、exclusion/all、exclusion/k1
各16为单独敏感性family，组内Holm。同时额外报告所有64项的Holm以供联合读取。
Taxon p按相同family另行校正。效应保留原始比例和百分点；诊断记录group/taxon/
observer数量、observer跨物种情况、最大cluster权重及符号交换性未验证标识。
不按Shapiro结果更换检验、不删离群、不用事后power解释未显著。

尺寸分层额外明确输出group-mean pixel_area与short_edge_px的全量失败分层，
分位数分箱只作描述，不作质量真值/模型选择或因果阈值。

## 3 Human annotation package — cannot be completed by fabricated raters

固定969photo IDs（967字节身份），与冻结A/B模板同覆盖，随机顺序按A/B独立固定
seed。自动生成只在127.0.0.1提供服务的本地盲评页：显示原图及可切换检测框，
隐藏taxon/observer/model预测/自动代理值；记录尺寸为显示技术信息。两名真实
独立标注者分别填写，不能互看答案，不自动预填代理或AI结果。

字段：bbox_valid yes/no/uncertain/not_detected；motion_blur_present yes/no/uncertain
（仅可见方向拖影外观，不是物理因果鉴定）；occlusion_fraction 0--1或uncertain；
background_complexity_1_5；confidence_1_5；target_ambiguous yes/no/uncertain；notes。
无检测、多鸟、裁切、主体太小或无法判断必须允许uncertain，不强制猜测。
遮挡为视觉估计、不是隐藏身体的可验证真值；人工框有效性不是自动重训练框。
所有人工记录需要本人rater_id、human来源声明与独立性确认，服务器按版本保留
每次提交；原照片不复制不删除。完成不以不确定项数量作为排除/重标门槛。

后续导入两份真实人类export，检验photo/hash覆盖、合法值与不同rater身份，再
输出一致率/分类kappa、遮挡分歧、uncertain coverage、关系映射及各rater独立
描述性分层和失败分析；不静默达成共识。真实人工输入不存在时只输出
PENDING_TWO_INDEPENDENT_HUMAN_RATERS，不能以“网页/模板已做”冒称D已完成。
motion/occlusion人类评级不能升级成光学机制真值。原G6保持false。

## 4 Verification and provenance

新版独立freeze绑定protocol/config/code/tests、父v1.5 freeze及automatic_done和
全部使用的预测/metadata/quality产物。CPU串行、checkpoint按run保存；停止/完成
有原子状态和hash，旧内容只读。133既有测试继续，新增点估计/权重/稀有物种/
ECE非线性/配对sign/多重校正/人工来源拒绝/HTTP边界测试。最终汇总分开列明
statistics_complete与human_quality_complete。暂不撰写性能结论。

## Source-verification boundary

本轮尝试获取SciPy配对置换检验官方说明，但未返回可用网页正文，不宣称外部
文献已核验。上述公式和假设自包含，并通过独立穷举/直接重采样测试核对。
ECE严格沿用本地冻结budget_metrics_v2_1.py的15-bin group-equal定义，无拟合。
