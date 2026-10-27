# Benchmark 报告口径与方法处理约定（用户2026-10-27确认）

目标是公平benchmarking既定方法，不是把单个比较方法调到更高分。方法表现一般、遗忘较大，只要实现/标准配置/评测正确就按冻结协议完成并如实报告。明显退化（全背景、NaN、监督梯度消失等）触发正确性审计，但不能单凭低分认定实现错误；只有查到明确实现、源配置或评测问题才修复。不得按事后分数挑seed、删低分、以救分为由反复改学习率、loss权重、clip或结构。

论文以manuscript/main.tex的指标定义为准，本次不修改论文正文/公式、不改变训练数据或预算。

| 论文指标 | 现有输出/计算 | 报告约定 |
|---|---|---|
| A-Dice | mean_dice = 最终任务矩阵最后一行平均 | 保留论文定义，不用全矩阵均值替代 |
| BWTR | backward_transfer = mean((d_Ti-d_ii)/d_ii), i<T | 代码已是BWTR；旧BWT简称改为BWTR，数值不变；原CSV字段名保留兼容 |
| RMA | restricted_model_ability（新方法需匹配独立任务参照） | 无匹配参照留空；未归一化对角线均值不能冒称RMA |
| E-FWT | forward_transfer对全部阶段/未来域配对求平均并减初始化参照 | 仅Domain-CL主表使用，不把Class/Organ的旧FWT字段当作论文E-FWT |
| WCD | Class-CL whole-heart共同标签空间最终Dice | 与任务条件矩阵A-Dice分开，披露当前患者/切片平均差异 |
| MPE | model_size_efficiency，逐阶段参数增量相对初始参数平均 | 参数计数口径与论文一致，辅助教师/shape模块成本另报 |
| DRR | 需按论文逐任务N_replay/N_train记账 | 旧samples_storage_size_efficiency是另一种总数据分母的互补量，不能直接改名DRR，也不能简单1减后声称满足论文；缺分任务证据先留空/注明待核对 |

BWTR要求分母d_ii为正；遇零或缺失应明确标不可计算，不临时加epsilon改定义。已有已报告BWT数值来自比例公式，现统一更正名称为BWTR；这不改变任何测值，不需重训。主结果继续逐seed及失败分列，五seed完整后给均值和样本标准差；部分结果注明实际n。

旧TED/S3R数值失效的更正仍有效：属于明确梯度/源配置错误修复，不是低分调优。Organ-LDKA当前低分和全程clip说明适配效果差，但未定位明确新错误前不继续调参或盲目重跑。其他比较方法同理。原70id、seed1..5、既定资源/原24h时钟和旧失败证据全部保留。
