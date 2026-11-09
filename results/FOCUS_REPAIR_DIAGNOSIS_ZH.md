# FOCUS 适配改进：第一项诊断与后继验证

此前统一ResUNet32的完整双seed六域结果为12.44%，而作者网络适配的20/150epoch结果为67.11%/76.78%。这不是作者原生复现对比，也不能据此认为论文FOCUS本身无效。本轮优先检查此前提出的原因，保留所有负结果，不以test选择方案。

## 固定checkpoint否定“只改BN推理即可恢复”

复用旧两个最终checkpoint，权重不变，不选择新checkpoint；在同样六域remaining-validation块上比较：

| 推理条件 | 跨域跨seed平均foreground Dice |
|---|---:|
| 原running BN、batch4 | 12.4442% |
| running BN、batch1 | 12.4442% |
| 仅BN使用当前slice统计、batch1，CCD保持关闭 | 6.8280% |

两臂均退出码0，原running-b4逐域数值成功重现，全部model/buffer状态在诊断后恢复。说明CauAug曾出现的“切换BN就恢复”不适用于当前FOCUS最终checkpoint；不能据此排除所有BN与训练机制的相互作用，但不再把单纯推理模式修正当作已证实方案。

额外的train-only粗粒度统计显示，所有训练slice的均值平均约−0.000135，标准差平均约0.946，整体已经接近中心化与单位尺度。因此“原始强度完全未归一化”的猜测没有得到支持；仍存在长尾值，且原始病例级预处理来源未核实。首次入口遗漏数据子目录而失败，未读取到数据、未训练，失败记录保留；修复环境路径后沿用原钟。

## 已冻结的训练稳定性诊断

选现有stream首尾两个域，分别从头独立训练；每域seed41/43、50epoch、batch4，共16臂。相同原ResUNet32、相同数据及first-val-block earliest-best规则。四个条件：

1. 完整FOCUS首任务训练：CCD .9、GIN .5，原SGD .01/momentum .99/Nesterov/weight-decay3e-5/poly .9。
2. 仅旁路CCD输出，仍生成同样mask以保留随机数消耗。
3. 仅旁路GIN输出，仍计算同样变换以保留随机数消耗。
4. 保留CCD/GIN，使用Adam2e-4、相同weight-decay和poly计划。该数值参考项目现有ResUNet医学训练，是明确的优化器适配，不是FOCUS官方设置，不开展学习率网格搜索。

记录损失、裁剪前梯度范数、被裁剪步数比例及logit尺度，以区分训练数值不稳定和单纯评估失配。完整16臂结束后才跨两个域、两个seed聚合判断。去组件条件仅用于诊断；首任务没有回放或跨域冻结，因此结果不能直接证明六域持续学习有效。后继主线应保留完整FOCUS机制，并独立验证序列学习。

## 额外可比性修正说明

代码审计发现，之前source/shared运行入口的域顺序为BIDMC→HK→ISBI→UCL→ISBI_1.5→I2CVB，原benchmark loader的任务顺序则为ISBI→ISBI_1.5→I2CVB→UCL→BIDMC→HK。这是此前报告未充分披露的差异，必须补充。它不能解释source网络到shared网络的落差，因为两者使用同一前述顺序；但进一步限制与原主表直接比较。既有结果不改写，本轮诊断继续固定原source/shared顺序以避免混入新的变量；若后续恢复原benchmark顺序，必须另立协议。

新授权窗口为2026-11-09北京时间14:28:59–20:28:59，累计新增存储严格小于10GB，8GB提前停止。当前训练诊断仍在运行，不报告效果改善已被证实。所有数据和患者派生checkpoint保留私有，公开仅自写代码、协议和跨域跨seed聚合；validation历史曝光、病例边界未证实等限制继续适用。
