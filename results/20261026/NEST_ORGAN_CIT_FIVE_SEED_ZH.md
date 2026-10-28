# NeST 与 Organ-CIT 五 seed 结果

两项 seed5 训练/测试均返回 0，分别 3 / 4 个 checkpoint 按任务顺序关闭、全部状态有限，Organ-CIT 使用 AdamW。NeST seed5 任务 A-Dice 0.7942164410、BWTR -0.0240930631，whole-heart 切片 Dice 0.6973723403 单列；Organ-CIT seed5 任务 A-Dice 0.6761317283、BWTR -0.2699116569。原钟分别 8.154 / 11.341 小时，旧 attempt 分别 0 / 0 项。

五 seed 各使用同协议，任务均值 ± 样本标准差：NeST A-Dice 0.7935700937 ± 0.0076811734，BWTR -0.0204125390 ± 0.0074959415；Organ-CIT A-Dice 0.7069962941 ± 0.0181294776，BWTR -0.2128132707 ± 0.0332174433。

二维框架适配非作者原生复现，Organ-CIT 未采用作者完整 warmup/PolyLR；任务条件宏平均与 whole-heart 切片平均分开，保留背景和既有 ASD 口径。RMA 缺匹配参照、DRR 缺逐任务计数证据，均留空。

两组合 seed1 原钟包含旧 attempt，旧结果未混入；Organ-CIT 的旧 SGD 配置未混入 AdamW v2。
