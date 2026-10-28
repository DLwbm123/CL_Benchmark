# EWF Class 与 Organ 五 seed 结果

两项 seed5 训练/测试均返回 0，分别有 3 / 4 个 checkpoint 按任务顺序关闭、全部状态有限。Class seed5 任务 A-Dice 0.5719459234、BWTR -0.0024960724，whole-heart 切片 Dice 0.6228759894 单列；Organ seed5 任务 A-Dice 0.5368956231、BWTR -0.2563253839。原钟分别 6.096 / 6.120 小时。

五 seed 均使用 perf-v1，任务均值 ± 样本标准差：Class A-Dice 0.5463795802 ± 0.0251187071，BWTR -0.0001053429 ± 0.0107919327；Organ A-Dice 0.5927305800 ± 0.0401581466，BWTR -0.1876337630 ± 0.0598345846。

二维框架适配非原生复现，任务条件宏平均和 whole-heart 切片平均分开；保留背景和既有 ASD 口径。RMA 缺匹配参照、DRR 缺逐任务计数证据，均留空。正 BWTR 不单独证明分割性能；无像素诊断不推断全背景。

两组合的 seed1 均曾按授权从初始 seed 重启，原钟包含旧 attempt，旧结果未混入；两项 seed5 无旧 attempt。
