# Class-LDKA 与 LGKD seed5 结果

两项训练/测试均返回 0，3 个 checkpoint 顺序关闭、全部状态有限。Class-LDKA 全任务裁剪，84600 步中 71795 步裁剪，峰值 5021.989746；任务 A-Dice 0.4784756634，BWTR 0.0181592392，whole-heart 切片 Dice 0.6156380565。原钟 13.564 小时。

Class-LDKA 五 seed 均为全任务保护 v2，任务 A-Dice 均值 ± 样本标准差 0.4851059301 ± 0.0098760059，BWTR 0.0132767177 ± 0.0041037071。

LGKD seed5 base 未裁剪，增量 56400 步中 55977 步裁剪，峰值 108.574211；任务 A-Dice 0.7778670773，BWTR -0.0368390301，whole-heart 切片 Dice 0.7061825820。原钟 8.644 小时。seed1 无 guard 单列，seed2 到原 24 小时未完成，seed3/4/5 为增量保护 v2，不混算完整五 seed。

均为 ResUNet 保护适配，非原生参数；任务条件与 whole-heart 切片平均分开，保留背景和既有 ASD 口径。RMA 缺匹配参照、DRR 缺逐任务计数证据，均留空。LGKD 旧失败 batch 未复现，不能推断唯一根因或长期稳定；低分和接近零的 BWTR 不单独证明防遗忘。

Class-LDKA seed1 原钟包含旧 attempt，旧结果未混入；本次两项 seed5 均无旧 attempt。
