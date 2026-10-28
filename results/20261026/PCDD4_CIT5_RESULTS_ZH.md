# PCDD seed4 和 Class-CIT 五 seed 结果

PCDD seed4 训练/测试均返回 0，3 个 checkpoint 按任务顺序关闭、全部状态有限；base 未裁剪，增量 56400 步中 56400 步裁剪，峰值 39970.109375。任务 A-Dice 0.4935690181，BWTR -0.2850644031，whole-heart 切片 Dice 0.3608150062 单列，原钟 11.353 小时。PCDD 4/5，非完整五 seed。

Class-CIT seed5 训练/测试均返回 0，3 个 checkpoint 顺序关闭、全部状态有限，AdamW；任务 A-Dice 0.7957460670，BWTR -0.0231831230，whole-heart 切片 Dice 0.8065236128，原钟 9.084 小时。五 seed 同 AdamW v2 协议，任务 A-Dice 均值 ± 样本标准差 0.8050409259 ± 0.0080863584，BWTR -0.0169453897 ± 0.0042168818。

均为二维框架适配，非作者原生复现；保留背景和原 ASD 口径，RMA 缺匹配参照、DRR 缺逐任务计数证据，均留空。任务条件宏平均与 whole-heart 切片平均分开，弱结果不据此调参。

Class-CIT seed1 原钟包含旧 attempt，旧配置结果未混入。两项本次 seed 无旧 attempt。
