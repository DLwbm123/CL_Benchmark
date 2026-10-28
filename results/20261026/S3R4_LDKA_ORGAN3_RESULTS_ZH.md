# S3R seed4 v3 与 Organ-LDKA seed3 结果

两项 train/test 均返回 0，全部任务 checkpoint 关闭、顺序正确且状态有限。任务条件宏平均如下：

- s3r_s4：A-Dice 0.6451790817，BWTR -0.1256786765，原钟 3.663 小时。
  E-FWT 0.1047605805；6 个任务 importance 均非零。
- ldka_organ_s3：A-Dice 0.4917081297，BWTR 0.0000000000，原钟 10.768 小时。
  增量 66750/66750 次梯度裁剪，峰值范数 180454.03125；基础任务关闭裁剪。

S3R 4/5、Organ-LDKA 3/5；均非完整五 seed。Organ-LDKA 为显式 2D task-aware 适配，弱结果和近零 BWTR 不证明成功防遗忘。无新像素诊断，不推断全部背景。背景及原 ASD 口径保留；无匹配 RMA 参照或逐任务 DRR 证据时留空。
