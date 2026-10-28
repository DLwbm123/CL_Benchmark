# TED seed3 v3 与 NeST seed3 结果

两项 train/test 均返回 0；所有任务 checkpoint 顺序、关闭状态及有限性已审核。以下为任务条件宏平均；NeST whole-heart 为独立切片平均。

- ted_s3: A-Dice 0.5313267720; BWTR -0.3117766169; 原钟 5.324 h。
  E-FWT 0.2747006908。
- nest_s3: A-Dice 0.8023796907; BWTR -0.0161311613; 原钟 7.834 h。
  whole-heart 切片 Dice 0.6979275494。

各方法仅3/5有效 seed，尚非完整五 seed。显式2D适配，非原作者原生复现；背景和原 ASD 口径保留。无匹配独立参照的 RMA 及未核实逐任务计数的 DRR 留空。TED logits 稳定实现的极端分支与作者 clamp 不保证位级一致。
