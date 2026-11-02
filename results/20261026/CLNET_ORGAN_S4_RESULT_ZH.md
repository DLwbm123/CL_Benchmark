# Continual Segment Organ-CL seed 4

2D task-aware 非原生适配，train/test 均返回 0，四个任务 checkpoint 顺序关闭且全部状态有限，每任务 15 次剪枝事件。任务 A-Dice 0.8109542777，BWTR 0，ASD 0.0112964855；原钟 7.450 小时。

BWTR 为零有共享编码器和旧 decoder 冻结的结构性原因，不单独证明分割性能。剪枝门控使用训练 Dice，未使用作者的验证集门控、3D 预训练和 body-part merging。同协议现 4/5，未形成完整五 seed 汇总；RMA 和 DRR 缺证据留空。
