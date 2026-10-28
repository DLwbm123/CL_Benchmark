# Class-LDKA seed4 与 LGKD seed2 状态

Class-LDKA seed4 训练测试均返回0，3个checkpoint顺序、关闭状态和有限性通过。任务条件 A-Dice 0.4917838017，BWTR 0.0072539951；whole-heart切片Dice 0.5754413944；原钟 11.995 小时。梯度裁剪统计 {"steps": 84600, "clipped": 71861, "peak_preclip_norm": 3310.694580078125}。该组合4/5，正BWTR不单独证明分割性能。

LGKD seed2重跑到第三任务epoch110时达到原24小时硬限预留停止，train返回124，未完成训练及测试，保持失败记录；先前非有限失败及本次日志checkpoint均保留。不可据此给出完整指标，不能延长时钟。seed3重跑另行跟进。

均为2D适配，非作者原生复现；背景和原ASD口径保留，缺参照RMA和缺逐任务计数DRR留空。
