# LGKD seed4 稳定性保护结果

训练与测试均返回0，3个checkpoint顺序、关闭状态、finite及增量cap1审核通过。任务条件 A-Dice 0.7793976537，BWTR -0.0223784857；whole-heart切片Dice 0.7187685785单列。原钟 9.133h，无旧attempt。增量裁剪统计 {"steps": 56400, "clipped": 56062, "peak_preclip_norm": 86.25395965576172}；base关闭裁剪。

这是明确的2D框架梯度保护适配，不是作者原生配置。seed1协议不同单列，seed2到限未完成，未形成可合并的完整五seed。背景和原ASD口径保留，RMA与DRR缺匹配证据留空。
