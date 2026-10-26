# TED seed1：已完成的负结果

训练与独立test均正常退出，六阶段checkpoint完整（current_task1..6、task_open=false、所有直接状态张量有限）。原首次实际执行至test退出约7.653小时，含代码替换前的中断成本；原时限未重置。协议仍为原冻结TED参数及perf-v1工程加速，未因低分重跑、改变学习率或选择其他checkpoint。

Test宏平均Dice0.4957892590、precision0.9916671053、recall0.5000000002，均含背景；沿用ASD0.0166657895，空前景/背景口径须谨慎解释。六阶段矩阵整行重复，最终代表性train切片全背景，支持退化判断，不能证明每个test像素均为背景。BWT0不能解读为成功防遗忘；无匹配参照，normalized RMA留空。

这是TED的1/5seed，S3R也仅1/5seed完成，目前共2/70项已审核完成，不报告五seed均值与标准差。该负结果与S3R相同宏指标，不构成两个方法等价的证据。其余冻结队列继续；适配PCDD已真实启动且Resolved protocol正确。
