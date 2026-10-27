# S3R seed2：第二个已完成的负结果

六任务checkpoint完整、current_task1..6、task_open=false、张量有限，训练/test退出码0。原首次实际运行至退出3.685小时，没有旧中断或重试；沿用冻结source-optimizer-v2/150epochs/batch8及每task50epochs mask AE适配，原24小时约束不变。

Test宏平均Dice0.4957892590、precision0.9916671053、recall0.5000000002、ASD0.0166657895、BWT0，normalized RMA无匹配参照，留空。六阶段矩阵整行重复，指标与seed1完全一致，符合退化模式；不能将BWT0或一致性解释为防遗忘成功、模型稳健。宏指标含背景，沿用ASD空预测口径，不能据此证明有效前景分割。本seed没有额外逐像素检查，不能声称所有test像素均为背景。

S3R目前仅2/5seed完成，总队列9/70已审核完成。保留两个单seed负结果，不将其当作完整五seed统计、不据此调参或重跑；其他冻结seed继续。
