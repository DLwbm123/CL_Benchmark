# ReBac seed3结果与LGKD seed3失败

ReBac seed3训练与独立test均返回0，三个任务checkpoint完整、current_task依次1..3、task_open=false、所有保存状态张量有限。Test任务条件A-Dice0.7116318385、BWTR−0.0714270459，whole-heart切片Dice0.7123930259单独报告。增量cap1累计56400/56400步裁剪、最大裁剪前norm808.5871，基础关闭；原总历时10.318小时，无旧attempt。该显式数值适配非原论文原生超参，未因成绩调参。

LGKD seed3在完成T0/T1后第三任务启动出现非有限loss，训练返回1、没有test，原历时4.452小时。T1 checkpoint全部状态张量有限。这是继seed2后第二次相近任务边界失败，支持数值不稳定具有重复性，但不证明唯一代码根因。本轮不重复旧的无信息前向诊断、不进行额外优化更新或held-out访问，也不自动重跑/搜索参数。保留原日志、checkpoint、成本和deadline。配置差异及此前有限审计见LGKD_SEED2_FAILURE_ZH.md。

所有指标含背景、ASD沿既定定义；无匹配RMA参照、未核对逐任务DRR计数均留空。任务条件矩阵与whole-heart平均方式不同。ReBac完成3/5，LGKD只有seed1完整、seeds2/3失败，不能隐去失败或称五seed齐全。

全计划30项审核完成、2项失败、12项运行、26项排队，总70。其他任务继续按冻结协议运行，没有修改训练配置。
