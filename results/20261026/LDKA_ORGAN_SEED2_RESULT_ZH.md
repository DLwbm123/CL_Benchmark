# Organ-LDKA seed2结果

训练与独立test返回0，四任务checkpoint完整、current_task依次1..4、task_open=false、全部保存状态张量有限；配置核验增量global grad cap1，基础阶段关闭。

Test任务条件A-Dice0.4917071071、BWTR0.0000008172、宏recall0.4999979522，阶段矩阵几乎不变。这是弱适配结果，不能以近零BWTR解释为成功防遗忘。未新增像素诊断，不称全部测试样本全背景。

增量66750步全部触发裁剪，最大裁剪前norm38157408；基础0步裁剪。此为明确披露的paper-derived 2D task-aware Organ数值适配，非作者原生超参/性能，不搜索裁剪阈值或根据分数修改损失。原总历时11.488小时，无旧attempt，原预算不重置。

背景包含、ASD沿用既定定义，RMA无匹配参照及DRR逐任务计数未核对均留空。该组合仅2/5seed，无完整五seed统计。全局26项审核完成、1项失败、12项运行、31项排队；LGKD seed2失败继续单列，不自动重跑。
