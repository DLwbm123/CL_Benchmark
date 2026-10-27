# Class-EWF seed3结果

训练与独立test均返回0，三个任务checkpoint完整、current_task依次1..3、task_open=false、所有保存状态张量有限。

任务条件Test A-Dice0.5695226827、BWTR0.0072829077，whole-heart共同空间切片平均Dice0.6189635546。二者任务信息/平均方式不同，分别报告。近零正BWTR不等于高分割性能；保持冻结配置，未为低分调参。

原总历时7.209小时，无旧attempt，原时钟不重置。背景包含，ASD沿用既定定义；无匹配RMA参照、未核对逐任务DRR计数均留空。Class-EWF完成3/5seed，无完整五seed统计。

全计划33项审核完成、2项失败LGKD seeds2/3、12项运行、23项排队，总70。旧无效实现与失败另列，不能混入有效均值或隐去失败seed。
