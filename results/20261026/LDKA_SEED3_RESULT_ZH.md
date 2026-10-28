# Class-LDKA seed3结果

训练与独立test返回0，三个任务checkpoint完整，current_task依次1..3、task_open=false、所有保存状态张量有限，全任务cap1配置正确。

任务条件Test A-Dice0.4724978991、BWTR0.0128999176，whole-heart共同空间切片平均Dice0.5942786371单独报告。弱分割结果保留，正BWTR不能解释为高分割性能；没有新增像素诊断，不称全部测试样本全背景，不为低分调参。

累计84600步中72173步触发裁剪，最大裁剪前norm2809.0476；这是paper-derived 2D ResUNet显式数值适配，非原论文原生参数。原总历时12.973小时，无旧attempt，原时钟不重置。背景包含、ASD沿既定定义；无匹配RMA参照及未核对逐任务DRR计数留空。

Class-LDKA完成3/5seed，无完整五seed统计。全计划36项审核完成、2项失败LGKD seeds2/3、12项运行、20项排队，总70。历史失败和无效协议不混入有效均值。
