# Class-CIT seed3结果

训练和独立test均返回0，三个任务checkpoint完整、current_task依次1..3、task_open=false、全部保存状态张量有限，AdamW配置正确。

任务条件Test A-Dice0.8177069412、BWTR−0.0176331361，whole-heart共同空间切片平均Dice0.8036271863。任务信息及平均方式不同，分别报告。背景包含、ASD沿用既定定义；无匹配参照RMA及未核对逐任务DRR计数留空。

采用冻结source-optimizer-v2配置，保留benchmark epoch schedule，非作者完整warmup/PolyLR recipe。未根据测试成绩调参。原总历时10.126小时，无旧attempt，原时钟不重置。

Class-CIT已完成3/5seed，无完整五seed统计。全计划31项审核完成、2项失败LGKD seeds2/3、12项运行、25项排队，总70。失败与旧失效实现另列，不混入当前结果。
