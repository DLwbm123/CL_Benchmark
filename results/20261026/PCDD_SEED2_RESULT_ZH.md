# PCDD seed2结果

训练与独立test返回0，三个任务checkpoint完整、current_task依次1..3、task_open=false、全部保存状态张量有限，incremental cap1配置正确。

任务条件A-Dice0.4872604850、BWTR−0.2927425022；whole-heart共同空间切片Dice0.3637971991单独报告。效果弱且有遗忘，保留结果，不因成绩调参。原总历时13.818小时，无旧attempt，原预算不重置。

基础阶段不裁剪，增量累计56400步全部触发cap1，最大裁剪前norm40383.2383；这是显式paper-derived 2D ResUNet数值适配，非原论文原生参数或作者性能复现。未额外像素诊断，不声称全部样本全背景。背景包含、ASD沿既定定义，缺匹配RMA参照与DRR逐任务计数时均留空。

PCDD完成2/5seed，无完整五seed统计。全计划35项审核完成、2项失败LGKD seeds2/3、12项运行、21项排队，总70。旧失败/失效结果另列，不与有效结果混算。
