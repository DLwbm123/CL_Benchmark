# Class-CLIP-CIS seed3与NeST seed2结果

两项训练与独立test均返回0，每项三个任务checkpoint完整，current_task依次1..3、task_open=false，全部保存状态张量有限。冻结配置不变，未按测试成绩调参。

| 方法 | seed | 任务条件A-Dice | BWTR | whole-heart切片Dice | 原总历时/小时 |
|---|---|---|---|---|---|
| Class-CLIP-CIS | 3 | 0.8220712558 | −0.0132572253 | 0.6388995164 | 8.286 |
| NeST | 2 | 0.7811785565 | −0.0316116735 | 0.6921103045 | 8.667 |

两项均无旧attempt。任务条件矩阵与共同空间whole-heart的任务信息、平均方式不同，分别报告。背景包含、ASD沿用既定定义；无匹配参照RMA、未核对逐任务计数DRR留空。方法为框架适配，不称作者原生完整复现。CLIP编码语义顺序来自训练掩码/编码推断，原H5转换脚本缺失的限制不变。

Class-CLIP-CIS3/5、NeST2/5，均非完整五seed统计。全局28项审核完成、1项失败LGKD seed2、12项运行、29项排队，总70。失败及历史无效attempt另列，未因低分重跑或删seed。
