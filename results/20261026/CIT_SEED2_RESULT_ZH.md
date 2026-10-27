# CIT Class-CL seed2结果（source optimizer v2）

训练与独立test退出码均为0，三个任务checkpoint的current_task依次1..3、task_open=false，全部保存状态张量有限，AdamW配置正确。原运行总历时10.101小时，无旧attempt。

Test任务条件A-Dice为0.8066008770，BWTR为−0.0178707669；共同标签空间whole-heart切片平均Dice为0.8040923066。两种指标使用的任务信息和平均方式不同，分别报告。背景包含，ASD沿用既定定义；无匹配独立参照，RMA留空，DRR逐任务计数未核对，暂不报告。

采用冻结的2D ResUNet适配：AdamW base2e-5/head10x/wd.01，保留benchmark epoch80衰减与150epochs，并非原作者完整warmup/PolyLR训练recipe。没有因分数调参或挑选checkpoint。

CIT目前完成2/5个seed，不能称五seed完整统计。全计划共12/70项已审核，12项运行、46项排队；旧TED/S3R无效结果另列，不计入有效完成。
