# CIT Class-CL seed1结果（source optimizer v2）

三个完整任务checkpoint经核验current_task1..3、task_open=false、状态张量有限，保存配置均为AdamW。训练/独立test退出码0。原首次运行至退出12.666小时，包含代码切换与旧SGD中断成本，原时钟不重置。

Test任务条件宏Dice0.8028428299、Dice BWT -0.0132093769；共同标签空间whole-heart切片平均Dice0.7806497349。二者的任务信息与平均方式不同，需分别报告。背景包含在宏指标，ASD沿用原定义；无独立匹配参照，normalized RMA留空。

本协议为2D ResUNet适配，采用作者CNN配置对应AdamW base2e-5/head10x/wd.01，仍使用benchmark epoch80衰减/150epochs，并非作者完整iter warmup/PolyLR recipe。旧SGD轨迹另存，不混入此结果。此项为1/5seed，不支持五seed均值/标准差、显著性排名或原作者原生复现结论。

加上S3R seed2，目前共10/70项已审核；12项运行、48项排队。未因低分或测试结果调整任何比较方法配置。
