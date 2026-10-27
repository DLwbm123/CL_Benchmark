# LGKD seed2数值失败记录

LGKD seed2完成前两个任务，在第三任务epoch0记录首个loss2.80379629后，下一次observe触发非有限loss检查，训练返回1，未进行held-out test。该项保留为失败，不能计为完成，也不使用前两个任务代替完整结果。

低成本审计确认T1 checkpoint全部状态张量有限。CPU从该checkpoint开启第三任务，对首8个train切片执行一次前向（输入有限、标签均背景0），supervised1.03634、KD5.60576，学生与教师logits均有限；额外优化更新0、无held-out访问。该批次不是GPU实际失败的打乱批次，不能声称复现了失败或确定唯一根因。LGKD损失使用log-softmax，尚未发现S3R/TED那类概率BCE梯度锁死错误。

配置沿用此前声明的benchmark plain SGD .008及epoch StepLR，与作者SGD momentum/Nesterov/weight decay/PolyLR存在差异；此差异应披露，但当前证据不证明其导致本次seed特定失败。不以无依据改学习率或增加裁剪来追分，不自动重跑。原日志、checkpoint、退出码、耗时与deadline保留；如后续确认具体实现错误，须先验证再统一版本，原预算不重置。

其他任务继续由既有调度执行。目前23/70项审核完成、1项失败、12项运行、34项排队。LGKD仅seed1完整；不得隐去seed2失败或称五seed齐全。
