# EWF 两个setting的seed1结果

Class-CL与Organ-CL seed1的训练/独立test均返回0，分别3/4个完整任务checkpoint，current_task依次到3/4、task_open=false，张量有限。原首次执行至退出9.663/9.753小时，包含旧中断成本，原时限不变。

| Setting | test任务条件宏Dice | Dice BWT | whole-heart切片平均Dice |
|---|---:|---:|---:|
| Class-CL | 0.512717750 | -0.017070850 | 0.598457362 |
| Organ-CL | 0.631102637 | -0.143234495 | 不适用 |

均为单seed，背景包含在指标中，不能与其他方法的五seed结论或原论文数值直接比较。任务条件矩阵有任务信息，whole-heart是共同标签空间、不同平均方式，不能混为同一个Dice；沿用原ASD空前景/背景口径。normalized RMA无独立匹配参照，留空。

Organ结果表现出遗忘；Class的任务条件指标与whole-heart出现差异，需分开如实报告，不能据此救分调参或重跑。当前累计8/70项已审核完成，各方法仍仅seed1，不报告五seed均值/标准差或显著性排名。其余冻结队列继续。
