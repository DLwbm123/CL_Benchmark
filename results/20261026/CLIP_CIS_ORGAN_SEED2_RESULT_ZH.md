# Organ-CLIP-CIS seed2结果

训练/test退出码均0，四任务checkpoint的current_task1..4、task_open=false、直接状态张量有限。原首次实际执行至退出7.277小时，无旧attempt，原时限不变。

| Seed | test任务条件宏Dice | Dice BWT |
|---|---:|---:|
| 1（既有） | 0.636515861 | -0.349957491 |
| 2（新增） | 0.709224106 | -0.258503259 |

两seed均有明显遗忘；仅2/5完成，不是完整五seed统计，不能据此做显著性排名。此为显式已知task路由的Organ适配，背景包含、患者平均矩阵、原ASD口径沿用；无whole-heart任务指标，normalized RMA无匹配独立参照留空。不能把新任务高分当作旧任务保留成功，不为低分调整配置。

当前11项已审核、12项运行、47项排队，总70id；旧TED/S3R三项数值失效另列。S3R seed1修复版已启动且解析s3r-source-logits-v3，TED修复版也在运行，两者仍使用原首次时钟，完整效果尚未验证。
