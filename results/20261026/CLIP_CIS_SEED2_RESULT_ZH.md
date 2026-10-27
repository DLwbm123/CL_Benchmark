# Class-CLIP-CIS seed2结果

训练/test均返回0，3个任务checkpoint核验current_task1..3、task_open=false、张量有限。首次实际执行至测试退出7.167小时，无旧attempt，原24小时时钟不变。

| Seed | test任务条件宏Dice | Dice BWTR | whole-heart切片平均Dice |
|---|---:|---:|---:|
| 1（既有） | 0.819436669 | -0.028314038 | 0.626722479 |
| 2（新增） | 0.834039266 | -0.007170173 | 0.616317092 |

仅2/5seed，不能声称完整五seed结论或显著排名。任务条件矩阵有已知task屏蔽且采用患者平均；whole-heart为共同标签空间、切片平均，必须分开报告。背景包含、原ASD和CLIP标签语义推断限制保留，normalized RMA无匹配独立参照留空。不能将0.834任务条件Dice写成whole-heart性能。

当前10项已审核、12项运行、48项排队，共70个id。旧TED/S3R三项数值失效另列，不计入当前可比较完成结果。TED v3已实际启动并核对Resolved protocol；S3R v3仍排队，原钟继续计时。此报告仅新增结果摘要，不改变任何方法参数。
