# ReBac 增量梯度保护适配 v2

旧seed1完成T0/T1，在第三任务初始批次非有限退出。T1状态张量有限。CPU同checkpoint、task3首8个train切片对照：无保护4步3.545/6.497/91.565/186.810均finite但明显放大，clip1四步3.545/3.519/3.486/3.446有限。未复现实际GPU失败批次，不能把此对照宣称为唯一根因证明或长期修复证明。追加等间隔8个真实train切片（标签0/6/7）资格2更新4.069/4.051有限，2/2触发保护。合计10次CPU工程更新，不复用权重、不访问held-out。

仅增量阶段global grad norm cap=1，基础关闭。原SGD/Nesterov/momentum .9/wd1e-4、base lr.01/incremental.001/old.0001、全部损失权重及BN路径不改，150epochs/task、batch8不变。明确称ReBac/ResUNet+incremental clip1，是数值适配不是原作者参数。所有5seed统一rebac-incremental-gradient-guard-v2，protocol仅覆盖rebac_grad_clip，stage checkpoint保存guard_stats。非有限loss仍失败并记录各损失项，不做nan/clamp掩盖问题。

用户明确授权故障自行修复验证继续。seed1失败/T0/T1归档attempts/rebac_unguarded_v1/rebac_s1；seed2旧协议正在基础训练，为五seed一致性仅中断该项并归档相应目录，披露丢弃的部分基础任务更新和耗时，其他方法不重启。均按原seed重新初始化，非无损恢复，原started_at/deadline不变。其余尚未开始seed不发生丢弃。诊断未证明长期稳定，不追加clip阈值搜索或旧轨迹回放。

新增保护与相关32项CPU测试通过。协议解析另对五个旧提交配置验证，确保长驻调度器在启动时解析新白名单覆盖。
