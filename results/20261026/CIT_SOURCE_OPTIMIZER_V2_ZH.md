# CIT家族源码优化器修正 v2

Organ-CIT seed1在完成第一task后，第二task开头非有限loss退出，原checkpoint/日志/成本保留。审计发现当前CNN适配沿用SGD .008，与固定作者源码config/dv3+/r50-D8_ade20k_dv3_cl.py的AdamW base lr2e-5、head倍率10、betas(.9,.999)、wd.01不同。

用户已授权有问题自行修改继续。CIT/Class及Organ-CIT的五seed均统一AdamW base2e-5、head2e-4、wd.01，每task重新初始化Adam状态。为现有共享2D骨干做明确分组：ResUNet全部为base组，query/key/presence/LayerNorm为head组。这不等价于作者DeepLab/ASPP或视觉预训练。

保留batch8、150epochs/task及本benchmark的epoch80乘.5调度；作者的1500iter warmup/80000iter PolyLR不直接套入该epoch预算。本次对齐优化器组件，不称完整作者训练recipe；这一差异必须报告。loss、teacher bank、任务顺序及数据划分不变，不按held-out调参。

AdamW会衰减预分配但尚未到达的query，因此每步保存并恢复未来query行，且清空它们对应的Adam矩；单位测试验证其权重逐位不变、未来矩为零，head/base lr比例正确。该保护是保证未来参数不提前更新的必要适配。

相同Organ-CIT首task checkpoint和同一真实UCL train batch的有界诊断：旧SGD四步loss77.89/96.51/70.28/1.37e9；AdamW四步77.93/70.66/63.78/57.58均有限。dropout随机实现不是跨臂严格配对，不能据此给出性能因果结论；只做标准配置的局部可运行性检查，不用诊断权重正式训练。共8个额外optimizer更新，无held-out。正式仍需完整多任务验证。

旧Class-CIT seed1尚在训练，按用户纠正配置的授权中断并归档；Organ-CIT失败同样归档。两者原first-start/deadline均保留，不重置24h，按原seed从头初始化；未启动seed首次实际执行才起钟。十个CIT家族run均由protocols/cit*_sN.json显式解析优化器覆盖，预算/数据/身份/phase字段不可覆盖。其他正常方法不重启。仍GPU仅1/5、每卡6并发及新增总根40GB预停<50GB。
