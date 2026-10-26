# Organ-LDKA 增量梯度保护适配 v2

用户已授权有问题自行修改并继续。旧seed1完成第一任务后，在第二任务开头出现非有限loss。输入UCL训练数组及首任务checkpoint浮点张量均有限，非数据NaN或checkpoint损坏。

用相同已完成任务checkpoint及相同UCL训练batch8复现：无保护时前两步loss为409.34、8.23e10，第三步非有限。临时改变辅助BN统计路径的假说对照也失败，已保留LDKA_ORGAN_BN_COMPARISON.json，正式代码不采用该变动。原loss、SGD lr=.001/增量.0001、momentum.9、wd1e-5、BN路径全部不改。

为跨organ的共享特征SKD适配增加增量阶段global gradient norm裁剪，阈值固定1，不搜索阈值或用held-out选参。基础任务阶段关闭。裁剪仅在原loss/各梯度均有限后执行，非有限仍退出，不跳batch或将NaN置零。相同checkpoint/batch的有界对照4步loss为409.34、404.96、397.06、386.71，裁剪前norm约3.70e4，均有限。该验证仅证明局部更新稳定，不能证明完整四任务训练稳定或性能改善。

该阈值是本框架Organ适配的显式数值保护，不是LDKA论文原生超参；对外须标注Organ-LDKA + incremental gradient clip=1，不能冒称原方法完全等价。Class-LDKA不改，其他方法不改。五个Organ-LDKA seed统一v2；不混入旧无保护失败结果。阶段checkpoint保存保护步数、实际裁剪次数及最大裁剪前norm，便于披露真实影响。

真实训练输入诊断无held-out：最初零优化更新的部件诊断1次，BN假说两臂共4个优化更新（均失败），梯度保护两臂共6个优化更新（旧2步后失败，新4步有限），合计10个额外优化更新，单列工程成本；资格权重不用于正式实验。首任务checkpoint与全部原失败日志保留。

正式seed1按原seed初始化重启，归档旧失败目录及checkpoint，保留最早started_at=1793026850.9914997、deadline=1793113250.9914997，不重置24h。其余未启动seed首次实际执行才起钟。仍150epochs/task、batch8、原任务顺序及train/val/test、GPU仅1/5每卡6并发，总新增物理目录40GB预停<50GB。其他正常运行不重启。protocols/ldka_organ_s1..s5.json只允许覆盖ldka_organ_grad_clip，不允许改lr、seed、数据或预算。
