# PCDD 增量梯度保护适配 v2

旧 seed1 完成第一任务，在第二任务初始批次出现非有限损失。原 checkpoint 所有张量有限；CPU 同一真实 MMWHS task2 train batch8 的固定 checkpoint 对照中，无保护版本总损失 5.352→4.643e20，完成2次更新后下一步非有限；local/global 原型对齐项升至约7.738e20。该证据支持数值不稳定，但不唯一定位全部触发原因，CPU批次不等同原GPU随机批次。

仅增量任务增加 global gradient norm cap=1，基础任务关闭，原 SGD .01/momentum .9、各损失公式与权重、原BN冻结适配、150epochs/task、batch8、数据不变。明确标注 PCDD/ResUNet + incremental clip1，不称论文原生超参或完整作者复现。五seed统一 pcdd-incremental-gradient-guard-v2。受保护同批次4步总损失5.352/20.680/16.904/7.952均finite，裁剪前norm1085至3588；不证明长期稳定或提升准确率。

诊断6次CPU优化更新（无保护2、有保护4），另部署实现资格1次更新，单列工程成本，权重不复用正式训练。无测试集访问、无阈值搜索。checkpoint 保存 guard_stats 步数/裁剪次数/最大原始norm。

失败目录原地归档到 attempts/pcdd_unguarded_failure_v1/pcdd_s1；按原seed初始化重排，保留原 started_at/deadline，24小时不重置。旧T0不冒称无损恢复（缺RNG/采样器状态），丢弃训练更新与时间保留披露。其他运行不重启。已有全部任务格式/框架44项测试通过，新增长期状态保护测试单独验证。
