# S3R 源码优化器修正协议 v2

用户最新明确授权：有问题自行修改并继续实验。本协议只修复S3R标准配置和错误定位，不对其他方法调参。

旧S3R seed1在第三任务epoch63的概率BCE触发CUDA断言，已退出；T0/T1 checkpoint、原日志、成本和原时钟保留。最后完整checkpoint参数及importance有限，但importance最大值0，不能称旧运行已学到有效选择性正则。当前冻结SGD lr=.008与取回的作者源码默认设置不同，尚不能唯一确定失败原因；不将数值崩溃描述为加速代码修改了数据的证据。

依据已固定版本的working/references/S3R：train.py默认Adam、lr=5e-4、weight_decay=1e-4；agents/default.py构造ExponentialLR(gamma=.98)，agents/default_CISR.py逐任务重设lr并在每个epoch开始前step，默认不重置optimizer。因此本次采用Adam（默认betas/eps）、lr=.0005、wd=.0001；第e个零起始epoch实际lr=.0005*.98**(e+1)，每任务重置该lr序列但保留Adam矩。框架额外StepLR设gamma=1，避免双重衰减。

保留ResUNet32、256×256、batch8、每任务150 epochs、任务顺序、原train/val/test及S3R各损失系数/MC8/全量importance统计。当前task训练mask ACNN预训练50epochs的适配继续保留并计成本。本协议对齐源码优化器部分，不冒称完全复现作者骨干、200epoch/batch5或私有ACNN权重。

新增logits及softmax非有限值检查，BCE之前明确退出，不以clamp、nan_to_num或跳过batch掩盖发散。旧SGD checkpoint拒绝当Adam的完整续跑状态。五seed统一新协议；旧SGD失败不进入v2均值，仍报告全部旧成本。

验证包括三任务CPU更新/importance、准确epoch学习率、NaN logits提前拒绝、旧SGD checkpoint拒绝、运行身份/budget/train-test phase不能被协议文件覆盖。真实输入资格只使用BIDMC/HK/ISBI各一个训练batch8：各task50个ACNN更新、2个segmentation更新和1个MC8 importance批次，共24张训练输入、150个AE更新、6个分割更新，无held-out、不复用资格权重。这不是三任务完整训练或性能证明。

现有长驻调度器持有旧提交配置，新增protocols/s3r_s{seed}.json只允许覆盖明确列出的优化器相关字段，run.py启动时解析并记录protocol_id；不修改其他方法，不重启当前正常运行进程。正式S3R从原seed初始化；seed1失败目录原地归档，重新入队，沿用第一次started_at/deadline及旧失败成本，不开启新24h；其余四seed解除暂缓后正常入队，首次实际执行才起钟。每服务器新增总目录仍40GB预停，严格<50GB；同一方法/seed运行<=原首次24h，时间不足则按时结束，不拆名或延钟。
