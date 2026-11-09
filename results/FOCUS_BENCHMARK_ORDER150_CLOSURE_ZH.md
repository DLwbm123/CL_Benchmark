# FOCUS 原域顺序/batch8核对：53.82%，停止后续调参

已完成两个seed、六域各150epoch的完整核对，退出码0、轮数与checkpoint有限性检查通过。恢复原benchmark域顺序和名义训练/回放batch8后，最终平均foreground Dice为53.82%，两个seed的六域均值样本SD为32.30个百分点，说明该配置下跨seed稳定性明显不足。

| 配置 | 最终平均Dice | seed均值样本SD | 刚学完均值 | 最终减刚学完 | 零值组合 |
|---|---:|---:|---:|---:|---:|
| source/shared顺序，batch4，Adam150 | 73.59% | 1.38 pp | 80.44% | −6.85 pp | 0/12 |
| 原benchmark顺序，名义batch8，Adam150 | **53.82%** | **32.30 pp** | 64.79% | −10.98 pp | 0/12 |

指标为逐域foreground micro Dice后对域和seed等权平均；两个seed的SD不是置信区间。没有挑选更好的seed。无零值组合不代表稳定或效果可靠。

## 如何解释

两组同时改变了域顺序和训练/回放batch，因此不能把均值下降19.77个百分点或seed差异归因于单个因素。首任务改变还会影响随后冻结的encoder及整个回放过程；这是机制层面的可能影响，尚无独立因果验证。名义batch由4变8也改变每epoch优化步骤数，而作者式域均衡回放在旧域数不能整除8时实际批量可小于8；分配规则保持未改。

用户认可的73.59%结果保留，严格对应其source/shared域顺序和batch4协议，不能替代本次53.82%的核对结果。该核对限制了将现有FOCUS适配直接移入原主表的依据；不能仅展示更高配置或更好seed来宣称原benchmark可复现。之前优化器适配和50→150训练计划扩展的同协议改善证据仍然保留，但不构成跨设置稳健性的保证。

## 核验与收尾

两seed41/43均完整6×150epoch；model、optimizer和五旧域回放库张量有限，每个回放域至少30项。实际新样本步骤49,350、回放36,150、GIN24,749。历时约54.08分钟，本实验实际占用约0.335GB，FOCUS窗口累计约1.407GB；worker和guard进程组均退出。

完整方法仍为ResUNet32-mid、Adam .0002/weight-decay3e-5/poly .9、CCD .9、GIN .5、首层特征回放、域均衡前景采样、首任务后冻结encoder含bridge/BN。原benchmark顺序为ISBI→ISBI_1.5→I2CVB→UCL→BIDMC→HK；旧source/shared顺序为BIDMC→HK→ISBI→UCL→ISBI_1.5→I2CVB。不能按相同数字任务索引直接逐域配对这两种顺序。

按用户已确认的意愿，FOCUS到此收尾，不再自动调参、加轮数、扩seed或新开训练。既有报告、负结果和私有checkpoint保留。本会话监测继续跟进另行授权的TED/CauAug，不代表恢复FOCUS实验。

本次仍用第一val元数据块选择earliest-best，其余val块开发评估，test未使用。病例/域身份与预处理来源、metadata块独立性尚未充分证实，validation有历史曝光，label沿用整数截断。恢复训练顺序和batch并不等于完成原benchmark最终test评估。公开只含自写代码、协议、核验与跨域跨seed聚合，患者、逐域/逐块结果、权重/特征不公开。
