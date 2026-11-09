# FOCUS完整六域50epoch验证：优化器适配后的结果

后续150epoch完整结果见 [150epoch报告](FOCUS_ADAM150_SEQUENTIAL_ZH.md)；本文保留50epoch发布时记录。

新双seed六域实验已完成并核验。保留完整CCD/GIN、特征回放和encoder冻结机制，采用Adam .0002后，最终跨六域、两seed平均foreground Dice为59.14%，相比已保存的同协议SGD .01版本12.44%提高46.70个百分点。两个seed的均值和全部12个seed×域配对均改善；没有选择部分seed或替换旧结果。

| 聚合指标 | 原共享骨干SGD50 | 完整机制Adam50 |
|---|---:|---:|
| 最终六域平均foreground Dice | 12.44% | **59.14%** |
| 两个seed的六域均值之间的样本SD | 1.86 pp | 1.05 pp |
| 各域刚学完时平均Dice | 32.29% | 75.07% |
| 最终减刚学完 | −19.85 pp | −15.92 pp |
| 最终Dice为零的seed×域组合 | 2/12 | 0/12 |

先在每域计算foreground micro Dice，再对域和seed等权平均。样本SD只有两个seed，不是置信区间；最终减刚学完为描述量，不等同标准BWT。开发验证存在重复使用，不把12个配对当作独立患者样本做显著性宣称。

## 结果解释

此前两个独立域的机制诊断为完整SGD21.13%、完整Adam61.21%，该数字不是六域结果。本次从头运行完整持续学习序列，进一步确认优化器适配带来的改善能够延续到六域设置。直接对照双方同为50epoch/batch4、同seed41/43、同ResUNet32、同stream、同HDF5和选择/推理规则；主要干预是优化器及基础学习率组合，不能单独断言Adam算法或学习率某一因素造成全部改善。两组在不同时段共享硬件上运行，非逐位可复现承诺。

刚学完75.07%而最终59.14%，说明仍有15.92个百分点的平均保留损失，不能说遗忘问题已经解决，也不能凭本结果宣称达到可靠论文主表标准。作者网络150epoch的76.78%仍为不同网络/监督/训练配置的参考，不能直接作为同预算比较或替换本结果。

## 核验与协议

两臂均完整执行6×50epoch，退出码0；最终model/optimizer和五域回放库张量有限，每个旧域至少30项回放。实际新数据步骤共32,250、回放24,650、GIN15,987，和原控制的计划步数一致。总历时约18.79分钟，worker、supervisor和独立存储watcher均已退出。

主线保留CCD .9、GIN .5、首层特征回放、域均衡与前景采样、5%且每域至少30项特征、首任务后冻结encoder含bridge与BN统计。Adam .0002、weight-decay3e-5、poly .9；原SGD控制为.01/momentum.99/Nesterov，其他选择保持。参数2,915,808，batch4；首val元数据块选择earliest-best，其余val块最终评估；未访问test。

## 150epoch仍在运行

按用户要求，已独立从头启动seed41/43、六域每域150epoch的同一完整方法。没有从50epoch最终checkpoint续训，因为poly学习率的计划时间尺度也应按150epoch设定。保持batch4和相同source/sharedstream，不能自动视作原主表150epoch/batch8/原benchmark顺序。

截至本报告，150epoch双seed仍在运行；不根据部分结果判断是否优于50epoch。其协议和150epoch核验脚本已单独提供，待完整结束后再比较。

HDF5域身份、病例边界、预处理来源未证实，metadata序列块不等于独立患者；validation历史曝光、标签沿用整数截断及stream不同于原benchmark等限制继续适用。本轮source/sharedstream为BIDMC→HK→ISBI→UCL→ISBI_1.5→I2CVB。任何后续恢复原benchmark顺序须另立实验，不能追改现有结果。

自写运行代码、协议、核验脚本和跨域跨seed聚合公开；checkpoint、回放特征、患者数据、逐域/逐块结果和原作者源码保持私有。
