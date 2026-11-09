# TED/CauAug 长训与数值修复更新

CauAug原配方双seed六域各150epoch已完成，但最终开发验证平均Dice降至15.70%，其中一个seed最终六域全零。TED的stable_all50组因独立两小时预算到限，只有一个seed完成，不能提供有效的双seed组间比较。

| 完整实验 | 最终平均 Dice | 最终零值组合 |
|---|---:|---:|
| CauAug 原配方50epoch（历史对照） | 59.09% | 0/12 |
| CauAug 原配方150epoch | 15.70% | 6/12 |
| TED 稳定监督项50epoch（已交付） | 61.12% | 0/12 |

CauAug150较50epoch降低43.39个百分点；两个seed六域均值的样本标准差为22.20个百分点，刚学完时平均45.39%，最终减刚学完为−29.69个百分点。延长训练没有改善适配，反而暴露明显退化。数值有限和流程完成不能证明科学效果可靠。不同方法/预算不作公平排名。

## 工程核验与失败记录

CauAug两seed均返回0，全部6×150epoch记录、最终六域输出、model/optimizer/teacher和六域archive有限性通过。CKNet170700次，FTGen和KD各147900次；未删除方法核心组件。总用时127.59分钟，相关进程组已退出。

TED stable_all50一个seed在最终域第16epoch后以124超时退出，另一个完整完成。合计566/600条epoch记录；不报告单成功seed作为修复效果，不重启、不延长原两小时预算。共享GPU负载与速度有关，但不据此臆测完整结果或声称算法失败。未完成臂保留既有checkpoint和日志；已结束的运行进程组均退出。

## 已启动的后继工作

TED stable_sup50完整双seed满足预先声明的平均改善与避免全域崩溃门槛。选择其最小监督项修复进行双seed六域各150epoch独立从头验证，保留batch4、shared顺序、全部蒸馏机制和原Adam/LR；没有从50epoch权重续训。stable_all未完成，因此这个选择只基于stable_sup的充分证据，不代表证实其优于stable_all或已找到最优配置。新组独立三小时，同时受原六小时总窗口约束。

CauAug使用新150epoch最终checkpoint执行固定train切片梯度诊断，判断是否出现旧50epoch最终checkpoint未观察到的错误饱和/零梯度。只诊断、不训练、不访问test；后续只按诊断证据修复，不启动学习率网格或删除核心机制。

## 适用范围

以上均为开发validation，test未访问；metadata块未证明是独立患者，真实域身份/预处理来源仍待核实，validation反复曝光。source/shared顺序、batch4不同于原benchmark配置。旧失败完整保留，不能以新值替换历史记录或充当最终论文成绩。公开仅自写代码/协议及跨域跨seed聚合，患者/逐域逐块值/权重/档案及第三方源码不公开。

## 新150epoch checkpoint 的诊断与修复

两个固定checkpoint共192个均匀抽样train切片，错分前景中旧监督梯度恰零比例合计60.23%，稳定logits写法为0%。其中全域崩溃checkpoint为97.69%→0%，最大绝对logit约4365.86。两路返回0，参数和buffers恢复，数学等价与极端梯度检查通过。相比旧50epoch最终checkpoint无同类采样现象，这说明长训后的失败状态出现了明确的饱和/零梯度，不能证明它是最早唯一诱因。

读取诊断输出前记录了后继门槛：固定诊断两路成功、崩溃checkpoint错前景存在且旧零梯度比例超过50%、稳定写法为0%。本次满足门槛。后继CauAug stable_sup150只将监督项改为稳定logits DiceBCE，保留CKNet、FTGen、六域档案、原width-axis KD和teacher/evaluation模式；不改学习率、不删模块。协议为独立从头双seed、六域各150epoch、batch4，独立三小时且不得越过原23:04:56总窗口。

CauAug修复版本已通过合成六阶段完整训练预检：CKNet/FTGen/KD均激活、六域archive与model/optimizer/teacher有限、验证前后模型buffers及RNG恢复。首次预检入口遗漏DATA_ROOT环境变量，未读取真实数据；保留错误记录、补齐合成占位环境后通过，原计时未重置。真实双seed从头训练已启动，尚不能宣称修复后效果改善。

## 完成更新

两组修复150均已完成，TED52.47%、CauAug56.11%，完整配对均无零值组合，但长训与遗忘问题仍存。见 [本轮收尾](TED_CAUAUG_REPAIR_CLOSURE_ZH.md)；上文运行中描述为历史状态。
