# TED/CauAug 改进：固定权重数值诊断与后继协议

用户授权为TED和CauAug新增6小时、累计新增存储严格小于10GB的独立窗口，只用train/validation。FOCUS已认可的73.59%版本固定；正在进行的FOCUS协议核对正常收尾，不恢复FOCUS调参。

## 发现及证据

旧统一ResUNet32结果中，TED一个seed最终六域全零。新source-component入口使用softmax概率截断后计算BCE；原benchmark分支此前已有logits稳定BCE实现，但这项数值保护没有沿用到新入口。源码检查和本次实测支持补上这项保护，而不能仅凭参数finite认定训练数值正常。

固定旧最终checkpoint，每方法两个seed，每个域均匀抽取最多16个train切片（每方法共192个切片），只测量损失对logits的梯度，不训练、不选checkpoint、不访问test，最终恢复全部参数和BN buffers。

| 跨六域双seed聚合 | TED | CauAug |
|---|---:|---:|
| 采样错分前景像素中，旧监督损失梯度恰为零 | 78.61% | 0% |
| 稳定写法下对应零梯度比例 | 0% | 0% |
| 全部采样错分前景均零梯度的seed checkpoint数 | 1/2 | 0/2 |

TED失败checkpoint的采样最大绝对logit约15457.6；它在本次固定采样中所有错分前景像素均失去旧监督损失的纠正梯度。稳定写法恢复非零梯度。上述计数是既有模型上的数值诊断，不是模型已经恢复或未来训练必然成功的证据，也不证明饱和是最早的唯一诱因。

在两类softmax和one-hot标签下，概率BCE在实数数学上等于类别CE。保留相同flattened Dice（smooth=1），将BCE项从logits稳定计算即可避免概率截断/下溢切断梯度。中等logit的损失值与梯度在float64下通过等价检查；极端错误logit案例确认旧写法零梯度、稳定写法非零且有限。极端区间有意修正了旧clamp/浮点饱和行为，并非声称与原浮点计算逐位相同。

CauAug当前最终checkpoint没有同样的采样零梯度现象；这不排除训练过程中的瞬态问题，但不据此主动改写其损失。

## 已启动的匹配实验

TED两种修复各seed41/43、六域各50epoch：

- **stable_sup**：只将监督DiceBCE改为稳定logits计算；图像KD、Gaussian SKA的概率路径及AKT/GUGF保持。
- **stable_all**：在前者基础上，将图像/SKA KD改为log_softmax计算；保持原class-axis归约、AKT/GUGF的原batch-broadcast行为、teacher train-BN、损失权重与Adam配方。

两组继续同ResUNet32、batch4、source/shared域顺序、首val块earliest-best、剩余val块评估。比较完整双seed结果和旧TED50记录，不挑成功seed；优先采用足以解决问题的最小修复。两种修复均通过合成反向、teacher参数不变和验证状态恢复检查，真实训练已开始。

CauAug则保持原损失、CKNet、FTGen、域档案、width-axis KD和teacher/evaluation模式不变，独立从头运行双seed六域各150epoch，与已保存的同协议50epoch结果59.09%作训练充分性比较。

以上训练尚未完成，不能声称TED稳定性已修复或CauAug性能已提升。训练入口、冻结协议和核验脚本公开，原作者源码不再分发。

## 预算与限制

新窗口为2026-11-09北京时间17:04:56–23:04:56，总新增<10GB，8GB提前停止。TED两组各独立2小时、CauAug独立3小时；各组2GB预算/1.8GB提前停止，同时受窗口累计守卫约束。每个方法/修复组一臂失败停止配对，保留证据，不自动盲重试。

HDF5病例/域身份和预处理来源、元数据块独立性仍未充分核实；validation历史曝光，label整数截断不变，stream/batch仍与原benchmark完整配置存在差异。所有结果仅用于开发，test未访问。公开仅包含自写代码、协议、跨域跨seed聚合，患者/逐域逐块指标/权重/特征/档案/凭据保持私有。

## 2026-11-09 完成更新

监督项稳定计算的TED50双seed已完成并核验：最终61.12%，0/12零组合，但仍有明显遗忘。详见 [完整报告](TED_SUP50_NUMERIC_REPAIR_ZH.md) 和 [聚合核验](TED_SUP50_NUMERIC_REPAIR_VALIDATION.json)。stable_all与CauAug150尚在运行；上文“尚未完成”为启动时状态。
