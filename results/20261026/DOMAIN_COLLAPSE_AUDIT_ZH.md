# S3R/TED 全背景退化：实现和配置审计（2026-10-27）

## 结论与证据边界

旧结果应标记为数值实现/配置失效证据，不能作为有效的原方法复现负结果。这是对先前解释的纠正，旧指标、日志、checkpoint、退出码与成本全部保留，不改写旧测得数值。

两者在第一任务T0就已退化，不是进入新域后才发生的灾难遗忘。用含前景的真实BIDMC train batch8检查：标签0/1且前景约4.08%，两者前景预测0、最大前景概率精确0。TED背景-前景logit差约5.02e8至3.49e11；S3R约1.23e3至2.26e5。旧概率BCE值都4.0766，但对全部logits的梯度精确0；数学等价的稳定logits BCE（binary时即CE）给出非零纠正梯度。梯度检查没有进行优化更新、未访问test，也未公布患者切片。

这是可直接验证的锁死机制：softmax饱和/下溢到0/1后，再用概率BCELoss，其内部有限log截断无法恢复softmax已经消失的导数。loss保持finite不等于训练有效。S3R T0保存的importance最大值0，选择性保护实际上没有有效梯度来源。此前只测正常范围随机输入/少量真实步数，未覆盖训练后极端logits的错误饱和场景。

不能唯一还原最早哪一个训练batch使logits开始放大：原训练缺逐batch参数/梯度快照，不重放完整150epochs，不声称所有数据集有无同一触发因子。原S3R代码本身也使用softmax后BCE，这个弱点在本ResUNet适配的大幅logits下暴露；不能把它描述成论文理论错误。

## 配置核查

TED确有标准配置遗漏：当前继承通用SGD lr.008/epoch80x.5；固定作者仓库TED_DCMIS提交446f79be52d3e83dea92d7ff352c883154404fbb的args.py/main.py实际为Adam，base2e-4、incremental1e-4、逐epoch gamma.99、每域新建optimizer，Adam分支不加weight decay。该修正依据源码，不是测试集调学习率。loss为Dice+BCE，AKT/SKA/GUGF公式和lambda.001维持原配置。作者get_outputs有概率clamp，稳定logits求值并非该clamp分支的位级复现，须作为显式数值适配披露。

S3R当前v2的Adam5e-4/wd1e-4/gamma.98（每epoch前衰减，跨task保留矩）、foreground Dice、SDF、shape embedding/importance系数与固定源2bd7808a4557fad7a0328b5e0b563d08d657990e相符。问题不是仍在用旧SGD；共享概率BCE的梯度锁死仍在。ACNN50epochs、ResUNet/dropout是已有显式适配，不借本次审计调整其超参。train标签有前景，不是标签全0；相同checkpoint在train/eval模式的代表性检查均出现饱和，不能只归因于eval模式。

## 修正协议 v3

新增稳定softmax-onehot BCE：binary情况下数学上就是categorical CE，多类用log_softmax和其他类logsumexp计算补概率。普通数值下与原概率BCE的值及梯度相同；极端错误预测下有意修正下溢导致的零梯度，不加clamp/nan_to_num、不改目标/权重。S3R保持0.5BCE+0.5foreground Dice；TED保持全通道Dice+BCE。新增s3r_bce_impl/ted_bce_impl用于旧协议审计，默认logits。旧概率BCE/旧SGD checkpoint拒绝当新协议无损续跑。

TED统一ted-source-logits-v3：Adam、base2e-4/inc1e-4、gamma.99逐epoch、每域清空矩、框架gamma1。S3R统一s3r-source-logits-v3，仅稳定BCE，保留v2优化器与其他参数。batch8/150epochs/task、seed1..5、数据划分、任务顺序不变。不能把参数完全不同的旧TED称为作者配置。两方法各5seed统一protocol覆盖，不改变其他方法。

162项本地CPU测试通过（含普通范围数值/梯度等价、极端错误背景纠正梯度、TED逐域优化器/学习率和旧checkpoint拒绝）。另做两个真实train任务的有限资格，不复用资格分割权重；S3R资格只复用当前对应训练mask的已训练shape prior以隔离数值验证，正式训练仍按原50epochs生成新prior。局部有限性不证明完整训练性能。

## 原时钟、旧结果与停止条件

旧TED seed1和S3R seed1/2完整运行在attempts/domain_probability_bce_v2中保留，标为superseded数值失效，不再与v3混合统计。保留原seed身份、started_at/deadline，重新初始化不是无损恢复；丢弃训练更新/用时如实报告。已测旧数值不删除，不伪装为独立新实验。7个尚未启动的TED/S3R队列在资格完成前暂挂，其他运行不终止、不重启。

正式修复/资格/重试/test共享原24h，每次启动以原剩余时间设置timeout并预留5分钟；不延钟或拆名。若预算不够完成，则到限失败，不能为了凑五seed延长。仍仅GPU1/5各最多6并发，新启动空闲>=16GiB；根新增40GB预停<50GB。资格权重/CPU更新单列，不用test选择超参或checkpoint。
