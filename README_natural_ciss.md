# 自然图像类别增量分割方法接入

前四个自然图像 Class-CL 方法为：`ewf`、`lgkd`、`nest`、`cit`。前三者分别为 **MiB＋EWF、MiB＋LGKD、MiB＋NeST 的 ResUNet 适配版**；新增 CIT 为 **Pattern Recognition 2025 的类别独立变换与累积蒸馏适配版**。本轮另外加入 ReBac、LDKA 两个自然图像方法及 PCDD 医学方法，详见文末2026年接入说明。这些是同骨干移植实现，不代表已复现作者论文数值。

此次只做代码实现和 CPU 合成输入测试，没有访问真实医学数据、启动服务器训练、下载模型权重或修改论文。此前的 S3R、TED、医学 CLIP Class-CL 代码一并保留。

## 为什么选择这三篇

| 方法 | 论文 | 本次 OpenAlex 对应记录引用数 | 接入理由 |
|---|---|---:|---|
| EWF | [Endpoints Weight Fusion for Class Incremental Semantic Segmentation，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xiao_Endpoints_Weight_Fusion_for_Class_Incremental_Semantic_Segmentation_CVPR_2023_paper.html) | 51 | 核心是任务结束时融合参数，作者提供 MiB 组合，无需更换为 Transformer |
| LGKD | [Label-Guided Knowledge Distillation for Continual Semantic Segmentation on 2D Images and 3D Point Clouds，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Label-Guided_Knowledge_Distillation_for_Continual_Semantic_Segmentation_on_2D_Images_ICCV_2023_paper.html) | 22 | 蒸馏目标重构，论文明确评估 LGKD＋MiB，迁移到医学背景标签歧义问题较直接 |
| NeST | [Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3745_ECCV_2024_paper.php) | 14 | 年份更近，重点是新分类器初始化；作者提供 MiB 组合，保留原像素分类式骨干 |

引用数查询日期为2026年10月26日，来源是 OpenAlex 的指定记录：[EWF](https://openalex.org/W4386071761)、[LGKD](https://openalex.org/W4390874455)、[NeST](https://openalex.org/W4403842150)。不同数据库和版本合并方式会产生不同计数；这些不是 Google Scholar 总引用，也不代表全领域引用排名。原始查询结果保存在 `docs/natural_ciss_citation_lookup.json`。

选择兼顾了引用积累、实现成本和相对已有 MiB 2020、PLOP 2021 的新近程度，并不宣称它们是截至2026年的最新SOTA。筛选中也看过2025年的 CIT，其对应 OpenAlex 记录当时为3次引用；更早的 DKD 为2022年，且涉及另一套分解分类器。前三篇最初按“较新、较易实现、有引用积累、有源码”选择；本次根据追加的2025年要求接入 CIT。CIT 引用积累较少，不将其描述为高引方法。未进行穷尽式方法排名。

## 作者源码与固定版本

| 方法 | 作者仓库 | 固定提交 |
|---|---|---|
| EWF | [schuy1er/EWF_official](https://github.com/schuy1er/EWF_official) | `560382d5ea6d79b6d49c365988cc5118129dd815` |
| LGKD | [Ze-Yang/LGKD](https://github.com/Ze-Yang/LGKD) | `78262c22043fe9d56261d1906a04c016b82930b5` |
| NeST | [zhengyuan-xie/ECCV24_NeST](https://github.com/zhengyuan-xie/ECCV24_NeST) | `f96185be8a46f7f1812d750e00fe2fd86b7fd2b8` |

GitHub 访问通过用户指定代理。只阅读作者代码和论文，没有执行下载仓库中的训练、下载或删除脚本。来源和许可说明见 `THIRD_PARTY_NOTICES.md`。

## 三种方法分别实现了什么

### EWF

先按 MiB 组合进行当前任务训练：平衡初始化、Unbiased CE、Unbiased KD。任务结束后才执行融合：

```text
alpha = sqrt(number_of_new_classes / (number_of_old_classes_including_background + number_of_new_classes))
shared_endpoint = (1 - alpha) * previous_endpoint + alpha * current_endpoint
```

共享骨干和已学类别行参与融合，新类别行不与旧模型中预分配但未学过的随机行融合。浮点 BN 统计参与融合；整数 BN 计数保留当前值，避免将元数据作浮点插值。融合后的模型再成为下一阶段教师并被保存。不是每个 minibatch 执行的 EMA。

默认开启作者 MiB＋EWF 脚本中的旧分类器冻结：旧类别行（包括背景）的参数不在当前训练步骤更新，但损失仍能通过这些行传回骨干。默认 Unbiased KD 系数10，教师缩放1。`ewf_train_old=true` 是需单独披露的变体。没有加入作者代码中明确标注为论文之外的 threshold-choice 增强。

### LGKD

实现论文评估过的 LGKD＋MiB，而非包含 PLOP 局部 POD 的另一种组合。旧教师概率扩展到当前已学标签空间；对真实标注的新类像素，将教师背景概率转移到该新类，保留其他旧类概率；对背景像素保留原教师分布，新类概率为0。蒸馏只用当前任务原始标注作引导，不利用测试标签，也不把伪标签当成新类真值。

默认旧/背景像素权重5、新类像素权重1，依据论文权重实验的选择及VOC脚本；作者通用 parser 中另有10/2默认值，因此这里明确记录采用的5/1。教师缩放1。忽略标签255不参与 LGKD 平均；某一子集为空或整批全忽略时仍给出有限损失。

作者工程中的蒸馏应用于分割器原生 logits 网格。ResUNet 本来就是全分辨率 logits，因此本适配在该网格计算，不额外伪造一个1/16分辨率头。监督损失沿用此组合的 MiB Unbiased CE。

### NeST

完整保留新增分类器预调的关键过程：

1. 用冻结旧模型在当前任务训练图像上提取特征与旧类概率，按作者选择流程缩放到默认128×128网格。
2. 对每个新类像素，计算旧分类器权重与特征的逐通道乘积正值掩码，并以旧类概率加权，得到各新类的旧类/通道重要性矩阵。
3. 用该矩阵及其通道求和后的 softmax 初始化重要性和投影参数；新类权重通过旧分类器的加权组合生成，同时学习背景权重变换。
4. 使用当前任务训练集，以 Unbiased CE 预调这些临时参数及新类偏置；特征提取器与旧分类器冻结。
5. 编译出普通新分类器，按作者流程匹配背景权重、新类权重和偏置范数，然后移除临时变换参数，再进入 MiB 训练。

默认额外预调5个 epoch，预调 SGD 使用 momentum 0.9、Nesterov、weight decay 1e-4、poly power 0.9，学习率为框架基础学习率乘 `nest_pre_lr_scale`。这参考作者VOC配置，不宣称是医学数据上的最优值，也没有搜索这些参数。预调额外成本和一次相似度统计遍历应计入后续冻结预算。

预调使用冻结教师作为与旧骨干等价的特征提取器，并置为 eval，避免 BN 统计在“冻结”阶段偷偷变化。调度器按 optimizer 更新后推进，避免原旧工程的调用顺序警告。遇到零范数向量跳过无法定义的缩放；不会随机替换分类器。如果某个新类在当前训练集的选择网格中完全没有标注像素，会明确报错，保持分类器未提交，不以零除或静默回退掩盖问题。

`initialization_info` 在完整 checkpoint 中保留相似度矩阵、学习后的重要性/投影矩阵、预调 epoch、步数与最终预调损失。它们属于训练审计状态；实际部署网络不带临时变换参数。合成测试为节省时间只设置预调1个 epoch，不构成真实实验协议。

## 追加：CIT（Pattern Recognition 2025）

[CIT: Rethinking Class-incremental Semantic Segmentation with a Class Independent Transformation](https://jinchaogjc.github.io/cit-demo/) 正式发表于 Pattern Recognition 2025，文章号111707；[arXiv预印本](https://arxiv.org/abs/2411.02715) 首次提交于2024年11月5日。它是自然图像类别增量**语义分割**方法，作者在PASCAL VOC和ADE20K上实验，提供CNN与Transformer实现。

参考作者仓库 [jinchaogjc/cit](https://github.com/jinchaogjc/cit)，固定提交 `bc20448317262f71cc9ccb69cc057b602bb8d869`。本次选择CNN形式，新增 `models/cit.py`、`models/utils/cit_head.py`、`configs/cit.json`、`tests/test_cit.py`；不依赖MMseg或额外视觉预训练权重。

**保留的机制：**每个前景类别一个query，无query之间的self-attention或FFN；Q/V/输出映射为恒等映射、K可学习。空间注意力使用8个head，head维度缩放；softmax之前的注意力分数跨head平均形成mask logits。注意力聚合经query残差与LayerNorm输出共享的存在/无目标二分类分数。推理前景分数为存在概率乘mask sigmoid，类别之间无需softmax归一化。

每个任务结束保存冻结快照。后续蒸馏时，每个旧类别的目标取自**首次被直接监督的任务快照**，按全局类别次序拼接，不把所有类别都改用最近教师。存在性和mask分别使用二元蒸馏，默认权重5/10、教师缩放1。当前新类使用直接query对应、存在性CE、sigmoid focal及Dice，默认权重1/20/1，focal alpha=0.25、gamma=2、Dice平滑1、无目标类别权重0.1。

**为现有医学框架作的适配，需要在后续论文中披露：**

- 复用可训练的2D ResUNet，默认将解码特征平均池化到stride8后接CIT头；没有复用作者DeepLab/ASPP、ImageNet预训练或其原始训练时长。池化改变空间细节，虚拟测试不证明其分割精度。
- 原医学标签0代表部分标注背景，因此仅对当前正类GT区域清零旧mask蒸馏目标；背景区域继续蒸馏旧类。标签255排除像素监督与mask蒸馏，全忽略图像不参与当前query的存在性监督。
- 医学背景没有单独query，定义背景分数为 `1 - max(前景分数)`，与前景共同argmax，相当于前景最大分数须超过0.5。这是显式背景适配，不能称为作者原始背景策略；没有用真实数据调这个阈值。
- 预分配全部前景query，未来query不参与损失或预测，输出固定为-1e4。CIT源码优化器修正v2采用AdamW：base lr2e-5、head倍率10、wd0.01；每步恢复未到达query并清零对应Adam矩。保留benchmark的epoch80乘0.5，未照搬原iter warmup/PolyLR，详见experiments/20261026_five_seed/CIT_SOURCE_OPTIMIZER_V2_ZH.md。
- 因骨干可训练，每个历史教师保留完整网络。训练计算、内存和checkpoint随任务数增加；推理只需当前网络。部署参数计数排除教师，但后续资源表还必须另报训练峰值和保存开销，不能声称其训练开销恒定。

已覆盖独立query扰动、独立attention参照、三任务生命周期、原始任务教师选择、冻结参数与BN、教师银行完整checkpoint恢复、二元蒸馏数值、背景/忽略标签、未来query隔离、256×256输入及完整框架train/test-only合成流程。当前完整测试结果见 `IMPLEMENTATION_STATUS.json`。

## 统一接入与未改变的部分

- 四种方法均使用共享标签空间，只支持 `class-il`；推理不利用 task ID 选头。
- 复用原 `begin_task / observe / end_task` 接口和前一批实现的完整 checkpoint 机制。
- 全部分类器行按照框架既有全局标签空间预分配，但只有已到达类别参与 softmax、损失与预测；未来行保持不更新。因此本实现不声称按动态分配分类器获得参数增长优势。
- 为实现 MiB 平衡初始化，对原无偏置的最终1×1分类器增加显式零初始化偏置，其余 ResUNet 随机权重保持原样。该适配需要在资源与架构描述中披露。
- 当前部分标注协议要求旧类在当前训练标注中为背景，只接受当前任务新类、背景及忽略值255。意外出现旧类真值或未来标签会报错。
- 原 `models/mib.py`、`models/plop.py` 不被替换或调参。三种新组合的公共 MiB 损失通过现有参考公式对照测试；这不等于已核实旧主表与新方法的全部实现细节一致。
- 原评估器的按测试任务类别掩码规则、患者划分和指标口径仍保留。真实实验前应统一信息条件与全类别评估，不能仅因新方法能无 task ID 预测就把旧评估结果称为任务无关全类别成绩。

这些是同骨干比较所需的适配，不能将未来取得的医学结果与作者 DeepLab/ResNet 的自然图像论文数值直接比较。没有迁移 ImageNet 视觉预训练、APEX、inplace-ABN 或原分布式训练配置，也没有调用外部视觉/语言模型。四种自然图像新增方法不需要额外的预训练权重文件。

## 文件和配置

| 用途 | 文件 |
|---|---|
| EWF | `models/ewf.py`、`configs/ewf.json` |
| LGKD | `models/lgkd.py`、`configs/lgkd.json` |
| NeST | `models/nest.py`、`configs/nest.json` |
| CIT | `models/cit.py`、`models/utils/cit_head.py`、`configs/cit.json` |
| 共同标签空间与 MiB 机制 | `models/utils/class_incremental.py` |
| 数学行为和生命周期测试 | `tests/test_natural_ciss.py` |
| 合成数据完整框架测试 | `tests/test_framework_integration.py` |

配置模板沿用框架基础训练参数，只是后续可审阅的起点；当前没有启动实验。继续通过中性入口读取配置，方法名放在环境配置中：

```sh
RUN_CONFIG="$PWD/configs/ewf.json" python run.py --validate-only
```

该命令只检查参数语法，不构造模型或读取真实数据。方法名与配置不进入 Python 可见启动参数。实际训练前仍须确认数据路径、类别编号、协议、预算和服务器规则。

## 测试结论

运行 `python -m pytest -q`。当前完整测试结果见 `IMPLEMENTATION_STATUS.json`。覆盖前一批医学方法的回归、新三种方法的三任务切换、教师参数/BN冻结、未来行隔离、完整训练与test-only循环、CSV、标准256×256输入、参数融合公式、概率质量守恒、MiB损失对照、LGKD独立逐像素参照、NeST统计公式及冻结/编译行为、错误设置、缺失新类和检查点配置不匹配。

全部输入由本地张量构造，临时检查点与分数无科研含义。源码测试通过只说明接口和所测机制行为正确，不证明医学数据上的有效性或与原论文结果一致。


## 2026年追加接入：ReBac、LDKA、PCDD

**交付范围：只实现 Class-CL 的2D ResUNet适配与CPU合成输入检查。三个方法均拒绝 domain-il / task-il；没有运行真实数据实验、下载模型权重或修改投稿。** `rebac` 是固定作者代码的类感知背景选择版本（ReBac-C式），并未另外实现ReBac-B。`ldka` 和 `pcdd` 是按论文重构的适配实现，不能标为“作者源码复现”。

| 方法 | 论文与实现依据 | 入口 |
|---|---|---|
| ReBac | [IJCV 2026](https://link.springer.com/article/10.1007/s11263-026-02827-x)，[作者代码](https://github.com/ANDYZAQ/ReBac)，固定提交 `dae1b48d461ba62bab99414d57d6474f5aa3d80b` | `models/rebac.py`、`models/utils/rebac_head.py`、`configs/rebac.json` |
| LDKA | [ICML 2026正式论文](https://proceedings.mlr.press/v306/zhang26jo.html)，正文公式及补充材料Algorithms 1/2；本次未取得作者实现 | `models/ldka.py`、`configs/ldka.json` |
| PCDD | [AAAI 2026正式论文](https://ojs.aaai.org/index.php/AAAI/article/download/38406/42368)，[扩展预印本v1](https://arxiv.org/html/2511.07749v1)；作者仓库检查版本 `ad9f5094da97ba5e002be0d467e27171f8d907c6` 尚无方法源码 | `models/pcdd.py`、`configs/pcdd.json` |

来源与公式解释另存 `docs/2026_method_sources.json`。作者论文/源码只作研究依据，没有运行其下载脚本、训练脚本或数据处理脚本。

### ReBac：保留任务分支和训练/推理不同的背景规则

在共享ResUNet特征上，为各任务建立3×3 Conv/BN/ReLU分支和1×1分类器。基础分支输出背景及基础类；新增分支输出当前新类，并有单通道背景残差适配器。残差为适配器输出减去新类最大logit的ReLU；后者停止梯度，匹配作者头部实现。

训练时，当前背景使用 `base_bg.detach() + current_residual`。推理时，先根据各任务前景logit最大值选出获胜任务；基础任务修正为0，其余任务使用 `-relu(-residual)` 修正基础背景。没有将训练规则错误地照搬到推理，也不需要测试task ID。

损失包括：sigmoid置信度阈值0.7的旧类伪标签、各类别BCE、伪背景/背景适应损失、背景掩码特征蒸馏、按任务分组的二元知识蒸馏。默认额外项系数依作者训练器为5/4/1/1。监督项停止更新旧前景logit，在高置信旧类伪标签区域也停止背景监督梯度；其他蒸馏仍可更新旧分支。

必须披露的适配与修复：

- ResUNet解码特征替代DeepLab/ASPP，分支宽度使用原ResUNet输出宽度，而非作者256维特征。新增分支卷积与BN仿射值从基础分支初始化，BN运行统计不复制。
- 保留作者增量阶段冻结BN统计及仿射参数的规则。所有任务分支按预先声明的任务分组一次性分配，未来分支不执行、不参与预测和优化。因此参数计数包含未来分支，不声称本实现按任务动态增长获得资源优势。
- 作者增量优化器没有纳入 `classifier.adps`；本适配明确将背景残差适配器纳入优化器，否则这些参数不会更新。这个正确性修复及其配置应在方法比较中记录。
- 显式忽略255；对空背景、空前景和全忽略批次作零损失处理，修复作者空区域平均可能产生NaN的问题。对正常非忽略输入，保留作者按完整网格平均背景掩码特征误差的方式。
- 使用SGD momentum=0.9、Nesterov、weight decay=1e-4。基础学习率0.01；增量新分支/残差适配器0.001，共享ResUNet及旧分支0.0001。没有ASPP组；每任务重置SGD动量，与作者逐任务独立启动一致。

### LDKA：PML、SKD、MAD及epoch状态

PML按真实/伪标签类与最强竞争类的logit差计算margin，使用 `kappa = p_y + p_c - (p_y - p_c)^2`，再以 `R=abs(margin)/kappa` 的经验CDF作百分位排名。相同R使用同一CDF值，忽略像素不参与排名；在log域计算归一化权重，避免幂运算溢出。训练进度采用 `(epoch+1)/(total_epochs+1)`，并以默认1e-3限于开区间，明确处理论文端点奇异性。权重本身停止梯度，作用于逐像素CE。

SKD按补充材料Algorithm 2，对特征差异的平方L2范数求一阶梯度并保留计算图，再惩罚其平方范数，产生双重反向传播。**这不是显式完整Hessian；对于平方误差，它代数上等价于相应缩放的特征差异惩罚，不能据此声称计算了真实高阶Hessian。** Algorithm 1的扰动在非线性decoder之前注入，再约束传播前后响应，而不是直接向最终特征加单位向量、形成常数损失。

MAD累计当前epoch真实/伪标签到预测的计数，epoch结束后按行归一化，作为下一epoch的竞争矩阵。对高混淆类别对惩罚类向量内积平方；初始矩阵为0。混淆矩阵、计数、训练进度都进入完整checkpoint。训练器新增可选begin_epoch/end_epoch回调，既有方法不需要这些回调。

必须披露的重构选择：

- 原论文为ViT-B/16及Transformer decoder；此处用ResUNet最后一个解码残差块作为非线性H，前一拼接特征作为扰动输入，最终解码特征作为蒸馏响应。分类器权重行代替原类别token，不额外引入Transformer。
- 通道L2平方后对空间位置和batch取均值；扰动按每个位置的通道向量归一化、幅度1，方向停止梯度。这些张量维度/求导细节原论文没有完整实现可供对照，属于明确的适配解释。
- 额外SKD传播暂时固定BN统计，原监督传播仍按原训练态执行，避免一个minibatch重复污染BN统计。所有模式在上下文结束后恢复。
- 原论文未明确伪标签筛选阈值，本实现默认将背景位置替换为旧教师argmax，阈值为0；只接受当前新类GT、背景和255，不读取旧类真值。新分类器初始化复用框架MiB平衡背景拆分，属于共同初始化适配。
- 默认SGD momentum=0.9、weight decay=1e-5，基础lr=0.001、增量lr=0.0001，采用正文VOC阶段学习率。每任务重置动量；对预分配的未来分类器行恢复其原值并清空动量，防止weight decay让尚未到达的行偷偷更新。

### PCDD：PGCD及DAPD的论文重构，须披露公式解释

实现原始熵小于0.7的旧类伪标签、旧区域校准蒸馏ORCD、当前区域校准蒸馏CRCD，以及本地到本地/本地到全局两路原型蒸馏DAPD。监督项复用MiB Unbiased CE；新分类器使用共同的MiB平衡初始化。默认系数为ORCD=1、CRCD=0.5、本地对齐=0.5、全局对齐=0.1，依据扩展版本的权重表及讨论选择，未在医学数据上搜索。

以下解释不能省略，也不能冒充已与作者实现核对：

1. 扩展稿Eq.3累计均值的分子存在笔误，按文字含义实现像素加权累计：`sum += batch_class_mean * pixel_count; count += pixel_count; mean = sum/count`。每类有独立计数，忽略像素不计入，缺失类别不伪造原型。旧前景全局原型保留，当前类别持续累计。
2. 每个任务重算背景；教师背景原型在当前数据上按背景与新类合并后的标签累计，避免跨任务沿用旧背景。当前模型背景单独累计，旧前景原型不混入新类GT。
3. Eq.10背景和式的下标按各类原型解释：当前背景均值与所有出现的新类均值相加，形成旧背景空间中的q，而非按像素频率重新池化。这一点有独立数值测试。缺失类不贡献虚构向量；DAPD按旧类别总数归一化。
4. CRCD公式对新类别没有对应教师通道，无法直接逐通道KL。此处明确采用MiB共同旧标签空间：学生背景+新类概率合并为教师背景，当前亲和度也将这些类的权重质量合并到背景，再对齐旧前景通道。这是保守的空间对齐解释，**不能称为已确认的作者CRCD实现**。
5. 按论文公式采用KL方向 `student || teacher`，ORCD的亲和度由冻结教师特征与旧前景原型计算，CRCD亲和度由当前特征与有效全局原型计算。类通道加权的KL分量和可能为负，不对它进行未经说明的截断或反向KL替换。
6. 原论文是3D SwinUNETR；这里为2D ResUNet解码特征，没有3D上下文。ResUNet引入了学生训练态和教师eval态的BN差异，合成连续更新检查观察到由此引起的原型蒸馏失稳。因此增量阶段固定学生BN运行统计，仍训练BN仿射参数；此行为由网络buffer保存，直接调用net.train()也不会解除冻结。

默认SGD momentum=0.9、固定lr=0.01、未额外设置weight decay。每任务重置动量，checkpoint内仍完整保留任务中途的动量及原型。全局原型候选在有限优化步骤成功后才提交；如果当前任务结束时某个新类始终没有原型，则明确报错，保持任务未完成，不能静默随机初始化代替。

### 配置、检查与边界

`configs/rebac.json`、`configs/ldka.json`、`configs/pcdd.json` 可由中性入口读取。配置使用seq-mmwhs、mid ResUNet、150 epoch和batch size 8作待审阅的框架模板；这不是三篇论文的原始自然图像/3D医学训练协议，也不是已冻结的真实实验方案。ReBac、LDKA沿用框架StepLR（默认80 epoch衰减），并非作者完整调度；PCDD将衰减系数设为1，保持论文固定学习率。

```sh
RUN_CONFIG="$PWD/configs/rebac.json" python run.py --validate-only
```

`--validate-only`只检查配置语法。当前没有启动任何真实训练。实际数据到位后仍需核对标签顺序、部分标注条件、患者划分，以及旧评估器的按测试任务类别掩码；本轮没有修改原有评估信息条件或论文主表。

测试见 `tests/test_2026_methods.py` 和完整框架测试，覆盖三任务、未来参数隔离、冻结教师、标准256×256输入、空区域/255、原型均值与概率质量守恒、ReBac训练/推理背景公式、LDKA经验CDF/非线性扰动/双重反向传播、PCDD原型对齐、完整checkpoint及恢复后的下一步更新一致性。LDKA的框架测试使用两个epoch验证真实训练器回调、进度、竞争矩阵与阶段学习率。最终通过数、运行时间及合成失败诊断见 `IMPLEMENTATION_STATUS.json`。

这些测试证明所测公式、接口和状态行为，不证明真实医学精度，也不代表三篇论文的完整数值复现。尤其PCDD的公式歧义和3D到2D适配应在正式对比前明确标注。


## Organ-CL独立适配：EWF、LDKA、CIT、CLIP-CIS

此次适配针对原论文中“左心房 → 前列腺 → 肝脏 → 脑肿瘤”的Organ-CL，即 `seq-taskincre / task-il`：四任务图像域和前景语义不同，每任务GT保持本地0/1，255表示忽略。测试任务ID用于选择该器官的二分类输出。**以下名称均是明确的Organ-CL适配版本，不是原作者已验证的Organ-CL算法，也不能与原Class-CL入口混称。**

| 注册名 | 配置 | 保留机制及适配 |
|---|---|---|
| `ewf_organ` | `configs/ewf_organ.json` | 端点权重融合；独立二分类头及旧任务头蒸馏 |
| `ldka_organ` | `configs/ldka_organ.json` | 当前任务PML/MAD、共享特征SKD双重反传及非线性扰动 |
| `cit_organ` | `configs/cit_organ.json` | 每器官独立query；保留各器官首次监督阶段的教师 |
| `clip_cis_organ` | `configs/clip_cis_organ.json` | 每器官图像/文本条件动态头；独立旧任务伪掩码 |

共享代码在 `models/utils/organ.py`。这些入口只支持task-il，并检查任务数、本地二分类标签和任务名称/顺序；`model(x, task_index=k)`返回[B,2,H,W]。没有task ID、ID越界、错误setting或全局类别编号会明确报错。四种原Class-CL方法不改为隐式多setting分支，其已有测试继续保留。

### EWF-Organ

监督当前二分类头，使用上阶段冻结模型在当前图像上产生各旧任务头的分布，以各头独立的categorical KL约束学生，默认按旧任务数平均、系数10。每个头有自己的背景，不把其他器官的预测写进当前器官GT，也不使用MiB的全局背景池化。`ewf_freeze_old=true`时旧头权重停止梯度，蒸馏仍能传回共享骨干。

任务结束后对共享骨干及旧任务头做端点融合，新头与未来头不融合。第t个新增任务（从0计数）的新端点系数为 `sqrt(1/(t+1))`，即将每个独立二分类头的bg/fg均计作其自身的两个输出：2/(2t+2)。这是从Class-CL类数量比例到多头任务数量比例的明确扩展。浮点BN统计参与融合，整数计数保留当前值。融合后的网络再成为教师。训练目标因此是“EWF＋任务头KL适配”，不是Class-CL版本的MiB＋EWF。

### LDKA-Organ

PML只处理当前头的GT0/1，不生成跨器官全局伪标签。MAD也只在当前任务的背景/前景两条分类器向量之间解耦，每任务分别保留2×2混淆矩阵；epoch结束行归一化供下一epoch使用，不把不同器官的背景或前景编号当成同一语义。

SKD约束共享ResUNet特征，沿用前面LDKA的Algorithms 1/2实现、额外传播BN控制与明确求导解释。过去和未来分类器头均不参与当前监督/MAD，且不会被SGD的weight decay或动量更新；共享骨干仍可更新，旧任务性能需要在真实数据上验证。默认基础/增量lr为0.001/0.0001、momentum0.9、weight decay1e-5。每任务清空SGD动量；中途checkpoint保留当前动量、进度及各任务的竞争矩阵。

### CIT-Organ

每个器官对应一个query和存在性分数。当前任务只监督其自己的query，复用存在性CE、mask focal与Dice。旧query的蒸馏目标来自各自首次被直接监督时的独立完整网络快照，而不是全部改用最近教师。

本地GT始终为0/1，不需要改成任务序号。旧器官query可能与当前器官前景同时为正，所以 `cit_zero_gt` 强制为false；当前前景不会抹除旧任务蒸馏目标。推理只取目标query的 `presence_prob * sigmoid(mask)`，其背景为 `1 - target_score`；其他器官分数不参与背景计算或argmax。训练时仅计算已到达query，未来query不更新。

仍使用固定总query数和ResUNet stride8特征等前述适配。教师银行随完成任务数增长，每个快照含完整网络；部署只使用当前网络。必须另报训练内存和checkpoint开销。

### CLIP-CIS-Organ

每任务用相应文本行和图像上下文生成动态卷积参数。监督目标是在旧任务伪二值掩码后拼接当前器官GT；不沿用Class-CL中“当前GT前景覆盖旧类掩码”的规则。四个任务独立，没有互斥全局标签竞争。

推理只运行被task ID选中的动态头，输出 `[0, foreground_logit]`，其softmax前景概率等价于sigmoid。训练只计算已到达任务头。监督使用有效像素的BCE和原形式逐样本/逐任务Dice，显式排除255，全忽略批次返回有限零损失。

**真实训练前必须补充本地真实CLIP文本编码，当前没有生成或打包可用于科学实验的随机编码。** 文件格式：

```python
{
    "embeddings": tensor_of_shape_4_by_512,
    "class_ids": [1, 2, 3, 4],
    "class_names": ["left_atrium", "prostate", "liver", "brain_tumor"]
}
```

这里class_ids是文本行/任务序号映射，**不是训练mask的全局类别编号**。真实CLIP的提示词可以采用规范英文器官名，但metadata须按上述任务顺序标记；更改任务顺序必须同步数据流和metadata，不能只改编码文件。名称或顺序不匹配会拒绝构造模型；checkpoint加载也会核对编码张量，拒绝同名但内容不同的编码文件。配置中的 `REPLACE_WITH_LOCAL_ORGAN_CLIP_EMBEDDINGS.pt` 是占位路径。测试临时编码仅用于验证接口和数值行为。

### 框架接入、资源和检查

- 原dataset的task-il、本地0/1标签和按任务ID评估保持不变。为四个新入口修复原 `get_backbone` 忽略baseline_size的问题，按实际ResUNet宽度生成四个二分类头；既有其他方法默认骨干不变。新增 `TASK_NAMES` 用于检查当前固定数据流的名称顺序。
- 所有任务头/query/controller按预声明任务数预分配，参数计数包含全部已分配模块。CIT和CLIP动态头替换原二分类线性头，不保留无用的原头参数。
- 原训练器在训练前和每个阶段评估全部任务以计算FWT。因此，允许用显式task ID查询尚未训练的随机头；这些输出仅服务既有FWT协议，不代表学过未来任务。训练损失不会使用未来头。
- checkpoint仍使用medical-cl-v1，保存教师/教师银行、任务ID与顺序、学习进度、优化器和scheduler。四个新注册名与Class-CL名称分离，不能互载。新增organ参数进入目标配置一致性检查。
- 不下载CLIP模型、读取患者数据或启动真实训练；也未改论文或重算主表。配置中的epoch/batch size仍为待审阅模板。源方法在跨器官/跨模态数据上的有效性尚未验证，旧头权重保持不变也不代表共享特征不遗忘。

检查配置语法：

```sh
RUN_CONFIG="$PWD/configs/ewf_organ.json" python run.py --validate-only
```

专项测试为 `tests/test_organ_methods.py`：四任务本地标签、任务路由、原训练器/train/test-only、教师参数与BN冻结、未来参数不更新、端点融合公式、独立背景、旧伪掩码不被当前GT覆盖、编码/顺序校验、标准256×256输入、空前景/全255、完整checkpoint，以及恢复后下一步更新一致性。CIT一致性测试使用相同随机状态，以正确对齐dropout。当前完整回归结果见 `IMPLEMENTATION_STATUS.json`。

2026-10-27修正：Organ-LDKA采用显式增量gradient clip=1（基础任务关闭），原loss/lr/BN路径不变；这是Organ适配的数值保护而非论文原生超参，所有seed统一，详见experiments/20261026_five_seed/ORGAN_LDKA_GUARD_V2_ZH.md。旧失败与诊断负结果均保留，未完成多任务前不声称复现性能。
