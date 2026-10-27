# 新增医学持续分割方法

本次仅实现方法并进行 CPU 合成输入测试，没有加载患者数据、启动真实训练、访问服务器、下载预训练权重或修改论文。原始代码 ZIP 保留；所有改动位于当前解压副本。

本文记录第一批医学方法。后续自然图像方法、2026年方法及本轮四种Organ-CL适配，见 `README_natural_ciss.md`；完整方法清单与当前验证结果以 `IMPLEMENTATION_STATUS.json` 为准。

## 方法与适用范围

| 注册名 | 对应研究 | 支持的 setting | 实现机制与边界 |
|---|---|---|---|
| `s3r` | S3R，TMI 2023 | 二分类 `domain-il` | 不确定性加权语义、SDF 回归、ACNN 形状编码、累计参数重要性与选择性正则；不支持类别扩展或多头 Organ-CL |
| `ted` | Tri-enhanced distillation，MedIA 2024 | 共享标签空间的 `domain-il`，二分类或多分类 | 图像蒸馏、标准高斯输入上的 SKA、AKT 与 GUGF；不将域蒸馏冒充类别增量 |
| `clip_cis` | Continual Learning for Abdominal Multi-Organ and Tumor Segmentation，MICCAI 2023 | `class-il` | 图像特征与 CLIP 文本条件生成类别专属动态卷积头、独立 sigmoid 预测、旧类伪掩码；2D ResUNet 适配版 |

此表三个原入口保持各自setting。本轮另提供 `ewf_organ`、`ldka_organ`、`cit_organ`、`clip_cis_organ` 四个独立Organ-CL适配入口，详见 `README_natural_ciss.md` 文末；不能用原Class-CL/Domain-CL入口替代这些版本。未实现前期候选CGR或Low-Rank MoE。旧框架 `--mib`/`--pseudo_label` 通用包装仍明确拒绝，由方法自行定义目标。

这三种实现保持用户框架的 2D ResUNet 接口及 SGD/学习率调度。它们是明确记录改动的框架适配，不能作为原作者完整训练配置的逐项复现。虚拟输入测试仅验证工程与机制行为，不证明论文数值或实数据性能。

## 作者源码依据

作者源码仅作为阅读参考，通过用户配置的代理获取；没有执行其训练入口。

| 方法 | 固定提交 | 对照文件 |
|---|---|---|
| [S3R](https://github.com/jingyzhang/S3R) | `2bd7808a4557fad7a0328b5e0b563d08d657990e` | `agents/regularization_CISR.py`、`agents/default_CISR.py`、`models/embedding.py`、`utils/compute_sdf.py` |
| [TED](https://github.com/PerceptionComputingLab/TED_DCMIS) | `446f79be52d3e83dea92d7ff352c883154404fbb` | `mp/agents/ted_agent.py`、`mp/eval/losses/losses_segmentation.py`、`args.py` |
| [医学 Class-CL](https://github.com/MrGiovanni/ContinualLearning) | `aa73c28af1969f5d6d714167e54619580f56aa4a` | `model/swinunetr_partial_v3.py`、`train.py`、`pretrained_weights/clip_embedding.py` |

## 必须明确的适配

**S3R** 使用现有单通道 ResUNet，补充 SDF 输出与高层 dropout。形状自编码器保持 16/32/64 通道、64维编码的结构思路，通过自适应池化和插值取消原实现固定384像素的限制。权重名称属于本适配实现，不能直接把原作者的 ACNN state dict 当作兼容权重。

原源码要求预训练 ACNN，但其权重路径属于作者私有环境，仓库也未提供完整预训练流程。因此每个任务必须提供本实现 ACNN checkpoint，或者显式设置 `s3r_shape_epochs > 0`，仅用当前任务训练掩码预训练。后者是新增的可用性适配，其重建 CE、Adam 和额外 epoch 应作为协议与预算单独记录，不能称为作者默认预训练配置。模板故意保留 `s3r_shape_epochs=0` 和空 checkpoint：在条件补齐前真实训练会清楚报错，不使用随机冻结形状先验。`s3r_shape_checkpoint` 支持零起始任务占位符 `{task}`。

S3R 的默认系数对应作者源码：正则10000、语义/距离/形状重要性权重1/0.001/0.05、距离与编码训练损失各1、MC次数8。默认重要性估计遍历全部当前任务训练批次；`s3r_importance_batches>0` 是需要披露的截断近似。作者 dropout 工具设置了不控制 PyTorch dropout 的 `is_training` 字段，此处改为实际切换 Dropout2d，同时保持 BN 为 eval。置信图为常数和空/全前景掩码的除零情况也做了数值保护。

**TED** 保留作者三项蒸馏之和及默认权重0.001。默认监督损失沿用作者 Dice+BCE；可显式选择 `ted_supervised=benchmark` 使用框架传入的监督损失，后者应单独标注为配置变体。SKA 输入是同尺寸的标准高斯噪声，不是给原图添加噪声。

TED 作者 `LossAKT` 中的 GUGF 使用 `[B,H,W]` 与 `[B,1,H,W]` 广播，批量大于1时会形成批间耦合。此处保留其字面行为，并对 batch 1、2、3 与作者公式逐项核对；没有悄悄替换为逐病例加权。`ted_no_gugf` 是明确的消融开关。该行为会受 batch size 影响，正式实验需固定并披露。

**CLIP Class-CL** 将原方法3D动态卷积改为2D，并以 ResUNet bridge 提取图像条件，保留 `256+512 -> 512 -> 153` 的类别控制器与 `8 -> 8 -> 8 -> 1` 动态预测头。由于当前 benchmark 已预定义完整输出数，控制器预分配但只计算已到达类别；未来类别头不参与更新，测试已验证其梯度为空且权重不变。这不是原方法按需扩展容量的原生架构，资源报告必须包含已分配的全部控制器。

旧类别伪掩码由冻结上阶段模型在线生成，使用独立 sigmoid 阈值0.5；当前真实前景标签优先。它替代原工程离线生成并合并伪掩码的管线，增加了在线教师推理成本，且在存在增强时不保证与离线缓存逐项等价。没有保留原3D SwinUNETR骨干或其视觉预训练，因此结果应命名为 `CLIP-CIS / ResUNet-2D adapter`。全部已学类别同时预测，不利用测试任务 ID 选择预测头；提供给原评估器的是背景0与各前景的 logits，等价于最大前景概率超过0.5才胜过背景。

## 后续需要提供的输入

真实医学数据尚未接入；数据根路径仍沿用用户原有 dataset 文件，待拿到数据后按实际位置适配。

Class-CL 还需要本地 CLIP 文本编码文件，格式如下。下方只说明结构，不提供随机编码替代真实语义编码。

```python
{
    "embeddings": tensor_of_shape_C_by_512,
    "class_ids": [1, 2, ..., C],
    "class_names": [name_for_label_1, name_for_label_2, ..., name_for_label_C]
}
```

其中 C 为前景类别总数，MMWHS 对应7。必须按实际 H5 的全局标签顺序生成文本编码，不根据论文叙述猜测 MYO/LV/LA 的编号。编码应来自真实 CLIP 文本模型，类名、提示模板和权重来源一并保留。实现不联网下载模型，不随机补齐缺失类别。合成测试中临时生成的编码仅服务测试，未作为可用于真实实验的权重交付。

## 配置与中性进程入口

三个配置模板位于 `configs/`。其训练轮数、学习率等公共项沿用上传框架，尚未作为新实验协议冻结；不要把配置文件存在视为已授权启动。S3R 的形状先验和 Class-CL 的文本编码仍需补齐。

新增 `run.py` 从环境变量读取配置，将方法参数留在进程内部，不把方法名或项目路径放入 Python 的可见启动参数。应先进入项目目录，再使用相对入口 `run.py`；使用带项目名的绝对入口会违背用户的进程命名要求。

只检查模板语法，不初始化模型或数据：

```sh
RUN_CONFIG="$PWD/configs/ted.json" python run.py --validate-only
```

`--validate-only` 不验证权重文件存在性或数据可用性。后续真实训练前还需要完成数据和科学协议确认，并遵守实验运行与服务器规则；本次没有执行真实训练入口。

## 保存与加载

沿用框架 `T{task}_{epoch}.pth` 命名以及 `save_network/load_network` 调用，但新增方法保存 `medical-cl-v1` 完整结构：模型、教师、重要性与锚点、已学习类别、任务位置、优化器、scheduler、配置。加载会核对方法、损失配置和类别名称映射。只保存 `model.net.state_dict()` 不足以恢复持续学习状态，应使用对应方法的保存接口。

完整 checkpoint 包含恢复算法状态所需的信息；没有新增自动恢复中途数据游标的功能，也不保证从某个 epoch 中间逐批重放相同随机序列。原训练 CLI 的一般续跑行为没有在此次扩展，不能把 checkpoint 读写测试说成已验证任意中断恢复。

## 必要的框架接入修复

- 方法注册改为按需导入，避免未选方法的 `quadprog` 等依赖阻塞新方法；发现路径不再依赖当前工作目录。
- TensorBoard 的可选导入推迟到实际使用时。
- 新方法的参数统计只计算部署模型，避免教师使参数量翻倍；S3R 的 ACNN、SDF、参数锚点和重要性属于训练辅助状态，其存储与训练成本仍需单独报告，不能称为没有额外存储。
- CSV 写入不修改运行中的 args，自动创建目录；新方法按运行输出目录分别保存，避免不同参数列混入公共 CSV。
- 原训练日志读取未赋值 `rmac` 会失败，现已修复。`new_task_dice` 保存未归一化对角线均值；归一化 RMA 没有匹配参照就留空。
- 新方法的测试流程不使用原脚本硬编码的历史 SGD 表。可用 `rma_reference` 提供 JSON，字段为 `dataset`、`seed`、`order`、`results`。验证元数据、矩阵形状与正对角线；独立训练参照的骨干、预算、划分等科学匹配仍须另外核实。
- 修复原 test-only 流程数据计数始终为0的问题，沿用原 train 流程的计数口径，避免存储指标除零。

没有重定义原框架的科学评估协议。尤其是原 Class-CL 评估器仍会按测试任务屏蔽部分后续类别；方法自身的无任务身份推理不等于旧评估结果已经是全部类别竞争。原 replay/SSS 的指标定义、独立训练参照和逐病例聚合也需在真实实验前另行审计。本次未更改这些定义或复算论文结果。

## 合成测试

在安装框架依赖的环境中运行：

```sh
python -m pytest -q
```

测试使用当前 ResUNet 的小配置32像素输入进行多任务和完整训练/测试循环，同时覆盖标准 `mid` ResUNet32 的256像素前向及反向。包括教师权重与BN冻结、学生更新、S3R重要性累积和正则生效、ACNN加载与训练、空/全掩码、TED公式对照、CLIP分组动态卷积数值对照、未来头不更新、真实标签优先、错误setting/权重映射拒绝、完整检查点恢复、RMA参照匹配、配置模板与CSV输出。

测试数据全部由本地张量临时构造，检查点和 CSV 写到测试临时目录；产生的分数没有科研意义。环境版本及最终执行结果见 `IMPLEMENTATION_STATUS.json`。

2026-10-27 S3R源码优化器修正v2：Adam lr5e-4、wd1e-4，逐epoch开始按0.98衰减、跨task保留Adam矩；避免框架二次StepLR衰减。BCE前检查非有限logits/probs。实际协议采用50epochs当前task训练mask ACNN适配，代码模板0仍要求提供prior或显式预训练，不使用随机冻结prior。旧SGD .008失败不混入新协议五seed，详见experiments/20261026_five_seed/S3R_SOURCE_OPTIMIZER_V2_ZH.md。

2026-10-27 数值审计v3：S3R/TED旧概率BCE在真实T0 checkpoint的错误饱和背景处出现零纠正梯度；改用数学等价的稳定logits求值。TED另纠正原来误继承的通用SGD配置，使用Adam base2e-4/inc1e-4、每域新optimizer/epoch gamma.99。旧TED s1/S3R s1,s2结果失效另存，不作有效复现负结果；原时钟不重置。162项CPU测试及两方法各两任务/两步真实train资格通过，尚未证明完整性能。详见experiments/20261026_five_seed/DOMAIN_COLLAPSE_AUDIT_ZH.md。
