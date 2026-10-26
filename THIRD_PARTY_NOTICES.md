# 新增自然图像方法的来源与许可说明

本地适配实现依据下列论文公式和固定版本作者代码编写。保留作者署名与来源；改动包括替换为用户的2D ResUNet、共享框架接口与完整 checkpoint、显式标签空间控制，以及文档列出的数值和设备适配。作者不为本移植提供背书。

- **EWF**：Jia-Wen Xiao、Chang-Bin Zhang、Jiekang Feng、Xialei Liu、Joost van de Weijer、Ming-Ming Cheng，CVPR 2023。参考 [schuy1er/EWF_official](https://github.com/schuy1er/EWF_official)，提交 `560382d5ea6d79b6d49c365988cc5118129dd815`，特别是 `run.py` 的默认 endpoint fusion、MiB训练设置及旧分类器冻结。作者代码附 MIT License，Copyright (c) 2023 Jiawen-Schuy1er-Xiao；原文保存在 `third_party/EWF_LICENSE.txt`。
- **LGKD**：Ze Yang、Ruibo Li、Evan Ling、Chi Zhang、Yiming Wang、Dezhao Huang、Keng Teck Ma、Minhoe Hur、Guosheng Lin，ICCV 2023。参考 [Ze-Yang/LGKD](https://github.com/Ze-Yang/LGKD)，提交 `78262c22043fe9d56261d1906a04c016b82930b5`，特别是 `utils/loss.py` 的 `LabelGuidedKnowledgeDistillationLoss`、论文的LGKD＋MiB配置和权重实验。作者仓库标注 Creative Commons Attribution-NonCommercial 4.0，许可说明保存在 `third_party/LGKD_LICENSE.txt`。
- **NeST**：Zhengyuan Xie、Haiquan Lu、Jia-wen Xiao、Enguang Wang、Le Zhang、Xialei Liu，ECCV 2024。参考 [zhengyuan-xie/ECCV24_NeST](https://github.com/zhengyuan-xie/ECCV24_NeST)，提交 `f96185be8a46f7f1812d750e00fe2fd86b7fd2b8`，特别是 `train.py` 的 `select/warm_up`、`segmentation_module.py` 的权重变换与编译、MiB＋NeST脚本。作者README标注 [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)，对应原始声明另存于 `third_party/NeST_LICENSE_NOTICE.txt`。

MiB公共损失与平衡初始化的研究来源为 Fabio Cermelli 等，Modeling the Background for Incremental Learning in Semantic Segmentation，CVPR 2020；用户原工程已有其实现。此次仅为三个新组合增加可独立检查的接口，不宣称创作这些方法，也不授予超出上游许可的权利。用户上传框架中原有的版权说明保持不动。

- **CIT**：Jinchao Ge、Bowen Zhang、Akide Liu、Minh Hieu Phan、Qi Chen、Yangyang Shu、Yang Zhao，Pattern Recognition 2025（预印本2024）。参考 [jinchaogjc/cit](https://github.com/jinchaogjc/cit)，提交 `bc20448317262f71cc9ccb69cc057b602bb8d869`，尤其 `mmseg_custom/decode_heads/dv3_head.py`、`mmseg_custom/segmenters/encoder_decoder_cl.py`、`mmeng_custom/loops.py`、`mmseg_custom/loss/atm_loss.py` 与 `criterion.py`。此次独立编写ResUNet适配实现，没有打包整个作者仓库。在所检查的固定版本目录树中未发现顶层LICENSE，不能据此推定MIT或其他授权；本文档不授予作者源码的额外使用或再分发权利。具体适配见 `README_natural_ciss.md` 的CIT段落。

- **ReBac**：Guangyu Gao、Anqi Zhang、Jianbo Jiao、Fangkunhan Liu、Chi Harold Liu、Yunchao Wei，IJCV 2026。参考 [ANDYZAQ/ReBac](https://github.com/ANDYZAQ/ReBac)，提交 `dae1b48d461ba62bab99414d57d6474f5aa3d80b`；主要依据 `network/_deeplab.py` 的类感知背景融合、`trainer/trainer.py` 的损失/初始化和 `utils/parser.py` 的配置。独立编写ResUNet适配，没有打包整个作者仓库。所检查源码中的SSUL衍生文件含NAVER MIT署名，但这不代表整个ReBac新增实现统一采用MIT；本交付不推定或授予作者代码的额外再分发许可。
- **LDKA**：Xinyue Zhang、Xu Zou、Wanjia Luo、Yanjie Wang、Jiahuan Zhou、Sheng Zhong、Luxin Yan，ICML 2026，PMLR 306。独立依据[正式论文](https://proceedings.mlr.press/v306/zhang26jo.html)正文公式及补充材料Algorithms 1/2重构，没有取得或复制作者实现。ResUNet分类器行、decoder映射及求导维度选择见README，不能称为原作者源码。
- **PCDD**：Shengqian Zhu、Chengrong Yu、Qiang Wang、Ying Song、Guangjun Li、Jiafei Wu、Xiaogang Xu、Zhang Yi、Junjie Hu，AAAI 2026。独立依据[正式论文](https://ojs.aaai.org/index.php/AAAI/article/download/38406/42368)与[扩展预印本v1](https://arxiv.org/abs/2511.07749v1)重构。作者仓库[shengqianzhu/PCDD](https://github.com/shengqianzhu/PCDD)检查版本为 `ad9f5094da97ba5e002be0d467e27171f8d907c6`，尚无方法源码。本实现的公式歧义解释、共同旧标签空间CRCD和2D/BN适配见README；不授予论文或作者资源的额外权利。

本轮 `ewf_organ`、`ldka_organ`、`cit_organ`、`clip_cis_organ` 继续依据本文及 `README_medical_methods.md` 中对应EWF、LDKA、CIT和MICCAI 2023动态CLIP头的论文/固定版本来源进行适配。没有新增第三方实现或放宽上游许可。这四个注册名均为用户benchmark协议下的Organ-CL适配，独立背景、任务路由及目标变化详见 `README_natural_ciss.md`；不归因于原作者已经验证的Organ-CL实验。
