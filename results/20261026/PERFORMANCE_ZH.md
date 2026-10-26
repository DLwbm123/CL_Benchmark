# perf-v1 实现、验证与替换记录

已实现三处加速：

1. H5按需只读RAM缓存。仅当前训练split或实际评估split在迭代前加载，保留原H/W/N索引，但将切片存为连续内存；原dtype、OpenCV resize、标签转整数及Class标签偏移均不改。父进程准备后原有fork workers共享数组；worker数、sampler、batch及随机数消耗不变。没有把未来训练数据预先加载给模型。
2. 各参数的isfinite检查仍全部执行，但合并为一次主机判断，保留loss/gradient非有限值保护。
3. 纯评估关闭autograd，保留预测、指标、模型模式切换及全部评估次数。

运行环境 `CL_H5_RAM_CACHE=1`、`CL_H5_RAM_CACHE_MB=4096`、`CL_EXECUTION_VARIANT=perf-v1`。4096MiB约束缓存注册表，不是整个进程RSS上限；活跃dataset/worker持有的数组可能继续驻留。缓存仅在RAM，不增加患者数据磁盘副本。没有启用AMP、修改TF32策略、改变batch或删减ASD。

验证：142项CPU测试通过，包含原训练/测试循环、缓存前后tensor/resize/标签/RNG一致、SGD一步更新一致及非有限梯度拒绝。Class的非原尺寸int64 resize在原OpenCV路径本身不支持，本轮不扩展该行为；正式输入保持256×256。

真实训练split各32个样本的单进程4线程读取/resize/转tensor微测：

| 数据 | 原读取32次 | RAM读取32次 | 一次加载 |
|---|---:|---:|---:|
| BIDMC | 2.800s | 0.0084s | 0.631s |
| UtahI | 6.068s | 0.0078s | 1.702s |
| MMWHS第一任务 | 9.420s | 0.0067s | 2.772s |

每种4个代表样本逐tensor比较相同。读取微测倍数不能当完整训练加速倍数。

GPU上的mid ResUNet32、batch8、256×256合成TED任务，中位训练步0.2306s→0.1052s（三组各5步、每种另2步预热）。默认CUDA下第一次逐位断言失败，保留INITIAL_PROFILE_FAILURE.json；对照发现原版自身重复与新老比较最大差均2.98e-8。单独启用确定性算子的验证中一步后全部模型tensor逐位相同。确定性选项仅用于诊断，没有改变正式实验backend策略。工程GPU微测共41次synthetic更新（首个失败对照2步、后续39步），单列成本，不作为医学性能证据。

用户明确要求用新代码替换当前实验。旧13项及其日志、状态和已有TED T0 checkpoint原地重命名归档到远程 `attempts/before_perf_v1/`；旧代码模块保存到 `previous_code/before_perf_v1/`。旧checkpoint没有RNG/采样器状态，不能保证无损接续，因此这次按原seed从头重启；已丢弃更新/时长必须披露。所有原started_at/deadline保留，24h不重置。4个旧控制器退出，旧训练全部退出；新的唯一dispatch PID475128负责70项队列，GPU1/5各6并发，GPU3禁止，原GPU7也排回1/5。启动后12项已进入优化，58项排队；不是70次完成声明。

PERF_SWITCH_RECEIPT.json为切换证据；PERFORMANCE_PROFILE.json为微测；实际运行仍需按日志、checkpoint和最终CSV验收。后续心跳已原位更新，不能恢复旧控制器、旧attempt或将旧阶段结果混入新的五seed汇总。

切换后第一任务短时完整epoch对照：TED 29.0→9.33s，S3R 30.8→11.25s，Organ-EWF 141.9→25.69s，Organ-CLIP 153.0→26.09s，Organ-LDKA 148.5→28.78s，Organ-CIT 135.4→33.0s。Class部分受所有任务进入GPU计算后的争用影响：CIT49.2→68.8s、LDKA46.9→64.0s、EWF47.0→52.7s，NeST/CLIP/LGKD约持平。不能宣称每种方法都加速，须以总完成吞吐和各方法完整计时判断；当前为不同共享负载下的启动阶段观测。详见PERF_LIVE_EPOCH_COMPARISON.json。
