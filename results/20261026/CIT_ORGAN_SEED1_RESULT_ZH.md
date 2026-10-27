# CIT Organ-CL seed1结果

训练/test退出码均0，四个任务checkpoint经核验current_task1..4、task_open=false、张量有限，保存优化器配置为AdamW。原首次实际运行至退出13.694小时，包含性能代码切换和旧SGD失败的成本，原24小时时钟未重置。旧轨迹单独归档，不混入当前v2结果。

Test任务条件宏Dice0.7227927685，Dice BWT -0.1853914576；宏precision0.8301388153、recall0.7050181788，原ASD0.0224225760。BWT反映存在遗忘。以上指标含背景，Organ任务路由依赖已知task，沿用原ASD空预测/背景口径，不当作完全任务无关分割。normalized RMA因无匹配独立参照留空。

这是CIT/ResUNet的Organ适配、仅1/5seed，不能声称作者原生实验复现、完整五seed结论或跨方法显著优势。AdamW采用作者CNN配置对应base2e-5/head10x/wd.01，仍保留benchmark epoch衰减，未复现作者完整warmup/PolyLR。其余seed保持统一冻结协议，不以低分改变配置。

累计11/70项已审核，当前12项运行、47项排队。S3R完成2seed，其他已完成组合各seed1，尚无任何组合的完整五seed统计。
