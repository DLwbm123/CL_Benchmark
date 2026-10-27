> **数值审计更正：本旧协议结果已失效，数值保留供审计，不纳入v3方法比较。参见[审计报告](DOMAIN_COLLAPSE_AUDIT_ZH.md)。**

# S3R seed1：已完成的负结果

仅1/5 seeds、1/70独立运行完成，不能计算五seed均值/标准差。协议s3r-source-optimizer-v2，ResUNet 2D适配、每task150epochs、batch8、当前mask形状预训练50epochs。六阶段完整checkpoint的current_task依次1..6、task_open=false，张量有限，训练/测试退出码均0。

独立test宏平均Dice 0.4957892590、宏平均precision 0.9916671053、宏平均recall 0.5000000002、原ASD定义0.0166657895，background包含在平均内。验证Dice0.4950952596。各阶段整行结果完全相同、BWT0，不能解读为成功防遗忘；代表性train切片预测全背景，支持模型退化判断，未证明全部test像素均为背景。normalized RMA无匹配参照，留空。ASD空预测/背景口径限制沿用原协议，不将其小值视为好的前景分割。

原首次起点至测试退出约6.29小时（含旧尝试、资格/等待），原时钟未重置。旧性能切换中断和旧SGD科学失败均保留；未因低分重新训练或调参。其余seed继续冻结协议。正式负结果与过程失败分列。
