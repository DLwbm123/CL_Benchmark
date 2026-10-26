# Class-LDKA 梯度保护适配 v2

旧seed1在第一task epoch122附近loss从约0.02/0.99升至78.79并随后非有限退出，没有完整任务checkpoint。不是SKD跨任务错误；第一任务主要涉及PML/MAD。不能唯一确定触发源，也不能将Organ的跨任务因果解释直接套用到这里。

用户已授权问题自行修复验证继续。保持PML/SKD/MAD公式、lr .001/.0001、SGD momentum.9/wd1e-5、150epochs/task、batch8及原数据不变，为Class的所有任务显式启用global gradient norm cap=1。这是ResUNet适配的数值保护，不是论文原生超参数，须标为LDKA/ResUNet + clip1；五seed统一，不能混入旧无保护协议。Organ的保护仍仅在增量task启用，不改变其v2。

资格使用MMWHS第一task一个真实train batch8，随机初始化并固定为epoch122对应progress，比较无保护/有保护各最多5更新，检查finite、保护计数及未来classifier行不变。没有失败时刻权重，所以这不是对实际已训练122epochs状态的完整复现，不能声称已证明长期稳定。资格权重不复用，成本单列。旧失败日志及原时钟保留，不做原122epoch轨迹重放或测试集调参。

stage checkpoint保存guard_stats步数、裁剪次数及最大裁剪前norm。旧失败目录原地归档，按原seed重排，复用原started_at1793025587.5574715、deadline1793111987.5574715，不重置24h。protocols/ldka_s1..s5.json仅可覆盖ldka_grad_clip，其他身份/学习率/预算不可覆盖。仍GPU1/5六并发、总新增根40GB预停<50GB，正常运行不重启。
