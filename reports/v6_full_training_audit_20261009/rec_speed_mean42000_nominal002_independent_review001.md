# nominal002 独立只读审查

本回放真实完成 REC41393 → speed42000 actor597 两端点比较，exit0、88.480345秒、0 PPO；模式为 interactive_unscored，qualification=null。不能据此称新模型更强、永久遗忘已证明或实车就绪。先前口头结果保留，本报告增加了脚本前缀差异与比较设计限制。

实际输入为 REC checkpoint `e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967` 和 speed checkpoint `7d7d53ad86898614da3dd74794c37fd3bea0c42976c1426e449e40fd5eca38be`。运行是CPU、7场景×4副本、每端点500策略步，50Hz策略／200Hz物理反馈PD，SIDE1.5rad/s，已开始BLEND完整执行。

三档静站均按原生首回合全部有效帧（warmup=0）统计。复核每组2000帧＝4副本×500帧，并按 frames 重算三档合并均值。每组原生均值按 SUM/frames 定义；本审查从报告均值×frames重构分子，不冒称重新读取未保存的全部50Hz样本。稀疏trace仅61／59行、52个共同时间点，不能代替完整帧数据。

| 场景 | 帧数／副本 | 世界水平平均速度 cm/s（REC → speed） | 高度MAE mm（REC → speed） |
| --- | --- | --- | --- |
| stand_230mm | 2000／4×500 | 2.277471 → 2.410679 | 2.090585 → 3.419466 |
| stand_305mm | 2000／4×500 | 2.186211 → 2.474413 | 1.315047 → 4.601556 |
| stand_430mm | 2000／4×500 | 2.371535 → 2.526329 | 5.867684 → 26.146990 |

三档合计6000帧，按帧数加权速度为 2.278406 → 2.470474cm/s，高度MAE为 3.091105 → 11.389337mm。漂移最大值有不同方向变化，不能据局部改进给整体更强信用。

四倒地场景分母是每组四个已结束的原始出生，失败仍计分母。完整链计数从 native ended ledger 的 true_success/failed 重算，并与每副本终止原因、physical_full_chain_completed诊断及原生success计数交叉核对；本轮右删失0。

| 场景 | 原生完整链（REC → speed） | script_timeout数量（REC → speed） |
| --- | --- | --- |
| recovery_left_side_90_stand | 2/4 → 0/4 | 2 → 4 |
| recovery_right_side_90_stand | 1/4 → 1/4 | 3 → 3 |
| recovery_prone_belly_legs_above_stand | 4/4 → 4/4 | 0 → 0 |
| recovery_prone_back_legs_above_stand | 4/4 → 4/4 | 0 → 0 |

四组描述性总计11/16 → 9/16；失败5 → 7均为script_timeout。全部场景闭链峰为 0.680881 → 0.654348mm，低于本次3mm门，不能转化为机械或实车资格。

nominal设计不进入命令定义的专项停车窗口：quiet target=null、全部quiet frames=0、first_applied_zero=null、完整10秒quiet窗口0个。因此源报告的quiet零值表示无观测，不能解读成完美停车。其它运动／停车协议不在本审查范围。

两端点所有12个已记录reset fingerprint字段逐项一致；actor出生观测SHA都是`1dda77036b7f82823c5f29eaf91697007f6cfbc2841d49d46759242a84d31830`。冻结驱动重置NumPy、Torch、eval/env RNG以及独立command_transport RNG和tick；没有证据证明存在未重置的shared RNG缺陷。原生reset也明确重置根／关节状态、速度、驱动和传输缓冲、动作历史、任务／恢复FSM／观测历史。不过指纹没有覆盖完整隐藏状态、每tick随机流和PhysX接触／求解器内部缓存；两个候选固定按REC→speed顺序复用同一Kit世界，未做逆序或独立世界对照。

实际稀疏raw显示16个倒地副本都在双方仍归script且早于各自首次BLEND时出现状态差异：belly首次观测1.22／1.42秒，back3.62／3.82秒，左右侧均4.82秒。首个已保存差异时间不是精确首个物理差异时间。差异在actor接管之前已出现，原因未确定，因此11/16→9/16不能隔离归因为actor遗忘或证明某端点更强。

本次只有单seed290929与四固定副本，没有广泛延迟、负载、摩擦应力或硬件覆盖；轮侧响应已认证的范围不能扩展成四腿真实曲线资格。审查只保存两个新文件，不追加回放、不改driver／native源码／训练合同／训练过程。

原始native、执行回执、两candidate、两trace、投影／预备文件、log、intent／process和event文件，以及实际冻结driver/native源码均按读前／后stat与SHA固定在 [独立审查JSON](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/rec_speed_mean42000_nominal002_independent_review001.json)。该JSON严格禁止NaN/Infinity；原始文件SHA在审查结束再次确认。

原始结果：[native_replay.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/rec_speed_mean42000_nominal002/native_replay.json)（SHA `90eacb6e0d9c18ca5c4fce21cd9cd23f59087f181ce3bd544df3e41f227fac2e`），[实际执行回执](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/rec_speed_mean42000_nominal002.execution.json)（SHA `7a22026158151e80210c220e7c31aff05753cbb4a533923be66dc028df22006b`）。目录内execution.json并不存在，真实回执位于上一级。
