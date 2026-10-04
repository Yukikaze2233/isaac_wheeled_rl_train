# V6 完整倒地 FSM 物理前缀最小探针计划

日期：2026-10-05。**状态：`proposal_not_implemented` / `not_run`。** 本文与 [physical_prefix_design.json](physical_prefix_design.json) 只保存设计；尚未新增入口、冻结探针合同、运行预检或 Isaac/GPU，也未读取远端 PT、修改 PPO、RMCS、原合同或当前 v14 训练。消费更新为 0，`qualification=false`。

本探针回答一个有限问题：同 V6 资产从真实倒地初态经原生 FSM 连续物理执行，是否能到达实际 BLEND，继而由指定 actor 接稳、解锁并完成后续静站。它将生成动态入口证据，而不把出生几何、软件 parity、V5 的成功轨迹或结束回合条件比例当成完整 V6 自起资格。

## 1. 先固定来源与兼容 actor

| 来源 | SHA-256 |
| --- | --- |
| `contracts/v6_task_campaign_recovery_worker_v8.json` | `f6aa77c6808c9746dc8dee431e8b3331b6b0f883db701335e57069e7df738ee8` |
| `contracts/v6_task_campaign_recovery_expectations_v8.json` | `e60563a1a7f687df99cf2da5052a3bf3751c68689ed5db3ee2b18146afe37abd` |
| V6 asset manifest | `875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba` |
| control math | `8073540d1cec0b32c6216699863b3ef53064a2d385b7110abe291d9da0e9b1ac` |
| `contracts/v6_recovery_profiles_v2.json` | `007363fdfe35804b01ad616a1453bf163bf70c1bb7dd59b6a8f743b26c0b4441` |
| `contracts/v6_height_lookup_v1.json` | `c3e215187828fae328e3df3af1728897feb6695616ec8b9bdd4322433ea00282` |

首个 baseline 使用同 V6 平地终点完整 PT：

`/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/flat/attempt_007/train/model_final.pt`

已记录 PT SHA 为 `7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f`，其 source contract SHA 为 `666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7`。来源为既有 `reports/v6_speed_quality_regression_20261005/comparison_key_metrics.json`；本次没有重新读取或认证远端文件。未来运行前必须重新核对 PT 文件、`infos` 与 source contract，不能把这个设计回执当作运行认证。

逐项核对 asset/control SHA、`physics_dt=.005`、`policy_dt=.02`、35D actor、81D critic、6D action、观测来源、动作顺序、网络架构和 `v5_control`。使用完整学习 checkpoint 与原生严格 actor factory，只做确定性推理；不迁移 optimizer 或预算，不调用 `learn/update/compute_returns`。执行前后 actor 与 PT SHA 必须相等。

## 2. 新入口必须直接支持 V8 合同

建议的新入口名为 `scripts/probe_recovery_physical_prefix.py`，**目前未创建，不能执行**。其输入应是明确的 V8 worker contract、完整 PT 与输出目录。

| 现有入口／helper | 可复用部分 | 必须适配的限制 |
| --- | --- | --- |
| `scripts/probe_v6_recovery_runtime.py` | `project_evaluation`、`project_scenes`、独立 full-role 设计投影、source snapshot/change checks | 旧默认 recovery plan 的物化路径和 20k／1kHz／20 子步 capacity 分支不适用于本探针 |
| `scripts/probe_recovery_takeover.py` | PT provenance/ABI 认证、`load_actor`、`IntervalAudit`、`outcome` 终止前捕获、真实 reset readback 模式 | 现 `authenticate` 要求 unchanged train role 的 4096 人口；不能仅传 `--num-envs 4`。其 `reset_rows` 的 train curriculum 假设也不适用于 fixed case |
| `FixedCaseEnv` | 冻结 case、seeded replica 布局、`reset_suite` | 初始化含父类与子类的多次合法写入，还含 recovery root z 的冻结 0–2mm seeded jitter |
| `RecoveryTraining` | 原生 `advance_reference → torque → effective_action_history → outcome` | 只观测执行结果，不替换控制输出、强制 phase 或把真值接触馈入部署观测器 |

先认证 V8 worker 与其原始 full role，再从**已认证的独立设计表**投影四个 case。runtime 与期待值分别投影；允许变化仅为 case/coverage 名单、scene 选择与份额、replicas、invocation 人口。物理、FSM、原 case 内容、门槛、模块启用状态保持不变。生成新的工程设计与合同及 SHA 后，按工程 full role 预检。不能从 runtime 自己反推期待值，也不能调用旧 1kHz plan 作为源语义。

## 3. 四个现有倒地 case，首轮每个一个 cohort

| 已冻结 case | profile | 真实初态覆盖 |
| --- | --- | --- |
| `recovery_prone_belly_legs_above_stand` | `prone_belly_legs_above` | 腹部落地、腿在上方 |
| `recovery_prone_back_legs_above_stand` | `prone_back_legs_above` | 背部落地、倒置 |
| `recovery_left_side_90_stand` | `left_side_90` | 左侧倾 |
| `recovery_right_side_90_stand` | `right_side_90` | 右侧倾 |

四例均为已有 flat case，命令 `[0,0,0.305]`，profile 自带 1s 零主动电机力矩释放；源 profile 明确 `geometry_valid=true`、`dynamic_recovery_validated=false`。第一轮 4 个环境各执行首个 episode，源 full-role seed `290929`，保留各 case **20s** episode 上限：最多 1000 策略步／4000 物理步。复现时可另冻结每 case 4 replicas，共 16 cohort。初始化失败应使探针无效，不能缩小分母。

四例不是所有 FSM 路线的覆盖证明。如需显式观察正反 ORBIT 分支，另增加已有 `recovery_front_down_90_stand`／`recovery_back_down_90_stand`；不强制当前四例走它们本来不会选择的路线。

## 4. 只有出生事务可以写机械状态

为每 lane 绑定 `(case, replica, seed, birth_id)`。出生初始化事务允许环境创建／`reset_suite` 中父类和子类的 native root/joint 写入，记录所有调用，核对 q18/profile、root/origin/seeded jitter、速度、闭合与有限性。事务在首个活 cohort 物理步之前关闭。

**出生事务结束后，活 cohort 不允许任何 root/joint state write。** 对 `write_root_link_pose_to_sim_index`、`write_root_com_velocity_to_sim_index`、`write_joint_position_to_sim_index`、`write_joint_velocity_to_sim_index` 观察调用及 lane。违反即标记物理前缀证据无效；不能通过 reset 修复后继续计成功。原生终止后的 autoreset 只允许发生在旧 cohort 终止状态已捕获并关闭之后，新 episode 不进入本轮统计。

完整链应是：一次出生几何注入 → 1s 零主动矩的真实重力／气簧／接触运动 → native FSM/PD 力矩 → actual BLEND → pure RL hold → unlock → followup。脚本插值只生成目标角，不写实际关节或 root；不固定底盘、不加牵引外力、不直接注入 BLEND 末态。释放结束的原生 FSM/编码器激活 reset 是控制历史操作，要记录，但不能顺带清空实际物理、驱动响应、延迟或动作历史。

`FixedCaseEnv` 的 expected root 必须计入冻结 seeded z jitter，不能照搬 train-role `reset_rows` 对 bank 原 root 的 `2e-5` 比较。q18 按实际 joint_names 核对 bank；两条 spring slide 是 prismatic，q/dq 单位分别为 m、m/s，其余转轴为 rad、rad/s，不能把全向量标成弧度。

## 5. 捕获真实 BLEND 入口和末端

连续保存选中 cohort 的 200Hz 物理流与 50Hz 实际 actor 观测／动作；事件快照使用事件当时的 clone。优先缓冲后批量落盘，不引入每子步的大量打印。

每事件带 policy tick、physical tick/substep、reference tick、物理时间、前后 phase，以及**本次实际施力使用的** blend fraction。至少保存：

1. 出生事务关闭、释放结束、每次 phase/route/reroute 转移。
2. `PREPARE→BLEND` 更新后、施力前的真实状态。
3. 第一个混合子步经过 transport/response 并完成物理步后的状态。
4. 最后一个 fraction<1 的混合子步，以及第一个纯 RL 子步的物理末状态。
5. 传感器接稳、真值接稳、首次解锁、followup 成功和终止前状态。

核对真实 40 个参考更新／200ms 混合及每个子步控制归属。仅“出现过 BLEND”或 phase 枚举进入 RL 都不等于完整接管。脚本失败而未到达 BLEND，应写 `never_reached_blend` 与前序失败证据，不能合成入口快照。

snapshot 字段详列在 JSON，核心包含：

- **机械**：named q18/dq18、root pose/线角速度、CONTROL q6/dq6、轮位置、闭合 gap、气簧压缩、轮／车体真值接触。
- **观测与目标历史**：实际 raw/continuous/delayed feedback、编码器 winding/last_raw、gyro/gravity、35D/81D、actor/effective/previous/before_previous action、观测 history、decoded targets、命令及加速度。
- **驱动链**：script requested/capped、RL requested、混合后、transport 后、USB 后、response 后、final applied torque、即时包络、response state/alpha/direct axes 与参数 SHA。
- **延迟**：signal-delay 的实际 enabled/profile/lag/feedback/strength，两个 native buffer 的 buffer/pointer/num_pushes/pending resets；command-transport pending/valid/generated/held/age/tick/loss/burst/CAN phase 与 RNG；USB 实际启用状态及存在时的 retained 状态。
- **FSM 与观测器**：phase/route/targets/timers/dwell/reroute/failure、support/body-clear、IMU previous velocity、轮 probe 的 evidence/age/bad samples/previous velocity/gyro/pulse/quiet/lost time、true-stable/followup/premature-release 历史。

**快照不声称可精确复原 PhysX 内部状态。** q/root 和驱动历史没有完整序列化接触、constraint/solver、warm-start 等隐状态。当前参考必须是同环境、同前缀不中断执行；事件快照是诊断与动态入口产物。以后若要 restore、降人口重放或末态采样，必须单独验证，不因拥有 pose/RNG 就宣称逐位一致。

## 6. 每道门用同一出生分母

主表对每 case／所有 cohort 均使用固定 `N=4` 或 `N=16`：脚本开始、各 native 路线／阶段、CAPTURE、PREPARE、BLEND 完成、pure RL、hold、unlock、followup 的到达数/N、首次到达时间、失败阶段。

分别记录 FSM 的 valid/support/body_clear/tilt/gyro/height/wheel_speed/leg_speed 八门，以及目标 remaining/error、联合通过、连续 dwell、break count、首个 blocker/最差余量。独立门曾通过不能代替同一时刻联合通过。

- PREPARE：八门与目标条件、连续 0.1s ready；保持原门槛。
- BLEND：实际 200ms 混合与完整子步控制归属。
- RL hold：原生传感器联合稳定连续 1s。
- 物理 hold：`outcome` 的真实接触／非期望接触／平台／姿态／gyro／高度／速度条件及连续 1s。
- unlock：传感器解锁与真值先达标分别记录，检查 `premature_release`；真值只用于诊断，不回馈 FSM。
- followup：保持原生连续 1s 完成条件；静站速度和速度/yaw误差带原阈值与样本数。

最终给每个出生 cohort 一个互斥归因分区：success、failure、timeout、boundary truncation、未完成/right-censored，按冻结 precedence 处理；五类之和必须为 N。原始 reason flag 可重叠，应另外保留。`success/ended`、`success/handed_over` 只作为标注条件分母的次级指标，不能名为整链完成率；未到达或未观测事件不算通过。不要用 ended episodes 或反复 autoreset 把难例从分母里洗掉。

## 7. full-role 延迟范围与后续独立条件

V8 full-role 明确冻结：dynamics randomization=true、contact domain=true、command transport=true、signal perturbations=false、**signal delay=false**。第一轮必须保留并逐项运行核对；USB 状态从冻结 effective config/startup 实读，不能由 command transport 是否开启推断。

所以首轮即使保存了 disabled/absent buffer 状态，也**没有验证训练整帧 signal-delay 随机化**。要验证该条件，另冻结 engineering-delay role，明确 profile/schedule/强度、初始/RNG/history 语义与对应独立期待值，保持同资产/ABI/200Hz/50Hz/FSM/判据。不能运行时热开后继续称原 full-role 身份或通过资格。

未来实现应输出独立 source/design/PT SHA 回执、effective engineering contract、cohort ledger、events、物理流、事件快照与 summary。图表若需要，接入 Kaiser TensorBoard；本机不启动服务。任何实际执行和结果须另写运行回执；本文与 JSON 始终是**尚未实现／尚未运行的计划**。
