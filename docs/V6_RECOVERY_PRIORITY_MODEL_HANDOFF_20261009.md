# 恢复优先模型与部署交接：2026-10-09

最新新增导出为 **speed42000 / 本轮高速 actor597**，文件位置如下。它继承真实恢复父端点 REC41393；42000 是累计消费时钟，实际学习谱系为28253，不能称为42000次高速学习。

| 文件 | 本机位置 |
|---|---|
| 最新 ONNX | [policy.onnx](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_speed_periodic_42000_actor597_20261009/policy.onnx) |
| 对应35D输入/6D动作及控制接口 | [interface.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_speed_periodic_42000_actor597_20261009/deployment/interface.json) |
| 完整模型与合同目录 | `/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_speed_periodic_42000_actor597_20261009/` |
| 本机回放操作 | [CHASSIS_GUI_SUITE.md](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/docs/CHASSIS_GUI_SUITE.md)；`--onnx`显式指定上面的新模型 |

新 ONNX SHA-256：`4ddf8d4245f344645af9c45d732af11393b6c263bf02f6c37462cd04e6ecc5a3`。官方导出及35组ABI向量通过，最大绝对误差`1.0728836e-6`；输入35D、输出6D确定性均值，50Hz策略/200Hz仿真反馈PD、腿160/2.5。必须使用新目录自己的合同、接口、ABI向量和SHA256SUMS，不能替换旧包内ONNX后沿用旧模型身份。

已完成本机两端点的名义条件原生短回放和两个全新进程的名义对照，但尚无完整固定评测或实车通过结论；`qualification=null`、`hardware_deployment_ready=false`。不能仅凭导出或训练时长称其为全功能最强模型。导出证据见[回执](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/source029_speed42000_export_execution001.json)，最新独立进程执行见[完成记录](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/rec_speed_mean42000_independent_world004/completion.json)。

最新独立进程名义回放为7场景×4固定副本，两端点各构造一个全新的Kit/FixedCaseEnv、读取同一份固定SHA的projection，初始状态指纹相同，复用原生FSM/物理/终止条件，0 PPO。结果如下；每场景4副本不足以建立整体优劣或唯一退步原因。

| 名义回放项目 | 父端点 REC41393 | speed42000 |
|---|---:|---:|
| 230/305/430mm静站平面均速 | 2.28/2.19/2.37cm/s | 2.41/2.47/2.53cm/s |
| 230/305/430mm高度MAE | 2.09/1.32/5.87mm | 3.42/4.60/26.15mm |
| 左/右侧倒地完整链 | 2/4、1/4 | 2/4、1/4 |
| 俯卧/仰卧完整链 | 4/4、4/4 | 4/4、4/4 |

最新独立名义对照未证明新模型更强；430mm高度误差仍明显较差，但倒地完整链已同为11/16。早先同一环境REC→speed复用所得左侧2/4→0/4、总11/16→9/16，在独立进程下没有复现；当时脚本/BLEND前已有物理前缀差异，不能据此声称actor遗忘。新对照的独立审查确认16倒地副本共532个首BLEND前采样的五项物理字段逐值相同、首BLEND时间相同；这仍不覆盖所有子步或求解器内部状态。详见[实际审查](../reports/v6_full_training_audit_20261009/rec_speed_mean42000_independent_world004_independent_review001.md)。名义协议未设置停车专项窗口，其静止专项0帧/0值不代表停车满分；运动和制动应读取单独请求协议。回放运行结束不等于全部场景通过，已有训练奖励也不替代这些行为读数。

追加四协议也已真实完成、0PPO。请求段采用5Hz稀疏trace抽样，归零停车采用50Hz完整10s窗口：

| 请求/停车项目 | 父端点 REC41393 | speed42000 |
|---|---:|---:|
| ±0.5m/s请求段速度MAE（正/反） | 0.141/0.170m/s | 0.102/0.130m/s |
| ±1rad/s请求段角速度MAE（正/反） | 0.052/0.056rad/s | 0.741/0.416rad/s |
| 正/反行驶后停车平面均速 | 2.87/2.64cm/s | 2.96/2.81cm/s |
| 正/反行驶后停车最大位移 | 0.129/0.231m | 0.156/0.258m |

新候选的行驶跟踪改善，但本轮正反旋转明显过转，不支持将它作为实车升级。两者所有停车副本均记录完整10s，这仅说明观测窗完整；不代表停车达标。四协议仍使用上述环境复用方式，尚未排除求解器内部状态影响；详细范围、逐场景失败原因和原始SHA见[对照结果](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/rec_speed_mean42000_remaining_protocols003_comparison.md)。这些中低速小样本不证明完整高速、台阶或跳跃能力。

**下文第1–8节及r2压缩包仍描述原REC41374交付，不是speed42000的身份表或能力证明；旧模型的比较和推荐仅适用于其已列明的回放条件。**

**按本次同条件“左侧自起＋制动”回放，优先使用旧 REC38000 作保守回放对照；最新 REC41374 的制动明显更差，不能因训练更久而称其更强。** 两者均未获得实车或完整高速能力资格。

旧 REC38000 ONNX：

`/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_38000_20261008/policy.onnx`

配套接口：`models/v6_recovery_candidate_38000_20261008/deployment/interface.json`，回放操作：`docs/CHASSIS_GUI_SUITE.md`。旧模型 SHA-256 为 `cd9c9be93ed65e56d9d4835ae9057490d271b4ad2e41cdf429778f1cf605b2ad`。

前次交付的封存候选为 **REC actor 14169 / 消费 41374** 的确定性均值 ONNX：

`/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_41374_20261009/policy.onnx`

REC41374 用于核查“倒地脚本完成后，由 RL 接稳”，不能据此跳过对制动退化的处理。后续误派课新增的19次恢复更新已另行封存为消费41393的完整PT，并已独立导出 ONNX，见下文补充；两个端点的文件身份不可混用。**目前没有同条件证据证明REC41374同时是倒地自起、高速两项最强的模型。** 本候选的实际恢复合同常规运动只覆盖 `|vx|≤1 m/s、|yaw|≤1 rad/s`；后续串行高速训练的成绩不能提前记到此 ONNX。它也不是由网络独立完成全部倒地动作：恢复脚本、传感器估计、200 ms 力矩混合与 RL 一起构成完整控制链。

`qualified=null`、`hardware_deployment_ready=false`。这是可加载、数值已核对的研究候选，尚无此端点同条件确定性全套物理评测或实车通过记录。本次未读取或修改现行 RMCS 安装版本，以下说明训练侧要求，不冒称 RMCS 已具备这些实现。

最新 CPU 导出补充：真实高速续训父端点 **REC41393** 已导出至 [policy.onnx](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_41393_20261009/policy.onnx)，必须配套使用该目录自己的 [interface.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_41393_20261009/deployment/interface.json) 和 [delivery_manifest.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_41393_20261009/delivery_manifest.json)。ONNX SHA-256 为 `4ceaa168f303f094cb6d106ab7de99ea20fe63f7220a7b7812a6fb94e3bb3dd8`；独立 CPU 重算25组原接口输入，原生均值动作与 ONNX 最大误差为 `4.917383e-7`。总消费41393、actor消费41223、恢复任务actor14188、学习谱系27646。此导出仅完成身份与数值核对，尚未完成该端点的物理能力回放，`qualification=null`、`hardware_deployment_ready=false`；不改变文首 REC38000 的保守回放推荐，也不把正在训练的高速能力记入父模型。独立证据见 [导出审查回执](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/rec41393_cpu_export_independent_review001.json)。原 REC41374 交付压缩包保持原模型身份，不包含这一新增导出。

REC41393 的原交付 manifest 内 `RNG_exact_restore_verified=true` 表述过强：实际 CPU 检查核对 actor/critic/Adam 状态及 RNG 保存结构，未安装并读回 CPU/CUDA RNG。以新增 [RNG 范围更正](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_41393_20261009/rng_scope_correction.json) 和 [更正后的独立校验清单](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_41393_20261009/DELIVERY_CORRECTED_SHA256SUMS)为准；原 ONNX、PT、合同与数值对照均保持原字节，不因此产生新的能力结论。

## 1. 身份、来源与验证

| 项目 | 固定值 |
|---|---|
| ONNX SHA-256 | `63574953c97aad374fa1554c53486cd540a3019857af72e0c06cc010214299bf` |
| 原始 PT SHA-256 | `ed24fe4e5e62fd8c8e01f81427c0741ae46799a237dee2b0fbd4b288df4e1085` |
| 合同 SHA-256 | `eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2` |
| 资产 manifest SHA-256 | `875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba` |
| 执行控制合同 SHA-256 | `8073540d1cec0b32c6216699863b3ef53064a2d385b7110abe291d9da0e9b1ac` |
| 训练源 | `6f07ec298e01d2686216787cfba34b394151f1aa`，source023 / REC worker v21 |
| 原始位置 | Kaiser：`train/recovery/attempt_009/train/model_final.pt`，隶属 `v6-task-campaign-20261001T110045Z-282358` |
| 总消费 / actor 消费 | 41374 / 41204 |
| 实际学习谱系 / stage actor / REC actor | 27627 / 27537 / 14169 |
| 原生 actor | RSL-RL 5.5.1 `MLPModel`，`35→256→128→64→6`，ELU |
| 推理 | Gaussian 的均值；单帧；无运行均值归一化；不随机采样动作 |

导出器通用字段 `learning_updates=41374` 沿用了消费计数名称，`consumed_updates=41204` 实际指 actor 消费；**真实学习谱系以 `checkpoint_infos.learning_lineage_updates=27627` 为准**，不能把 41374 称为此模型连续学习了 41374 次。

原始 `source.checkpoint.json`、原始 PT 全状态、合同、原生 agent 配置、进度和完成回执均保留。导出工具来自已冻结的 `bd4bdd052a07cab205e764f50513bcc752bec84e`；下载回执核对相关导出、观测、控制文件与原 REC 源逐字节相同。

独立 CPU 复核通过原始 full seal、合同/模型绑定、全包 manifest/SHA、原生严格加载和接口检查。25 组 ABI 快照重建的35D观测逐值相同；再加256组固定seed合成观测和零帧，共282帧，原生均值与 ONNX 通过 `atol=rtol=1e-5`，扩展集最大绝对误差 `1.52587890625e-5`（较大输出同时适用相对容差）。这不是行为误差或恢复成功率；本次独立复核没有 GPU、Isaac 场景、优化器更新或模型修改。

证据：`reports/v6_full_training_audit_20261009/rec41374_download.json`、`rec41374_independent_verification.json`、`verify_rec41374_handoff.py`。

## 2. 当前可据以判断的能力

最后已核验训练100窗为 REC13726–13825，早于本次14169端点；含训练随机化/探索，不是本ONNX确定性成绩。

| 内容 | 已有证据 / 边界 |
|---|---|
| 静站、变高度 | 230/305/430 mm静站平面均速3.42/2.68/2.94 cm/s；高度MAE2.22/2.47/6.39 mm |
| 常规前后运动 | 约±0.5 m/s指令实际均速+0.407/−0.359；约±1 m/s实际+0.748/−0.628，仍欠速 |
| 正反旋转 | yaw **角速度** MAE 0.131/0.125 rad/s；不是角度误差 |
| 停车 | 条件均速3.267 cm/s，35.27%样本超过3 cm/s |
| 恢复混课 | 686/7608=9.02%，混合容易出生、零力矩释放与原始倒地链；不能称纯倒地自起率 |
| 高速、跳跃、台阶跳 | 此候选没有对应端点的合格成绩；接口存在不等于功能已经学会 |

### 本次原生确定性短回放

两个候选在相同原生两 lane 场景、相同冻结合同、相同左侧90°出生与命令序列下运行，CPU ONNX 使用均值动作，0 PPO。每种场景只有一次固定出生，不能据此排名全功能“最强”。第二次回放增加位置与共同时间窗口记录；**两次场地宽度均为8 m，物理合同相同，第二次不是拓宽测试**。

| 同条件结果 | 最新 REC41374 | 旧 REC38000 |
|---|---:|---:|
| 左侧90°倒地：进入纯RL | 5.48 s | 5.48 s |
| 连续接稳后放行运动 | 6.48 s | 6.48 s |
| 左倒地原生完整链完成 | 7.46 s | 7.46 s |
| 完成时 roll / pitch | +0.26° / −0.80° | +0.52° / +1.37° |
| 0–5 s 加速窗 vx MAE | 0.393 m/s | 0.556 m/s |
| 同加速窗最大倾角 | 17.89° | 5.50° |
| 5–7 s 制动窗 vx MAE | 1.295 m/s | 0.165 m/s |
| 6.34–7 s 实加命令已为0：平面均速 | **1.965 m/s** | **0.083 m/s** |
| 同零命令窗最大倾角 | 35.43° | 10.54° |
| 2π rad/s旋转请求实际曝光 | 3帧；实加最高0.24 rad/s | 150帧；实加最高2π rad/s |

最新模型的加速跟踪更接近目标，但姿态更倾斜，制动后运动明显残留。零命令窗口两者均已完成命令斜率归零，不能归因为“目标还在减速”。该窗口只有0.66 s、33帧，并不等同于完整停车验收。旧模型旋转窗 yaw 角速度MAE仍为1.509 rad/s、最大倾角28.98°，也不能称高速旋转已经合格。

最新 flat 回合在7.06 s触发原生侧界截断，local y=3.713 m超过±3.7 m；fall、非轮接触和闭链失败计数均为0。**这是实际侧向偏移问题，不是摔倒；截断后的高速旋转曝光不足，不能判旋转通过或失败。** 旧模型存活至12 s回放末尾，属于右删失，不能计作整回合成功。原始 `stand_drift_max` 包含先前行驶距离，不能解释成独立10 s静站漂移；被动出生时 `first_motion_release_s=0.02` 也不是RL有效放行，正确有效放行是6.48 s。

此次只测左侧倒地链与短时平地命令序列。右侧、腹部、背部自起、反向高速、rough、普通爬阶、主动跳阶、原地/行进跳跃均未覆盖。**普通爬阶与主动跳阶是两个独立任务**：普通爬阶随前向运动进入，抬高车体、搭台后收腿，不发送JUMP请求；主动跳阶由跳跃请求触发，经历起跳、腾空收腿、落台及接稳。两者分别记录覆盖和成功率，不能互相替代；当前后续课程优先5–40 mm普通小楼梯，大双级爬阶与大双级跳阶另列训练与验收。

两个模型的 `contract.json` 与轮侧控制合同逐字一致，`deployment/interface.json` 仅模型身份字段不同，控制/观测接口相同；必须使用所选模型自己的identity、ABI快照与SHA清单。本文件第3–5节的共同接口说明可用于两者，表1身份和r2包内主模型则专指REC41374。

回放提取证据：`reports/v6_full_training_audit_20261009/model_compare/native_comparison_addendum.json` 与同名 `.md`。r2交付包附这两份小型结果，原始回放和旧交付包保留。

旧高速对照模型为：

`/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_speed_candidate_19837_20261005/policy.onnx`

SHA：`294d80c8618fdc4baced6587c3c36edd8d97cfba3d99afd05b3639c681e8b738`。它是 `speed←flat` 的另一条旧学习链，记录过±5 m/s和最高3圈/s请求，但已收到高速姿态、漂移质量反馈，且没有本次恢复接管训练。**不能将它的高速覆盖与 REC 的恢复数据合并，称本次主候选已掌握两者。** 旧模型保留作相同场景下的对照，不建议仅因请求范围更大而替换主候选。相关历史边界见 `docs/V6_REAL_DEPLOYMENT_HANDOFF_20261005.md`。

## 3. 35D观测与6D动作

输入名 `obs`、`float32[batch,35]`；输出名 `actions`、`float32[batch,6]`，实机通常 batch=1。输出是原始动作均值，**ONNX 本身不完成全部裁剪、目标换算和 PD**。

P策略轴序：`[LH,LK,RH,RK,LW,RW]` = `[L_joint1,LL_joint1,R_joint1,RR_joint1,L_joint3,R_joint3]`。LK/RK是第二主动驱动轴，不是被动大小腿内夹角。C控制轴序：`[LH,LK,LW,RH,RK,RW]`。

- P→C：`C=P[[0,1,4,2,3,5]]`。
- C→P：`P=C[[0,1,3,4,2,5]]`。

| 35D索引 | 内容 |
|---|---|
| 0–2 | `[vx_cmd,0,yaw_rate_cmd]`，m/s、m/s、rad/s |
| 3 | 支撑面相对底盘高度命令×5 |
| 4–6 | 机体系角速度×0.5，原单位rad/s |
| 7–9 | 机体系单位投影重力；正立为`[0,0,-1]` |
| 10–15 | P序相对名义输出轴角的wrapped delta；两轮角槽强制0 |
| 16–21 | P序六轴输出侧速度×0.1，原单位rad/s |
| 22–27 | P序上一策略步的有效原始动作；普通RL为已裁剪动作，恢复脚本/混合时按下述有效历史重建 |
| 28–34 | 命令模式及命令时钟上下文；普通平地和恢复接稳使用`[1,0,0,0,0,0,0]`，不是物理接触相位 |

机体系X前、Y左、Z上；最终拼接裁剪±100。无真实平面速度、真实接触、被动关节角、气簧长度或地形扫描输入；这些不能补进35D。IMU安装变换、编码器正负与零位必须由部署侧既有标定映射为模型坐标；本交接没有重新验证实车映射。

名义角P序为 `[-0.42,0.13742282595395358,0.42,-0.1374155762580851,0,0] rad`。观测角差为 `atan2(sin(q−q0),cos(q−q0))`，不能把膝内夹角直接填入第二主动轴。已经是输出侧的关节/轮速度不再除第二次减速比。

动作先按P序裁剪腿±3、轮±9，再变为C序。四腿目标为 `q0+0.25*a` rad，普通RL取相对当前角的最近整圈；轮目标为 `10*a` rad/s。轮速度定义在输出端，15.8减速比用于真实驱动换算，不再乘进policy动作。实际轮方向要映射到模型正号，不能直接假设左右驱动相同。

特殊命令槽有精确语义：28普通、29普通抬高/台阶模式、30预设落台高差×5、31台阶收回、32跳请求、33跳高×5或台阶高度×5、34请求时钟裁剪0–5 s。只有实现冻结的 `references.py::CommandReference` 与 `scut_observation.py::build_manual35` 才能使用这些非普通入口；本候选的跳跃成绩仍未验证。

## 4. 控制频率、PD与传输

| 项目 | 本合同 |
|---|---|
| 策略推理 / 目标更新 | 50 Hz / 20 ms |
| 仿真物理 / 新反馈PD | 200 Hz / 5 ms，每策略步4次 |
| 恢复参考FSM | 200 Hz / 5 ms |
| 腿PD | `tau=clip(160*(q_target−q)−2.5*dq,−40,40)` N·m |
| 轮速度P | `tau=clip(0.6*(omega_target−omega),−4.5,4.5)` N·m，输出端 |
| 命令斜率 | vx 1.5 m/s²；yaw 4 rad/s² |

这份合同明确是 **PC侧算力矩**，其注释记录MIT帧增益为0；`160/2.5`不是要求再将160/2.5同时写进DM驱动的MIT Kp/Kd，避免重复闭环。部署侧可以使用MIT格式发力矩，但帧格式、DM内部闭环与主机计算闭环是不同概念。

1 kHz发送/重发帧不等于1 kHz使用新反馈重算PD。合同保留原硬件1 ms记录以及200 Hz仿真近似，`hardware_equivalence_verified=false`。本次未核对现行RMCS的计算/发送调度；不能沿用旧文档把历史实现称作当前状态。若部署采用1 kHz反馈PD，需如实标注与本200 Hz训练近似的差别。

轮侧响应参数已在仿真中使用：左右附加惯量约0.003867/0.003818 kg·m²，库仑阻力0.049507/0.069577 N·m，黏性0.0006085/0.0003429 N·m·s/rad。它们属于plant，不应再重复作为硬件控制前馈。腿bag尚未qualified，四腿响应仍为工程先验，不能称全部六轴实测响应已经接入。

训练含信号年龄随机化（观测0–80 ms、动作0–60 ms，75%环境，±5 ms jitter）及另立的USB/CAN短持有/丢包压力模型。它们是工程随机化范围，不是实测等价声明，也不是让实机人为加上这些随机延迟。

组合普通命令按 `motion_limits.py` 投影：先共同缩放vx/yaw使 `|vx|+0.4373|yaw|/2≤5.4`，再缩小yaw使 `|vx*yaw|≤2.06`。只放宽遥控器上限而没有相同坐标/投影/斜率，会改变policy看到的命令语义。

## 5. 恢复脚本与RL如何交接

与模型配套的恢复逻辑来自原训练提交 `6f07ec298e01d2686216787cfba34b394151f1aa`，而非任选一套旧自起动作。`recovery.py`中历史注释曾写1 kHz PD，当前实际频率以本冻结合同和 `RecoveryTraining`的运行配置为准。

1. **脚本阶段**：根据编码器/FK/IMU选择PREPARE或FOLD/PLANT/ORBIT/THRUST/SIDE/CAPTURE路线，使用连续输出轴角；绕圈脚本不能每帧重新取最近圈而反向。SIDE实际速度覆盖为1.5 rad/s。脚本的目标姿态由冻结profile给出，prepare还包含静载力矩除以160的目标修正，不能只用几何站姿代替。
2. **进入交接**：PREPARE同时满足传感器条件、参考剩余误差<0.02 rad、实测跟踪误差<0.12 rad，连续0.1 s进入BLEND。这里不是任意动作计时结束便硬切RL。
3. **200 ms混合**：`tau=(1−alpha)*tau_script+alpha*tau_RL`。混合的是各自限幅后的**力矩**，不能改成只插值policy动作。混合结束进入纯RL；脚本/混合区间不当作PPO有效样本。
4. **纯RL接稳**：此时RL已经接管，vx/yaw仍保持0。八项条件每5 ms同时成立并连续1 s，才释放后续运动命令；条件打断则连续计时归零。这个1 s是已存在的运动放行逻辑，本交接没有增加新gate。

八项条件为：输入有效、FK/IMU支持成立、车体净空估计成立、倾角<8°、角速度模长<0.75 rad/s、估计高度在(0.27,0.36)m、最大轮缘速度<0.25 m/s、最大腿轴速度<2 rad/s。support与body_clear来自可部署估计，不是真值接触：需要有效闭链FK、两轮估计高度差<0.035 m、最小高度>0.12 m、加速度计比力模长6–16 m/s²及gyro<8；body_clear还要求最小高度>0.27 m及轮响应探测或已到BLEND/RL。

脚本上限8 s，已开始的200 ms混合可以走完；失败状态必须由显式复位结束。当前逻辑不因纯RL失稳而无限重跑脚本。训练侧额外用真实接触/平台与后续命令完成情况判定结果，这些评测真值不能导入实机命令或actor观测。

**前一动作历史必须连续**：在脚本/混合期间，按脚本腿目标相对名义角除以0.25、轮目标除以10构造脚本动作，再按alpha与RL动作混合；零力矩释放/失败历史为0。不能在进入纯RL时随意清空全部历史，或把上一PD力矩填入22–27槽。参考 `RecoveryTraining::effective_action_history`。

## 6. 交付文件和恢复配套

部署普通RL最少需要主模型目录中的：

- `policy.onnx`、`policy.onnx.json`、`policy.onnx.contract.json`/`contract.json`、`SHA256SUMS`、`bundle_manifest.json`；
- `deployment/interface.json`、`abi_test_vectors.json`、`scut_observation.py`、`references.py`、`v5_control.py`、`v6_wheel_response_control_200hz_v1.json`、资产`manifest.json`。

要复现恢复链，还必须一起带上**原REC训练提交的下列内容**，不能仅交一个ONNX：

| 文件/合同子树 | 作用 |
|---|---|
| `contract.json.recovery_training`完整子树 | dt、0.2 s blend、1 s stable、超时、SIDE速度、wheel_balance、有效动作历史等绑定 |
| `contracts/v6_recovery_profiles_v2.json` | 名义/fold/thrust/support姿态、静载prepare参考、轴符号、合法出生几何；SHA `007363fdfe35804b01ad616a1453bf163bf70c1bb7dd59b6a8f743b26c0b4441` |
| `contracts/v6_height_lookup_v1.json` | 闭链编码器→轮心几何；SHA `c3e215187828fae328e3df3af1728897feb6695616ec8b9bdd4322433ea00282` |
| `src/wheeled_tasks/chassis/recovery.py` | 核心参考FSM、状态转移、稳定计时、混合系数 |
| `recovery_training.py`同目录 | 观察器接入、脚本轮平衡、力矩混合、命令保持、动作历史；仿真适配参考，不直接复制真实状态写入到RMCS |
| `continuous_encoders.py`、`mechanism_observer.py` | 连续轴角/闭链分支、FK高度几何 |
| `recovery_observer.py`、`step_trigger.py` | IMU/轮响应探测及条件高度估计；硬件要使用实测IMU，不能用仿真root速度生成加速度 |
| `motion_limits.py` | 普通命令可达域与投影语义 |
| `contract.json.v5_control`、`command_slew`、`motion_limits`、`command_reference`、`manual_context35`、`policy_action_order` | 动作、命令与35D接口必须成套匹配 |

上述源码精确SHA见独立验证JSON的 `recovery_source_bindings`。特别是工作树 `recovery_training.py` 已有后续集成课程改动，与原REC源不同；恢复参考包应从上述固定git提交提取，不能直接打包当前同名文件。profile/lookup是模型几何来源，不能当作实车轴零点标定完成的证据。模型目录本身是导出包；恢复源码参考附件需由完整交付包另行带齐。

`model.pt`、`source.checkpoint.json`、`model.checkpoint.json`、`agent_config.json`、`provenance.json`、`progress.json`和`completion.json`用于追溯与续训，不是推理进程的必要内存负担。

## 7. 部署侧最短核对路径

在模型目录执行 `sha256sum -c SHA256SUMS`。用包内25组快照逐槽对照35D与6D均值，容差 `atol=rtol=1e-5`；随后核对P/C序、名义角、输出侧单位、动作裁剪和PD目标。ABI通过仅证明计算接口，不能替代真实方向/姿态/响应验证。

恢复联调同时记录脚本phase、blend、八项通过掩码、stable秒数、motion_released、原始/有效动作和实际力矩，从而区分脚本未完成、已接管未接稳、已接稳未完成后续运动。当前低速与恢复候选命令域保持上文范围；高速比较使用旧speed模型单列结果，等待新串行高速端点再交付同一学习链的新候选。

本机GUI回放仍复用 `scripts/play_chassis_suite.py/.sh`，把 `--onnx` 明确设为所选38000或41374路径，不依赖默认“最新”。本次已完成上述两场景的原生无GUI短回放及CPU数值核对，没有证明全部18个GUI场景通过。原生回放入口和操作见 `docs/CHASSIS_GUI_SUITE.md`。完整课程问题及接续方案见 `docs/V6_FULL_TRAINING_AUDIT_20261009.md`。

## 8. r2完整交付包

最新41374模型及其冻结恢复FSM参考、接口和新增回放说明位于：

`/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_priority_handoff_20261009_r2.tar.gz`

解包目录的 `model/policy.onnx` 仍严格是REC41374；此包**不包含旧38000模型**，以免把较好回放结论混到最新模型身份上。旧38000请使用文首独立模型目录及其自身SHA文件。原始 `v6_recovery_priority_handoff_20261009.tar.gz` 保持不变，r2增加同条件回放证据和上述选型说明，省略推理不需要的原始PT与大型完成报告，包内manifest如实列出省略项及源SHA。原始完整导出目录与r1包仍保留这些文件，不改变任一ONNX字节。
