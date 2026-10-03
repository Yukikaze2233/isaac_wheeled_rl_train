# V6 平地模型部署交接：2026-10-03

推荐交付对象是 **V6 平地研究候选**，本域完成 13,000 次 actor 更新，累计消费 14,020 次（包含历史课程与 critic 适应）。在已做同条件配对评测的平地终点与恢复 checkpoint16000 中，这个平地模型的基础平移、旋转和静止保持更有证据支持；不是所有历史模型的全局最优，也不是六任务集成模型，没有硬件发布资格。恢复线已合法封存，Kaiser 当前从这个平地父模型完整继承 actor、critic、Adam，训练独立的高速域；本文件交付的平地文件不会随继续训练改变。

## 1. 模型与证据位置

本机模型目录：

```text
/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_flat_candidate_14020_20261003/
```

| 文件 | 用途 |
| --- | --- |
| `policy.onnx` | 实际推理模型；确定性 actor，35D 输入、6D 输出 |
| `policy.onnx.json` | 模型、原检查点、资产、控制合同 SHA 与学习时钟 |
| `policy.onnx.contract.json` | 训练时原始物化合同，原字节保存 |
| `deployment/interface.json` | 本次交接的 ABI、名义角、动作解码、频率及适用范围 |
| `deployment/abi_test_vectors.json` | 观测→ONNX 原始输出对照；用于部署侧数值一致性测试 |
| `deployment/v6_pair_from_dev_rmcs_rl.json` | 既有 V6 编码器映射参考，保留未完成的硬件核验状态 |
| `deployment/manifest.json` | 本次 V6 232 mm / 105° 资产身份与名义关节位置 |
| `deployment/v6_wheel_response_control_200hz_v1.json` | 本次实际使用的原生控制合同 |
| `deployment/scut_observation.py`、`v5_control.py` | 冻结训练源码中的观测组装与动作解码参考 |
| `bundle_manifest.json`、`SHA256SUMS` | 包内每个交接文件的 SHA-256 |

完整交接压缩包是同目录的 `v6_flat_candidate_14020_20261003.tar.gz`。压缩包旁的 `.sha256` 验证压缩包；解压后在模型目录执行 `sha256sum -c SHA256SUMS` 验证内容。压缩包不包含已训练完成的恢复/跳跃模型。

10月3日新增的本机交付目录为：

```text
/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_model_delivery_20261003T2235/
```

新包 `v6_model_delivery_20261003T2235.tar.gz` 位于该目录；`recommended_flat/` 包含上述推荐平地 ONNX、原始完整 PT、合同和接口，`latest_speed/` 包含最新高速检查点 PT、ONNX、合同和对应源码。根目录 `selection.json` 记录推荐依据与两者身份，`handoff_document.md` 是本交接说明的更新副本，并附评测和验证证据。原平地包保持不变；“最新”和“推荐”分别保留，不能把未经评测的较新高速模型直接称为更好的模型。

Kaiser 原始学习状态：

```text
/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/flat/attempt_007/train/model_final.pt
/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/flat/attempt_007/train/policy.onnx
```

| 身份 | SHA-256 |
| --- | --- |
| ONNX | `4006bf79182e14074f38c3e8f573fe1870fdfeba3fcc0760bb24cc5752161e8a` |
| 原始 PT | `7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f` |
| 平地物化合同 | `666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7` |
| V6 资产 manifest | `875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba` |
| 控制合同 | `8073540d1cec0b32c6216699863b3ef53064a2d385b7110abe291d9da0e9b1ac` |

生成这个平地模型的源码是 `2778206b5905c2760ccc381f2592b08e1cad611a`。v8 recovery已合法封存；v9仅调整后续speed/terrain训练终点并通过原生review从保留flat父接speed，六worker合同不变。这个交接目录内的平地模型不随远端续训改变。实际运行源码、是否开始真实PPO及消费以 `reports/v6_task_campaign_20261001/deploy/launch.json` 为准；恢复重设计仍是讨论稿，未在本轮启用。

本次另交付的最新高速不可变检查点为 `train/speed/attempt_000/train/checkpoints/update_00016500/model.pt`，累计消费 **16,500**、高速域 actor 更新 **156**，SHA-256 为 `6e29e0a4c80fddd83aff3af3402af91d4b16a5311016ef8dbd0a58fb1a430438`，运行源码为 `9caf96dbdc7e08e91f3f61d25e27069f3739cf5b`。这是高速训练中的快照，未做与推荐平地模型同条件的固定行为评测，状态为 unassessed；156 是高速 actor 时钟，不能把16,500解释成高速训练轮次。

## 2. 当前能力与适用范围

10月3日完成了固定确定性配对评测：平地终点与恢复不可变 checkpoint16000 在相同冻结评测器中顺序运行，使用高斯策略的确定性动作中心，不加探索噪声。范围为 **canonical quick 的63用例、每用例4个副本**，`signal_delay=false`，物理步长0.005s、策略周期0.02s，判据保持不变。下面列出推荐平地模型的关键读数：

| 固定用例 | 平地终点实测 | 判读 |
| --- | --- | --- |
| `forward_05` | 实际vx 0.48272 m/s；vx MAE 0.01728 m/s | 该用例通过 |
| `backward_05` | 实际vx −0.48777 m/s；vx MAE 0.01223 m/s | 该用例通过 |
| `rotate_1` | yaw角速度 MAE 0.01064 rad/s | 该用例通过 |
| `rotate_1_reverse` | yaw角速度 MAE 0.01531 rad/s | 该用例通过 |
| `stand_305mm` | 10s峰值位移0.06371 m；高度MAE 0.447 mm | 该用例通过 |
| `start_stop_05` | 驻停均速0.03075 m/s | 略高于0.030 m/s线，该项未通过 |

对应恢复 checkpoint16000 的前进vx降至0.26351 m/s、反向vx为−0.23564 m/s，正向yaw角速度MAE为0.30981 rad/s，305mm静止峰值位移为0.12067m；但它的驻停均速改善到0.01424m/s。因此推荐平地模型用于本次基础能力交接，不把恢复模型描述为所有项目都退化。此对照不是完整322用例验收，不证明全延迟随机化或硬件质量；不能将“63用例×4”记成完整域通过。原始结果与汇总见 `reports/v6_recovery_takeover_20261003/learning_audit/paired_flat_clean_means.json`、`paired_recovery16000_clean_means.json` 和 `paired_clean_means_review.json`。

最近平地完整 100 更新窗口（全强度延迟随机化、带探索）的前/后 0.5 m/s 速度 MAE 约 3.2/2.7 cm/s，正/反 yaw **角速度** MAE 约 0.085/0.081 rad/s。230/250/305/430 mm 高度 MAE 约 1.6/2.0/1.3/5.6 mm；停车均速约 2.2 cm/s。完整 10 s 静止窗口的峰值位移均值约 8–9 cm，少量长尾仍有 37–51 cm。这些是训练曲线统计，不能替代固定确定性验收。

本机已通过原生 ONNX 导出数值检查；原生校验最大绝对误差 `5.662441253662109e-7`。此前 13,500 消费快照完成了 12 s CPU PhysX 自由底盘回放；本次最终模型已经在本机 Isaac GUI 实时运行。GUI 使用单环境、名义评测参数和 box 平地，无外部牵引，不能据此宣称随机化全域或实车通过。

本次交付适用于平地静站、升降、正反平移、旋转、启停的接入与回放。高度训练域为 0.23–0.43 m。高速、台阶、跳跃、非常规倒地自起尚没有被这个导出模型训练完；平地 ONNX 的跳跃请求应保持关闭。与 V5.3 的资产、控制合同和评测域不同，尚无同条件结果支持“整体一定更强”。

## 3. 35D 观测 ABI

ONNX 输入名 `obs`，类型 `float32`，形状 `[batch,35]`；输出名 `actions`，形状 `[batch,6]`。部署使用 batch=1，不使用 critic 的 81D，不堆历史帧，不再追加运行均值归一化。

**策略顺序 P**：`[LH,LK,RH,RK,LW,RW]`，对应 `[L_joint1,LL_joint1,R_joint1,RR_joint1,L_joint3,R_joint3]`。这里 LK/RK 是并联机构的第二主动电机，不是被动大小腿铰链；不能把观测角替换成实际内膝角。

| 0 起始索引 | 内容与缩放 |
| --- | --- |
| 0–2 | `[vx_cmd,vy_cmd,yaw_rate_cmd]`；普通平地 `vy_cmd=0`，单位 m/s、m/s、rad/s |
| 3 | `height_cmd * 5`；是高度命令，不是实测高度 |
| 4–6 | 机体坐标系 gyro XYZ，rad/s，乘 0.5 |
| 7–9 | 单位重力方向在机体系的投影；正立应接近 `[0,0,-1]` |
| 10–15 | P 顺序角偏差：前四轴 `atan2(sin(q-q_nom),cos(q-q_nom))`，两个轮角固定 0 |
| 16–21 | P 顺序六轴输出侧速度，rad/s，乘 0.1 |
| 22–27 | 上一策略步的 **已裁剪原始动作**，P 顺序；不是角目标、速度目标或力矩 |
| 28–34 | 命令上下文；本平地模型普通模式固定 `[1,0,0,0,0,0,0]` |

所有输入最终 clip 到 ±100。部署入口 reset 清 previous action；普通 50 Hz 推理不重复清零。未来任务的上下文要随对应模型交接，目前不要用接触相位、轮力或台阶真值填这些槽。

机体系为 X 前、Y 左、Z 上；IMU 使用部署侧既有安装外参统一旋转。重力方向使用单位向量，不填 m/s² 加速度。轮速为减速箱输出侧，不再乘/除 15.8。绝对车速、地面真值高度、被动铰链、气簧位移和接触力都不在 actor 输入中。

## 4. 6D 动作与下层控制

策略输出顺序仍为 P。训练内部控制顺序 C 是 `[LH,LK,LW,RH,RK,RW]`；`C=P[[0,1,4,2,3,5]]`。部署使用 P 顺序时不要再额外置换一次。

P 顺序名义位置（rad）：

```text
[-0.42, 0.13742282595395358, 0.42, -0.1374155762580851, 0.0, 0.0]
```

四个腿原始动作裁剪至 ±3；两个轮原始动作裁剪至 ±9。

```text
desired_leg = nominal_leg + 0.25 * clipped_leg_action
leg_target  = current_leg + atan2(sin(desired_leg-current_leg), cos(desired_leg-current_leg))
wheel_target_rad_s = 10.0 * clipped_wheel_action

tau_leg_model = clamp(160*(leg_target-q_leg) - 2.5*dq_leg, -40, 40)  # N·m
tau_wheel_model = clamp(0.6*(wheel_target_rad_s-dq_wheel), -4.5, 4.5) # 轮输出侧 N·m
```

目标每 20 ms 更新并持有，下层 PD 使用当前反馈。目标角使用连续电机输出轴坐标和最近共同圈数，不能独立把髋、第二主动轴重新归零。±90 rad/s 是动作解码上限，不代表这个平地模型已学会该速度。

气簧是仿真中的被动机械力；本模型未使用预览重力/气簧前馈。部署侧不能默认再加一份气簧补偿，否则控制合同改变。轮端 4.5 N·m 和腿端 40 N·m 是本次训练的控制上限，不是 MIT 协议量程或电机额定参数。

| 层 | 本次训练/回放 | 部署交接事实 |
| --- | --- | --- |
| ONNX/策略目标 | 50 Hz | 50 Hz |
| 仿真物理与新反馈 PD | 200 Hz，dt=0.005 s | 明确的计算近似，未证明与 1 kHz 等价 |
| 当前本机 RMCS V6 控制 | 不属于本训练运行 | 200 Hz 算 PD、1 kHz 持有并重发力矩 |
| 用户目标/录包 | raw bag 1 kHz | 若改成真正 1 kHz 新反馈 PD，需另作同轨迹响应核验，不能仅把 CAN 发送频率称为 PD 频率 |

已辨识轮侧附加 armature、静摩擦和粘滞阻尼用于仿真；它们不是要重复注入实车的补偿力矩。腿侧 bag 尚未 qualified，四腿响应仍为显式工程先验；160/2.5 是使用的闭环参数，不能称为已完成腿电机响应辨识。

## 5. 当前 RMCS 对接状态与仍待验证项

**2026-10-03 22:47（Asia/Shanghai）只读复核**：本机 RMCS 的 `rmcs_ws/src/rmcs_rl/models/wheel_leg/policy.onnx` 已是本次推荐的 **V6 flat_14020**，SHA `4006bf79182e14074f38c3e8f573fe1870fdfeba3fcc0760bb24cc5752161e8a`；旧 V5 文件留在 `models/wheel_leg/legacy_v5/`。当前 `wheel-leg-infantry-rl.yaml`、`policy.hpp`、`observation.cpp`、`action.cpp` 已包含 V6 profile、名义角、映射、35D 观测、动作解码和控制参数。此前“当前 RMCS 仍是 V5、尚待更换”的描述已失效。本次交付只读复核了这些本机文件，没有修改 RMCS、连接硬件或验证机器人当前运行的安装包。

| 本机 RMCS 已核对的软件项 | 当前值 / 边界 |
| --- | --- |
| 模型 profile / 身份 | `v6_flat_14020`；启动校验上述 ONNX SHA 与 nominal |
| 名义角与四腿映射 | 与本交接 P 顺序名义角、`J=-I`、offset `[1.6,2.93,-1.6,-2.93]` 一致 |
| 轮侧符号候选 | `wheel_model_scale=[1,1]`；实车逐轮正方向仍未验证 |
| 观测 / 动作 | 35D 单帧、普通上下文 `[1,0,0,0,0,0,0]`、previous clipped action；腿0.25rad / 轮10rad/s缩放 |
| PD与节拍 | 腿160/2.5、±40N·m；轮P0.6、±4.5N·m；50Hz推理、200Hz新反馈PD、1kHz executor持有重发 |
| 命令域 / 参考 | 高度0.23–0.43m；vx斜率1.5m/s²、yaw斜率4rad/s²；遥控vx±0.5m/s、yaw±1rad/s |
| 部署侧输入整形 | 高度6s smoothstep；普通直立接管默认直接接管，可配置0–300ms smoothstep；这些不是实测高度反馈或完整自起验收 |
| 硬件与自起门控 | `calibration_ready=false`、`soft_limits_ready=false`、`recovery_enabled=false`、`recovery_profile_ready=false`；`imu_alignment_ready=true` |

RMCS 已移植独立的冻结 V6 自起 profile、闭链 LUT 与200ms线性接管路径；软件实现存在不等于默认已启用或硬件通过。旧 V5 几何与结果仍只用于 legacy 回归，当前 V6 软件/仿真证据以部署仓库的最新文档为准。

V6 编码器参考来自 `.script/identification/calibration/v6_pair_from_dev_rmcs_rl.json`，四轴 P 顺序：

```text
q_v6  = -I * q_api + [1.6,2.93,-1.6,-2.93]
dq_v6 = -I * dq_api
tau_api = (-I)^T * tau_v6
```

这是 RMCS 驱动 **已经 reversed、bias=0 后的 API 反馈** 到 V6 的映射；驱动 raw 角不是这个 API 角，不能重复应用 reversed。参考文件的 `hardware_multipoint_g0_verified=false` 和本训练 `hardware_motor_mapping_verified=false` 保持原事实。轮侧沿用设备 15.8 输出单位；最终两个轮轴符号需与 V6 实际模型逐轴对照后填写。

既有 `configuration.cpp / observation.cpp / action.cpp / policy.hpp` 已完成上述基础软件对齐，继续使用 RMCS 原生 Component status/command 接口和唯一六轴力矩出口；后续接入应复核实际安装版本与硬件事实，而不是重复移植：

1. 读取交接文件、检查 SHA、35→6 接口及 `abi_test_vectors.json`；数值对照用 `atol=rtol=1e-5`。输出使用确定性均值，不随机抽样。
2. 核对机器人实际安装包中的模型身份、V6 名义角、四轴 J/offset、轮侧符号候选和高度域与本机源码一致。当前本机 `assemble_observation_` 已接受0.23–0.43m，vx/yaw参考斜率已对齐训练的1.5m/s²与4rad/s²；不再沿用旧V5的高度与斜率差异描述。轮符号和多点机构标定仍需实测，不能因配置已写入就记为通过。
3. 用同一 IMU/编码器快照同时产生 Python 原生 35D 与 RMCS 35D，逐槽比较；再核对六输出动作、角/轮速目标、PD 模型力矩与 API 力矩变换。优先保持原生实现，不另造传感器回读或控制输出链。
4. 相对膝止挡采用本次 V6 40–105° 的闭链/LUT定义；105° 是机械参考，106/107° 实测误差要作为标定误差处理。主动大腿轴 continuous，不按被动膝上限裁剪髋角。现有 V5 40–110° 自起轨迹/成功率不能直接转作 V6 证据。
5. 正常入口、双下 `0xFD` 失能和既有传感器 freshness 保持原生语义。先记录实际运行的模型/配置身份与同轨迹响应；接管前脚本姿态及 blend 必须与 V6 机构匹配。本平地交接不启用跳跃，不解除当前硬件门控，也不把恢复训练或软件移植标成已验证实车自起。

当前部署说明：`/home/yukikaze/Documents/workspace/RMCS/docs/zh-cn/wheel_leg_v6_model_deployment_20261003.md`；最新V6自起实现与完整C++闭环结果：`/home/yukikaze/Documents/workspace/RMCS/docs/zh-cn/wheel_leg_v6_recovery_alignment_20261003.md`。已有软件回执记录25组ONNX ABI、7组原生快照/3项Component对照通过；本次只读复核没有重跑这些测试。20261002自起部署文档现为V5历史记录，不再作为当前V6配置的权威入口。上述本机文件、软件对照和Isaac动态结果均不能替代真实CAN/USB/BMI088响应、六轴方向、多点机构标定或实车倒地验收。

## 6. 本机回放与后续训练

10月3日11:11本机启动 systemd user unit `v6-flat-wasd-arrows-20261003T111124.service`，复用 `scripts/play_v5_grounded.py` 与原生 `KeyboardCommand`。点击 viewport 获取焦点：W/S 平移、A/D 旋转、Q/E 或 T/G 升降、Space 停车、P 暂停、R 重置。默认键盘上限 0.5 m/s、1 rad/s；可调滑条，但超出训练命令域的操作不作为能力验收。此平地模型不显示 J/K 跳跃按钮。

车体上方白箭头表示水平朝向，青箭头表示平移命令，青弧箭头表示yaw角速度命令，粉箭头表示实测水平移动速度。方向由当前机体姿态转到世界水平面；移动箭头长度为0.6m/(m/s)，显示长度封顶0.6m，零值隐藏。原有电机力矩和气簧力箭头保留。该显示仅影响本机回放，不修改训练、策略输入或控制。回放实测正在执行ONNX，前后方向和旋转机体系的箭头坐标验证通过。

```bash
cd /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule
OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/home/yukikaze/.local/lib/compat \
  TMPDIR=/home/yukikaze/.cache/kit-tmp OMNI_KIT_ACCEPT_EULA=YES \
  /home/yukikaze/isaacsim60-venv/bin/python -B scripts/play_v5_grounded.py \
  --onnx models/v6_flat_candidate_14020_20261003/policy.onnx \
  --device cpu --floor-boxes --output reports/v6_flat_manual_replay_$(date +%Y%m%dT%H%M%S)
```

自动课程顺序为 **flat → recovery → speed → terrain → jump → stepjump**。flat actor=13,000已封存；recovery于actor2294／总消费16334合法封存，未通过完整恢复验收。v9指定speed首轮3563、terrain首轮1186 actor更新，分别对应本项目平地高速域和地形/普通上台阶域；它们不等于SCUT所有同名rough任务已覆盖或已达标。jump/stepjump仍各15000，20k每域消费上限及总121000保留。每500保存，每域常驻PPO，不等待45项gate。speed、terrain从flat父独立分支开始，critic适应计消费、actor时钟从0开始；到期自动结束为unassessed，不能称六能力已经集成。

当前speed域全部是flat；rough在其后的terrain域，占4%，当前静态4096布局约164个环境，命令0.4m/s、305mm高度、起伏上限20mm。terrain还含上下/横坡、材质、普通台阶与楼梯；recovery里的平台初态用于自起接管，不替代rough通行。当前rough配置不代表独立粗糙地高速专训，且执行顺序不等于累计父链：recovery、speed、terrain均从flat分别初始化。

TensorBoard 服务仅在 Kaiser；浏览器入口 `http://127.0.0.1:6006/#scalars`。当前训练 run 为 `v6-task-domains-d073fe4/speed/attempt_000/train`；`recovery/attempt_001` 与 `flat/attempt_007` 是已封存记录。本机当前打开的仍是该平地ONNX回放，并非远端高速训练画面。定时 Codex heartbeat 仍 PAUSED，tmux 原生自动课程独立运行。读取最新 `deploy/launch.json` 确认身份，不因旧停止回执重复启动。
