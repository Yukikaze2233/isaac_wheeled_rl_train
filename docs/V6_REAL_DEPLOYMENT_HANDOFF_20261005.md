# V6 高速平地 ONNX 交接：2026-10-05

主交付是已完成高速平地训练的确定性 ONNX：

`/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_speed_candidate_19837_20261005/policy.onnx`

它覆盖同一策略的静止、低速、高速和变高度命令，使用 V6 232 mm / 105°闭链底盘及现有轮响应模型。它是完成训练的 **未验收候选**，不是新跳跃模型，不包含10月5日新增实际触地落地成本，也没有证据表明它优于当前平地部署模型。ONNX 可以运行，数值、ABI 和身份已验证；`qualified=null`、`hardware_deployment_ready=false` 保留真实物理和硬件验证状态。RMCS 侧由使用者补齐，本交付没有修改 RMCS 或替换机器人模型。

## 模型身份与训练范围

| 项目 | 实际值 |
| --- | --- |
| ONNX SHA256 | `294d80c8618fdc4baced6587c3c36edd8d97cfba3d99afd05b3639c681e8b738` |
| 原始 PT SHA256 | `123841b609439939e9c9c8962c0171330892276446a692c936c8f33375ce11da` |
| 合同 SHA256 | `37837b286de10a66404ef155ae84ead613a90f76375d4abe6874a042eae6b406` |
| 资产 manifest SHA256 | `875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba` |
| 执行控制合同 SHA256 | `8073540d1cec0b32c6216699863b3ef53064a2d385b7110abe291d9da0e9b1ac` |
| actor 消费时钟 / 文件名 | 19,837；与合法学习链和优化器时钟分开 |
| 学习更新时钟 | 19,917；学习链更新16,981；global actor16,931 |
| speed 域 actor 更新 | 3,563；最后一块1,879 actor更新、1,889总学习更新 |
| 原始完成时间 | `2026-10-04T05:02:08.070808+00:00` |
| 训练域 | `speed` / `mixed_locomotion`，129组全部 flat |
| 原始检查点 | `kaiser:/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/speed/attempt_001/train/model_final.pt` |

原始每文件封存为 `source.checkpoint.json`；`model.pt` 保留相同原始 PT 字节、actor/critic/Adam/随机数/课程状态。原合同、`agent_config.json`、完成回执、进度和源文件 SHA 一并封装。导出直接构造原生 RSL-RL `MLPModel`，严格加载保存的 actor，无 Isaac 场景、GPU 训练或优化器步骤。网络为 `35→256→128→64→6`，ELU，无观测运行均值归一化，使用 Gaussian 的确定性均值。

冻结训练命令的直行范围为 **±5 m/s**、高度 **0.23–0.43 m**；包含1–5 m/s启停和速度曲线，变速过渡5秒。固定旋转任务达 **±18.84956 rad/s**（3圈/秒），连续高速旋转采样为±8 rad/s。两者是训练请求范围，不是实车已通过的速度范围。训练分配为高速度25%、高速过渡10%、普通平移15%、普通旋转10%、普通过渡15%、静止锚点12%、连续静止8%、变高度5%。

非稳态覆盖包括启停、速度曲线、变高度和训练扰动；**没有倒地插值自起后的RL接管专项训练**。实际父链是 `speed←flat`，recovery是独立分支，不能把recovery训练成果视为这个speed模型已经继承。全局 `recovery_training.enabled=true` 只说明模块存在；本域129组没有recover/takeover/born/edge自起组。最后一块完成回执 `script_or_blend_transitions_in_block=0`、有效策略转移185,696,256；这与本域未进行脚本自起/接管覆盖一致，也不授予自起资格。

组合命令沿用 `motion_limits`，半径0.06 m、轮距0.4373 m、轮速上限90 rad/s、侧向加速度上限2.06 m/s²。冻结实现先共同缩放vx/yaw使 `|vx|+0.4373*|yaw|/2≤5.4`，再仅缩小yaw使 `|vx*yaw|≤2.06`；不是任意把±5 m/s与最大yaw同时发送。命令变化斜率vx为1.5 m/s²、yaw为4 rad/s²。

## 数值验证与物理证据边界

原生确定性 forward 和原生 `as_onnx` wrapper 逐值相同。既有导出器的 ONNX checker、Runtime 对照以及包内35组ABI向量通过，含±5 m/s、±8 rad/s、变高度、普通模式和跳跃上下文数值快照；最大绝对误差 **1.9073486328125e-6**，比较容差 `atol=rtol=1e-5`。这份 ONNX 的 SHA 与训练端完成回执中的原导出 SHA 完全相同。跳跃向量只验证计算接口，不授予跳跃能力。

原训练功能静态预检及运行配置预检通过，记录了50 Hz策略、200 Hz新反馈PD与随机化/信号延迟配置实际开启；完成回执明确记录功能 transport / precision / randomization 探针 **not_run**。最后一块带探索训练有185,696,256有效策略转移，记录fall 8,124、nonwheel_contact 17,984、closure_gap 103、spring_travel 11；这些聚合训练计数不能转换为确定性高速成功率。没有新的确定性高速技能验收、同轨迹实车响应或实车高速测试结果。已有轮侧响应辨识和气簧证据继承，腿bag仍未qualified，腿驱动响应仍是工程先验。

## 35D观测与6D动作

输入 `obs: float32[batch,35]`、输出 `actions: float32[batch,6]`，实机batch=1；单帧，无历史。P策略轴序：`[LH,LK,RH,RK,LW,RW]`，即 `[L_joint1,LL_joint1,R_joint1,RR_joint1,L_joint3,R_joint3]`。LK/RK是第二主动轴，不是被动内膝铰链。

| 索引 | 内容与单位 |
| --- | --- |
| 0–2 | `[vx_cmd,0,yaw_rate_cmd]`，m/s、m/s、rad/s |
| 3 | 支撑面相对底盘高度命令×5 |
| 4–6 | 机体系gyro，rad/s×0.5 |
| 7–9 | 机体系单位投影重力，正立`[0,0,-1]` |
| 10–15 | P序输出轴相对名义角的wrapped delta，rad；两轮角固定0 |
| 16–21 | P序输出侧六轴速度，rad/s×0.1 |
| 22–27 | 上一策略步已裁剪的原始动作，P序；不是关节目标或PD力矩 |
| 28–34 | 本高速平地接入使用普通模式`[1,0,0,0,0,0,0]` |

各拼接项最终裁剪±100，无额外乘法、运行归一化或root速度输入。机体系X前、Y左、Z上。IMU与模型坐标必须使用既有已登记变换。关节delta用 `atan2(sin(q-q0),cos(q-q0))`；已是输出侧的编码器速度不再除一次减速比。

名义角P序为 `[-0.42,0.13742282595395358,0.42,-0.1374155762580851,0,0]` rad。C控制轴序为 `[LH,LK,LW,RH,RK,RW]`，`C=P[[0,1,4,2,3,5]]`。腿动作裁剪±3，轮动作裁剪±9；腿目标名义角加 `0.25*action`，再取相对当前连续角的最近圈数，轮目标为 `10*action` rad/s。不能继续使用原平地profile的轮动作±3上限来声称覆盖高速训练执行域。

腿PD为 `160*(q_target-q)-2.5*dq`、模型力矩限幅±40 N·m；轮速度P=0.6、输出侧力矩限幅±4.5 N·m。原执行控制合同保留力矩与速度约定。气簧是被动机械力，不额外增加一份气簧前馈。

四腿API到模型参考为 `q_model=-I*q_api+[1.6,2.93,-1.6,-2.93]`、`dq_model=-I*dq_api`、`tau_api=(-I)^T*tau_model`。这里API已应用驱动reversed、bias=0，不重复reversed。模型左右轮前进滚动原生符号分别为`[+1,-1]`，它不替代实际驱动API到模型的逐轮方向核验。轮侧15.8输出单位、多点机构标定和软限位保留现有真实验证状态。

策略50 Hz，训练物理与新反馈PD200 Hz。当前RMCS为200 Hz计算PD、1 kHz持有重发力矩；1 kHz发送或raw bag频率不等于1 kHz新反馈PD。该近似没有同轨迹证据证明硬件等价。

## RMCS侧接入参数与可执行入口

只读核对，本机RMCS仍绑定 `v6_flat_14020` / SHA `4006bf79182e14074f38c3e8f573fe1870fdfeba3fcc0760bb24cc5752161e8a`，普通context实现已匹配。使用者需建立/更新候选profile的SHA、轮动作±9、拟使用的vx/yaw命令范围与组合投影；现有低速profile的vx±0.5/yaw±1不是本候选高速合同。保留已有IMU/机构/soft-limit门控，实际安装版本与源码一起核对。

训练仓库验证包内文件：

```bash
cd /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_speed_candidate_19837_20261005
sha256sum -c SHA256SUMS
```

以下仅为模型文件安装命令，**本次没有执行**；应与RMCS新profile SHA及参数一起使用，之后由原生RMCS构建/安装流程部署实际运行包：

```bash
install -m 0644 \
  /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_speed_candidate_19837_20261005/policy.onnx \
  /home/yukikaze/Documents/workspace/RMCS/rmcs_ws/src/rmcs_rl/models/wheel_leg/policy.onnx
```

包内 `deployment/abi_test_vectors.json` 给出了35组完整传感器快照、观测和动作。用同一快照比较Python和RMCS的35槽、6输出、目标角/轮速、PD模型力矩及API变换，容差 `atol=rtol=1e-5`。`deployment/interface.json`、冻结观测源码、命令参考和控制文件为接入依据；本候选的普通平地接入不发送跳跃请求。

本机CPU可视回放入口（默认键盘低速上限不代表已切换为5 m/s命令）：

```bash
cd /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule
OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/home/yukikaze/.local/lib/compat \
  TMPDIR=/home/yukikaze/.cache/kit-tmp OMNI_KIT_ACCEPT_EULA=YES \
  /home/yukikaze/isaacsim60-venv/bin/python -B scripts/play_v5_grounded.py \
  --onnx models/v6_speed_candidate_19837_20261005/policy.onnx \
  --device cpu --floor-boxes --output reports/v6_speed_manual_replay_20261005
```

包内提供原始PT/封存、合同、原生agent config、完成/进度回执、数值验证、源SHA和本交接文档。重导出使用 `scripts/export_chassis_candidate.py --checkpoint <sealed.pt> --output <不存在的新目录>`，无需Isaac启动；不覆盖既有封存包。
