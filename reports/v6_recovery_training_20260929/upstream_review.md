# V6 恢复训练：华南虎、复旦源码复核与遗漏场景

日期：2026-09-29。只读审计；本文件不证明任何新场景已经接入训练或通过 V6 物理测试。用户要求本轮先完成训练侧，RMCS 修改、构建或部署不作为训练前置。网络保持 actor 35 / critic 81、原 MLP 和六维动作。

## 1. 上游实际启用的内容

核对以下本地冻结克隆，对所列训练文件的 `git status --porcelain` 输出为空；同时打开官方 GitHub 的固定提交文件确认来源。以下默认值只属于指明的类，不能扩大到其全部已发布模型或比赛实机。

|仓库|固定身份|本地路径|
|---|---|---|
|华南虎训练|`b8ff79f3df855faf9dc92f4a282bd80c42649466`|`/home/yukikaze/Documents/workspace/robot_rl/scut-wheeled-legged-rl`|
|华南虎部署|`daa34f54d56cab91b3989d8152a7ce7b61092994`|`/home/yukikaze/Documents/workspace/robot_rl/scut-wheelbipe-ros2-sim2sim`|
|复旦训练|`8204e853dfd2ed06d85a322e1a998c3d20a3be2c`|`/home/yukikaze/Documents/workspace/robot_rl/fudan-wheel-leg`|

### 华南虎：非标准腿形 reset 是启用机制，recover 模式不是

普通 `WheelbipeV14FlatEnvCfg` 继承 `use_leg_random_start=True`，明确 `use_predefined_leg_random_start=True`；地面 reset 在两类腿形和普通状态间混采。前向腿形概率 0.3，后向腿形概率 0.2；进入该分支后两条腿独立选择类型，因此存在前后不对称。0.2 s 窗口不施加主动驱动力矩，然后恢复策略控制。重力、接触及启用的气簧仍存在。[配置定义](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/cfg_utils.py#L198-L225)

精确执行位置（均相对训练根目录）：

- `source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env_cfg.py:421`：基类腿形随机化打开。
- `source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py:584-589`：普通 V14 开启该 reset，状态机总开关为 False。
- `.../wheelbipe25_v3/env.py:5353-5530`：抽样腿长/腿角、地面/空中子分布，按腿独立采样地面类型，再经 IK 初始化。
- `.../wheelbipe25_v3/env.py:2847-2924`：零主动驱动力窗口及气簧施力。
- `.../wheelbipe25_v3/env.py:2811-2815`：每步置 `finish_init/start_reset=True`。`start_reset_time=1.5` 不能解释为 1.5 s 无条件关闭终止；也用于临时命令覆盖。
- `.../wheelbipe25_v3/env.py:1789-1799`：`recover_state` 恒 False；对应观测槽存在不代表恢复 FSM 生效。
- `.../wheelbipe25_v3/env.py:2818-2824,3979-4003`：专用 standup 阶段和奖励是注释代码。

普通 V14 有 40° roll/pitch 原始终止条件，连续持续计数 20 个策略步后终止；数值异常立即终止。它允许短暂失姿经验，不能据此声称覆盖 ±90°侧翻或倒扣翻身。终止详见 `wheelbipe25_v3/env.py:4863-4895,4911-5030`、默认角度见 `wheelbipe25_v3/env_cfg.py:178-179`、V14 开关见 `wheelbipe_V14/env_cfg.py:588-590`。

`FlatEnvCfg_v1` 另启用空中 reset 和 airborne FSM：roll ±0.1 rad、pitch ±0.2 rad，及速度/腿形采样，属于落地恢复练习，不是大角度倒地矩阵。见 `wheelbipe_V14/env_cfg.py:1488-1550`。普通类 `curriculum=None`，而变体仍可包含迭代调度，不能把类存在当成默认执行。

### 华南虎：PREPARE 是关节限速插值，不是完整多姿态翻身脚本

`StatePrepare::enter` 读取四个腿关节当前值，`run` 限制参考每次增加 `max_vel * dt`，然后计算 PD；未调用策略，也没有依据前趴/后仰/侧躺选择接触路径。默认目标四轴为 0、Kp 80 / Kd 2、参考速度 1 rad/s，属于对方资产参数，不能移植为 V6 参数。[PREPARE 源码](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/daa34f54d56cab91b3989d8152a7ce7b61092994/src/controllers/template_ros2_controller/src/fsm/states/state_prepare.cpp#L22-L62)

本地精确位置：`src/controllers/template_ros2_controller/src/fsm/states/state_prepare.cpp:22-62`，默认值在 `config/template_ros2_controller_parameters.yaml:125-143`。PREPARE→RL 的分工值得参考，但本项目需要自己的经过物理验证的翻身及捕获轨迹，不能写成“华南虎已经验证本机十四姿态插值自起”。

### 复旦：当前 plane 默认不是任意姿态恢复 reset

注册任务使用 `WheelLeggedCfg()`：固定初始 root quaternion `[0,0,0,1]`、根高度 0.1 m、固定默认腿角；reset 给六轴根速度各采样 ±0.5，关节速度清零。`_reset_dofs` 的注释写随机比例，但实际赋值是默认角，不能按注释报告。连续失姿超过 1 s 才终止；这给瞬态恢复时间，仍不等于侧躺/倒扣初态覆盖。[reset 与终止源码](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot.py#L765-L819)

精确执行位置：

- `plane/wheel_legged_gym/envs/__init__.py:49-51`：注册实际类。
- `plane/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py:42-88`：实际初态、PD、资产。
- `plane/wheel_legged_gym/envs/base/legged_robot.py:209-233,765-819`：终止与 reset。
- `plane/wheel_legged_gym/envs/base/legged_robot_config.py:48,106-110`：1 s 失姿计数及正立默认 quaternion。
- `plane/logs/wheel_legged/recover_from_lied_on_ground/` 虽保留历史配置和完整源码，其中该版本 reset 仍为默认关节/固定 quaternion，不应把目录名当任意倒地训练证据。

可借鉴其窄/宽速度跟踪组合、质量/惯量/COM、摩擦、控制增益及电机强度随机化，但不能拷贝虚拟串联膝坐标、对方控制常数或接触假设。本轮不引入复旦的历史速度编码器，保持网络结构约束。

### 对奖励的正确借鉴

华南虎、复旦都有稠密姿态、速度、站高和控制代价，但“采用了 exp”不等于局部 PPO 优势充足，也不证明全倒地奖励有效。复旦 `_reward_orientation` 为 projected-gravity 的 xy 平方和（`legged_robot.py:1727-1730`），正立与完全倒扣同为零。大角度恢复若额外设计姿态参考，须使用有符号朝上误差或角距离；不能只照抄该项。接管后的站高/姿态/角速广域捕获项与精核应共存，近正立才增加精度要求。

## 2. 与本轮训练的关系

`c15acfc / v6_step_curriculum_v1` 已含普通上阶、请求跳阶、20k 并行、真实轮响应、通信扰动及基础受扰初态；它的 `R0` 仍标 `separate_script_validation_not_claimed_RL_self_righting`，不包含完整倒地轨迹或接管状态训练。该结论是审计时源码状态，后续实现须以新合同和回执为准。

用户指定的 `V6_PAIR_REAL2SIM_TRAJECTORY_20260928.md` 是架空整腿辨识协议，不能替代车体接触/翻身证据。历史地面资料主入口：

- `/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train/docs/V5_MOTOR_ENVELOPE_RECOVERY_TIMING_20260927.md`
- `/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train/docs/V5_FLAT_SELF_RIGHTING_TRAJECTORY_20260927.md`
- `/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train/docs/V5_SELF_RIGHTING_PRONE_STEP_20260927.md`
- `/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train/reports/v5_24v_split_zero105_full14x5_20260927/report.json`

历史报告是 V5 资产、旧策略、200 Hz 反馈/400 Hz 物理与 40–110° 膝域；本轮 V6 名义机械止挡 105°、1 kHz PD/物理、新坐标，必须重跑，不能继承通过数。旧 105°设零不代表旧机械域已是 105°。历史资料还区分“曾稳定 RL”和“最后持续稳定”，台阶顶必须仍留在目标上层，不能滚落后在平地站稳而计成功。

训练实现可以先直接在当前 Isaac 环境适配已有恢复参考，不等待 RMCS 代码修改。必须贯穿同一重力/气簧/闭链、PD/轮 P、扭矩包络、响应模型及通信执行链，不能用 `write_root_state` 翻正或绕开执行器直接施加脚本力矩。腿响应仍未 qualified；不能把气簧拟合或 Kp/Kd 整定误写为全构型执行器辨识完成。

## 3. 最小必需场景与分项归因

### 平地核心矩阵：重放历史十四类，并增加两类壳体趴地

历史十四类的机器名为：

```text
upright
crouched
upright_release
crouched_release
left_side_90
right_side_90
pitch_forward_45
pitch_backward_45
front_down_90
back_down_90
front_down_left_forward_right_back_90
front_down_left_back_right_forward_90
back_down_left_forward_right_back_90
back_down_left_back_right_forward_90
```

此外 `prone_belly_legs_above`、`prone_back_legs_above` 不应被前后 90°释放简单替代：壳体已承重、腿位于壳体上方、气簧/止挡预载不同。正立无释放类承担“不能误触发翻身”的阴性对照。每家族先名义，后持出角速/腿不对称/朝向与动力学组合，不能随机拆同一条轨迹相邻帧当持出数据。

### 地形矩阵：至少保留五类历史异常并扩展双级出口

1. 高低轮跨 0.20 m 支撑面。
2. 壳体已搭上台、轮在下层。
3. 壳体/轮靠台阶立面，法向接触不能当垂向承重。
4. 上层顶面正立，必须留在目标顶面接稳。
5. 上层顶面趴地，不能掉到下层后计成功。
6. .20 m→再升 .25 m 的双级第一/第二落点，首踏面 .80 m；分别覆盖正常搭台收腿失败出口与跳阶失衡出口。双级总顶面 .45 m，不把两级当 .20/.25 m 绝对高度。

后续扩展应从实际轨迹中采集斜向上阶、单轮先触地、边缘半轮悬空、落地反弹和后退边缘状态；不能独立均匀抽 q/root pose 生成穿透、不闭链或不可能的支撑组合。持续使能下主动越过边缘与“先失能释放 1 s 再恢复”是不同类别，须独立标记。

### 统一但分层的成绩

每次完整试验记录：起始类别 → 脚本是否到达可接管域 → 混合是否完成 → 首次持续稳定 → 末尾持续稳定 → 继续运动/转向/跳跃是否成功。记录失败发生的环节，不能将脚本捕获失败归因于 RL，不将精度尚未达标表述为完全不会。

最终完整链成功率的分母包含脚本失败样本；同时报告“条件于脚本成功”的 RL 接稳率，以便定位问题。短暂进入 RL、达到目标关节角、曾稳定、末尾稳定是四个不同字段。机械/数值失效不能被恢复宽限吞掉。

## 4. 优先补入的遗漏情况

以下是本项目设计建议，不是上游已经证明有效的参数或功能。

|优先级|遗漏/风险|最小训练或验证动作|
|---|---|---|
|P0|髋共同回转圈数、编码器跨 ±π/±2π、PD 参考的圈数不一致|脚本出口记录共同圈数与所有主动/被动 q。要么证明出口回到现 actor 可表示的 canonical 分支，要么显式版本化最近等效圈数解码；禁止仅 wrap 观测而让未 wrap PD 误差突跳|
|P0|恢复时重置了响应/通信/上一 action，形成不真实的干净接管|通过在线脚本前史进入 RL；如导入快照，须恢复控制缓存与事件年龄并验证短段重放等价性，不回退整个 batch 的全局时钟|
|P0|同一腿姿态在地面、悬空、壳体承托和异高轮支撑中含义不同|FK 高度只作条件估计；actor 不读取真实接触或地形标签。支撑估计必须有单轮、立面、无承重等反例，评分用原生接触真值|
|P0|恢复动作混进 PPO 但 log-prob 对不上实际 sampled action|脚本优化/接管优化边界明确；对混合段可将混合作为环境控制变换并保留 actor 原始采样及 log-prob，但必须审计观测/相位的可部署性。首版从真实混合结束开始 PPO 更易审计|
|P0|可接管域只是姿态阈值，出口关节/角速已超现 actor 的可控范围|导入前核对动作目标域、关节闭链分支、残余角动量与支撑。拒绝未验证出口，不用放宽普通 fall 条件冒充覆盖|
|P1|同类只有某一翻转方向/左侧通过，镜像被总均值遮蔽|左右、前后劈叉独立配额与门，重点复习历史最弱类别，不把一侧模型代称双侧能力|
|P1|混合首拍、1000 Hz反馈年龄和短丢包相位正好重合|同合同抖动/丢包贯穿脚本—混合—RL；专测旧 torque 保持横跨混合边界，记录实际输出冲量/峰值而不只 target 连续性|
|P1|自动支撑估计被落地大加速度/机身快速旋转污染|名义与传感器扰动下比较预计支撑和真实承重，保留误判案例；不把仿真精确姿态当 IMU 姿态滤波实测性能|
|P1|恢复后抢先执行旧 forward/jump 请求，形成二次失姿|评测持续请求、重复、取消及恢复中到来的请求；运动解锁后再执行明确保留/丢弃语义，不能复用过期 jump 时钟|
|P1|恢复在平地成功，但台阶顶掉落被记为成功|按目标平台/实际垂向支撑验收；碰到竖面不计支撑，掉到下层单独记失败，即便最终正立|
|P1|持续降级与即时 IID 噪声混淆|质量/COM、气簧强度、单侧降额/低摩擦在同一整链保持；训练见习与极端能力验收分开，不要求失去执行自由度仍完成全部技能|
|P1|单次自起成功后重复两次却失败，控制缓存/力矩历史累积|S6 包含恢复→移动→再扰动→恢复链；轮角、圈数、控制历史不通过隐含全 reset 清零|
|P2|温度、老化、供电跌落、IMU偏置未实测|明确标为工程先验/未覆盖。可用有界持续力矩折扣与 bias 压力测试，但不能写成完成温度老化辨识|

无需新增网络即可先纳入上述真实接管状态、现有可见反馈与命令分布。若出现相同 35D 可见状态需要互相矛盾的动作目标，先修正 supervisor 的控制分工和可接管域，不悄悄给 actor 注入恢复库标签或世界真值；网络结构约束并不允许隐藏语义变化。

## 5. 课程建议与真实性边界

- S1/S2 保留基础能力训练；自起轨迹物理回放和出口生成可并行，不计 PPO 更新，也不把这些回放称为策略已经学会自起。
- S3 加平地真实出口接稳与边缘姿态捕获，和原推扰复习共存；按姿态家族做固定最小配额。
- S4/S5 的对应地形/跳跃训练加入其真实失败出口；不将“脚本不能到达”的高低轮出口直接放入 RL 并假定其可学。
- S6 验证完整插值/捕获→混合→稳定保持→运动/转向/再次技能链，以及重复、取消、短丢包和持出动力学组合。
- 对已通过姿态保留复习底线；难度时钟绑定全局更新与合同，不按某次失败随机退回隐藏课程等级。
- 配额、阈值与随机化取值须由本机轨迹/现有合同冻结；本报告不给未经验证的“更大胆”力矩或上游尺寸，避免把建议伪装成已辨识参数。

交付声明必须同时列：实际启用的 reset/脚本/接管分布、物理回放覆盖、策略训练采样、分姿态验收。仅增加配置字段、仅通过单元测试或仅生成计划不能称完整恢复功能已交付。
