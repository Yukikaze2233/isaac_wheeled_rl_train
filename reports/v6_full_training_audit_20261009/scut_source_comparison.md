# 华南虎源码与 V26 训练合同核对

日期：2026-10-09。此文件仅做本地源码与冻结配置审计，不表示 V26 已部署或相应能力已学会。远端实际执行状态、模型质量由同目录运行与评测报告给出。

核对基准为华南虎官方仓库提交 `b8ff79f3df855faf9dc92f4a282bd80c42649466`。本机完整克隆 HEAD 已核验一致；另有本项目 `references/scut_curriculum_b8ff79f3/manifest.json` 的固定 SHA。读取三条实际保存任务线：`flat_and_rotation/2026-07-29_23-20-03`、`rough_dash/2026-07-19_19-15-07`、`rough_rotation_stair/2026-07-30_09-16-27`。源码可选能力与这些保存配置实际启用项分开记录。

机器可读证据：本目录 `scut_saved_config_extract.json`、`local_v26_comparison_extract.json`。V26 对照以 `contracts/v6_integrated_runtime_recipe_v26.json` 的 component patch/runtime 为准；裸 worker 内的旧 origin 不代表集成运行的实际时钟。

## 1. 需要先纠正的历史说法

| 曾出现的说法 | 本次核对结论 |
|---|---|
| 华南虎没有阶段、相位、奖励门控 | 有预训练继承、全局迭代技能开关、腾空/落地相位奖励和速度奖励乘性门。不能以“无相位”解释它不出现奖励漏洞。 |
| `flat_orientation_x/y_v=-2` 是横滚/俯仰角速度惩罚 | 它是随前向命令速度变化的 projected-gravity 姿态平方项；真正的平面角速度平方项为 `ang_vel_xy=-0.05`。 |
| 三条线都激活 `stand_still_lin_vel=-2` | flat 为 −2，rough_rotation 为 −1；所核验 rough_dash 保存配置没有该 reward 项，且 deadzone 开关为 false。不能把它作为 rough_dash 有效驻停支付的证据。 |
| `rel_standing_envs=.1` 表示固定 10% 环境永久站立 | 它是命令重采样时的概率，special mode/地形覆盖还可改写命令；不等于档位级保底份额。 |
| 其延迟是 50 Hz 帧上的 10–40 ms / 10–30 ms | delay buffer 在物理子步推进，`torch.randint` 上界不含。dt=.005 时，所核验观测 [1,4] 实际为 5/10/15 ms，动作 [1,3] 为 5/10 ms。回合内固定 lag，不是每个物理步重采的延迟抖动。 |
| 华南虎 actor 接收 raycast 高度/地形，而我们只有单帧 35D | 三份配置 actor 也为单帧 35D。raycast 地面高度用于 reward/状态机等，不在 actor 拼接的地形观测里；actor 的高度输入是目标高度。 |
| 几千更新或 20k 上限说明课程已充分/优于我们 | 上限不是实跑量、实跑量不是通过率；不同环境数、有效 PPO 样本、出生难度与任务分布不可只比更新数。 |

以上通过[共同环境源码](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py)、[flat 保存配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/flat_and_rotation/2026-07-29_23-20-03/params/env.yaml)、[rough 保存配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml)交叉验证。

## 2. 课程的可复用部分与 V26 对应

| 机制 | 华南虎实际配置/源码 | V26 对应与差异 |
|---|---|---|
| 持续指令曝光 | 普通速度连续采样，命令保持 5–15 s；普通 vx ±2.7 m/s、yaw ±2π rad/s、h=.20–.42 m | 已有 SCUT 普通命令映射，保留本机构 .23–.43 m 域与原生高速/锚点。映射组名如 `forward_05` 已不一定表示固定 .5 m/s，不能用名字推算跟踪误差。 |
| 全局迭代技能解锁 | flat：dash 2000、spin_low 3000、spin_mid 4000；rough 继承平地权重后从 0 开启相应模式 | 已有 actor 时钟难度阶梯。speed 0/120/240/360 即逐步开到 2/3/4/5 m/s 与最高 3 rps，同时扩展高度；比 flat 源配置快很多。是否合适必须结合继承 checkpoint 的能力与实际有效样本判断。 |
| 地形×命令绑定 | 大台阶限定直进/退与高度，关闭部分特殊模式和非标准出生；小台阶、坡、粗糙面各有命令覆盖 | 已有 terrain-bound 组。需要按实际发出的命令×高度×地形×相位曝光审计，不能只统计 pool 的总份额。 |
| 常驻基础能力 | 同一策略里普通移动、站立抽样、spin/dash 共存；rough 从 flat 权重初始化 | V26 顺序 speed→terrain→jump→stepjump，后阶段保留先前运动、驻停、恢复练习；是一个 actor 的混采巩固。串行本身不会自动保证不遗忘，需比较固定 checkpoint 回放。 |
| 非标准出生 | 预设腿姿态随机出生、约 .2 s 零力矩，随后策略接管；事件扰动常驻 | 本项目已有原生非标准腿姿态、零力矩接管、完整脚本恢复链。后者不是可从其源码直接复制的同一任务。 |
| 主动跳跃辅助与高度课程 | 源码支持概率×力度双通道衰减、global iteration 高度范围课程 | V26 已有前 300 actor 更新辅助退火及高度 bin 概率随 actor 更新重分配；不应再次列作缺失。 |

技能迭代接口见[V14 环境源码](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env.py)。命令与地形绑定见[命令模块](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/manager/mdp/isaaclab/commands.py)及[地形命令管理器](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/manager/mdp/terrain/terrain_command_manager.py)。

### 台阶覆盖不能只理解为 5–40 mm

所核验 rough_rotation_stair 配置含 5–15 mm、25–40 mm 小阶、10–30 mm 短障碍、坡和粗糙面；rough_dash 同时含 150–400 mm 大阶、较长踏面、下降断崖等。小台阶适合作为初始支撑/接触课程，但不替代本项目的 200+250 mm 双级任务。

rough_dash 的高阶组把 |vx| 限在 1.5–2.7、低阶组 .5–1.5 m/s，h=.22–.34 m；台阶中强制很小 yaw 并关闭 special modes。它展示的是“先保证该地形得到有用经验”，不是任意旋转/跳跃与任意台阶的全笛卡尔积。配置见[rough_rotation_stair](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_rotation_stair/2026-07-30_09-16-27/params/env.yaml)。

## 3. Reward：已有项、真实缺口和归因边界

华南虎同时使用正指数核与负平方误差，姿态项和动作平滑项并存。三份配置并未证实 reward 核宽随迭代“先宽后紧”；主动起跳状态机在这三份保存配置中都关闭，rough_dash 开启的是 airborne/landing。不能把可选 `JumpTakeoff` 的辅助机制称为这些交付 run 已实际验证的主动跳跃成绩。

源码 `jump_takeoff._sample_peak_height_from_bins` 使用 range 与 prob/probability；没有读 weight 字段，也没有这里自动按通过率重分配 bin 的机制。概率×力度辅助退火确实有源码支持。见[主动起跳状态机](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/state_machines/jump_takeoff.py)。

V26 的实际对照：

| 项 | 已有实现 | 本次判定 |
|---|---|---|
| 低速驻停 | L1 平面速度 −2、threshold .1；dense 宽核 .1/fine .04 m/s，权重 1.5/.75 | 不是没有漂移惩罚。旧 `precision_tracking.stationary_width=.03` 在 dense 路径启用时不参与支付，不能再归因为旧核唯一病因。 |
| yaw/平移/高度精度 | dense 宽细两核同时启用 | 不是没有 yaw 精修信号，也不是整体只有紧核。应核查实际 mask、支付、有效样本和均值策略。 |
| 普通姿态 | 原生姿态项、pitch/roll floor、参考姿态项 | 不能把 SCUT 的 `_v` 误译成角速度后重复新增一套错误项。 |
| 主动跳跃核宽 | `full_tasks.py` dense 高度 .035 m、垂速 .35 m/s、tuck .04 m；reference jump 等也为固定核 | 未找到 V26 actor 时钟的跳跃核宽退火。已存在 bin/assist 难度课程与回合内相位，它们不是核宽课程。 |
| performance curriculum | 源码有依 episode 结果改变 height width 的模块 | V26 component patch 未启用；不能当作本轮已生效的宽→紧课程。 |
| 落地奖励 | V26 原生 contact-triggered landing、零跟踪基线与明确窗口 | 有修复，仍需检查真正腾空/接触后的支付；不再复述旧 11 mm 恢复白拿奖励为当前未修事实。 |

本地证据入口：`src/wheeled_tasks/chassis/env.py` dense/precision 分支、`rewards.py`、`full_tasks.py:149`、`performance_curriculum.py`、`step_assist.py`、`jump_landing.py`。

指数核的敏感区是 |e| 与宽度同量级；大误差时正核接近零、紧核对区分更大误差的作用很小。因此先大误差纠正再精调有道理。但 PPO 不通过物理 reward 对状态的导数反传；“核饱和”只能说明反馈区分度不足，不能单独证明 actor 梯度为零。平方/L1/Huber、实际支付尺度、探索与有效采样都必须一起看。

建议把跳跃宽→紧作为**待冻结的一项候选**：初期在有效腾空/收腿窗口使用覆盖实际误差范围的宽核，随后按全局 actor 时钟收紧，保留大误差仍能区分的负项；不修改完成定义，也不把辅助期或仅几何收腿算作完整跳跃。先用现有核误差分布与支付探针选宽度，不直接照抄数字。

## 4. 延迟、随机化与观测

V26 已有毫秒级 physical signal delay，75% lanes、obs 覆盖至 80 ms/action 至 60 ms，并含 ±5 ms 抖动与 actor 0–1000 渐进窗口，范围比上述 SCUT 配置宽。它是鲁棒性分布，不能称为真实 USB 延迟的精确辨识结果；需区分策略/观测滞后、命令 transport、执行器一阶响应，避免重复计同一延迟。

SCUT startup COM 为 x±4 cm、y/z±2 cm，flat/rotation base mass .9–1.3，rough_dash 为 1.0–1.4，腿/轮质量 .9–1.1；接触摩擦、推扰和 actuator gains 另有随机化。V26 当前 dynamics 配置 base .9–1.1，COM x±1 cm/y±.5 cm/z=0，腿/轮 mass/inertia scale 固定 1。**此前“双方质心范围几乎一致”的说法不再成立**。是否扩大应依据本机实际负载与误差并按难度渐进，避免同时把高速度、极端高度、长延迟、大 COM 全部开满。

单帧 actor 缺少真实平面速度在滑移、执行延迟下构成部分可观测问题，两边都存在。SCUT raycast 并不自动解决 actor 的这项问题；我方历史编码器/速度估计器是可比较方案，不应先判 MLP 容量不足。height command、由腿几何得到的本体高度、地面相对高度必须分别记录，尤其腾空和台阶高度变化时不能当成同一量。

## 5. 继承、时长与剩余课程审计

保存 agent 配置均为 4096 环境、24 policy steps/update、20k 上限、每 500 更新保存；rough_dash 为 256/128/64 MLP，flat/rotation 为 128/64/32，不能说网络都完全相同。rough_dash 明确从 flat model_5500 初始化，rough_rotation 从 flat model_8000 初始化；`resume_training:false` 在其训练脚本意味着权重初始化/finetune，而非保持旧优化器与迭代的原样续跑。见[训练脚本](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/scripts/rsl_rl/train.py)和[rough agent 配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/agent.yaml)。多个保存任务目录不能证明实车一定切多个 ONNX；每个策略内部确有多模式混采。

V26 课程不是缺“静止/延迟/扰动/台阶/跳跃”这些名字，而更可能缺充分而可审计的**联合曝光与交接质量**。优先改进顺序：

1. 查有效 PPO 样本、实际执行命令与出生/脚本交接状态。区分“请求运动”与门释放后真的运动，区分辅助脚本步与可学步。
2. 用同 checkpoint、同种子、同物理/随机化对比均值与采样策略；漂移、yaw 角速度、速度偏置、接稳连续时长分别报告。训练 reward 不替代能力结论。
3. 已有基础复习不能取消；按低/中/高姿态×静止/加减速/正反旋转×flat/rough 的有效样本设可见配额，尤其 release-stop 和临界高度。固定 checkpoint 旁路评测不阻断 PPO。
4. 针对恢复交接先校验脚本完成瞬间的 pitch/roll、角速度、腿/轮速度，再渐进增加原生非标准姿态、长延迟和推扰。不要用“再加处罚”掩盖交接初态与学习曝光缺口。
5. 台阶继续先小阶/坡/粗糙面支撑能力，再大阶、正常搭台收腿与主动跳阶分别采样/验收。补审 descending step、台阶边缘、窄障碍和停在阶上等组合；本次静态配置审计不宣称这些能力已经通过。
6. 主动跳跃先把离地→收腿→着地→接稳链的各段误差/支付/曝光报告齐，再冻结核宽退火或采样调整。前后保持同 actor 的基础练习，不自动记为已学会全部能力。

这些是有证据的修复方向或待检验假设，不是无需评估即可热改的运行合同。当前执行回执优先，避免审计过程中重复启动训练。
