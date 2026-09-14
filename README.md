# isaac_wheeled_rl_train

这里放轮腿底盘的强化学习训练代码、运行脚本和实验记录。当前主要想解决几件事：不同高度能站稳，停车后不一直前后晃，转向更快，受到推扰后能恢复。

训练使用 Isaac Sim / Isaac Lab 和官方 RSL-RL PPO。每次实验会保存代码、模型和参数的身份信息，方便把曲线、checkpoint 和回放对应起来。

## 这次 Round4 训什么

本轮已部署到Kaiser，实际代码版本、进程和回收位置见[运行记录](docs/ROUND4_RUNNING.md)。

按目前的决定，**先用旧七刚体串联等效研究模型，从零训练**。新两级四杆的几何预览已经整理出来，但还没有完整动力学参数，本次不把它当成训练模型。

| 设置 | 当前值 |
|---|---|
| 环境与预算 | 1024 个环境，30000 次 PPO 更新，48 步 rollout |
| 网络 | 125 维历史观测 → `[256,128,64]` ELU → 6 动作；critic 29 维 |
| 控制时钟 | 200 Hz 物理，100 Hz 策略 |
| 命令采样 | 30%站立、20%原地旋转、30%直行、20%组合转弯 |
| 速度课程 | 从 vx±0.5 / yaw±1 起步，逐渐到 vx±3 m/s / yaw±6 rad/s |
| 高度 | 0.29–0.32 m，增加端点采样，高度和速度独立重采样 |
| 摩擦 | 启动时分配64个材质桶，约30%环境保留名义参数 |
| 推扰 | 一半回合有推扰，速度增量上限逐渐从0.1增到0.5 m/s |
| 站立抑振 | 保留零平移速度L1，加入车体角速度和弱轮速死区正则 |

组合转弯会考虑附着和轮速余量，不会把最大线速度和最大旋转速度随意叠在一起。这里的速度上限是训练目标，实际表现要看后面的评估。

这段训练是平地鲁棒与机动训练。坡面、台阶、起跳和落地仍是后续任务，路线写在[多场景设计](docs/ROUND4_MULTISCENE_SPEC.md)里。当前实现和参数解释见[本轮设计](docs/ROUND4_SERIAL_FULL.md)。

## 运行环境

当前 `isaac60` 分支使用 Python 3.12、Isaac Sim 6.0.0.1、Isaac Lab `v3.0.0-beta2.patch1`、Torch 2.11.0+cu128、torchvision 0.26.0+cu128 和 RSL-RL 5.5.1。

入口会检查实际版本和资产。安装及环境记录见[本机验证](docs/SIM60_LOCAL_VALIDATION.md)和[Kaiser操作说明](docs/ROUND4_RUNBOOK.md)。旧文档里的 Sim 5.1 / Lab 2.3 是以前实验用的环境。

## 开始一轮训练

本轮配置在 `contracts/own_v40_round4_full.json`，主入口是 `scripts/train_v40.py`。部署到 Kaiser 时，推荐用下面的封装，它会上传指定 commit、启动独立 tmux，并建立结果回收任务：

```bash
mkdir -p reports
python scripts/round4/schedule_serial.py start \
  --commit "$(git rev-parse HEAD)" \
  --ground-usd /home/kaiser/robot-rl-sim60/experiments/a-evaluation-recovery-20260914T070110Z/default_environment.usd \
  --destination "$PWD/reports/round4-$(date +%Y%m%d-%H%M%S)" \
  --host kaiser@192.168.64.234 --ssh-port 2222 \
  --control-path /tmp/opencode/kaiser-training-control \
  --evaluation-entry scripts/round4/evaluate_policy.py \
  --execute
```

该命令针对已配置好的 Kaiser 环境，需要有效的 SSH ControlMaster。具体连接、目录和重试方法见[runbook](docs/ROUND4_RUNBOOK.md)。去掉 `--execute` 可以先看部署计划。

本轮直接进入正式训练，不另起一轮 PPO 快测；CPU 回归和正式启动时的模型、材质、推扰读回检查仍会执行。远端 tmux 训练不依赖本机一直在线，本机关闭后需要恢复结果回收程序。

## 怎么看结果

- TensorBoard 看学习过程；checkpoint 每100次更新保存。
- 结束后导出 ONNX，并校验 checkpoint、模型文件及配套配置。
- 自动评估四档高度站立和单次推扰恢复，保留逐帧CSV、失败计数和恢复时间。详见[评估说明](docs/ROUND4_EVALUATION.md)。
- 本机可用 `scripts/play_v40_onnx.py` 打开原生 GUI。已回收的 Round3-A 修复外观回放可用 `scripts/play_repaired_gui.py`；它仍使用旧等效物理模型。

“跑满迭代”和“行为达标”分开记录。比如非轮净接触力不等于非轮触地，reward 变高也不等于站得更稳。Round3-A 的[训练复盘](docs/ROUND3_A_TRAINING_REVIEW.md)和[最终回放结果](docs/ROUND3_A_EVALUATION_RECOVERY.md)保留了这些差别。

## 目录与提交内容

- `src/wheeled_world`：机器人资产和物理配置。
- `src/wheeled_tasks`：环境、观测、奖励、命令和课程。
- `src/wheeled_algo`：训练接线、导出和运行记录。
- `contracts`：每轮使用的参数与接口约定。
- `scripts`：训练、回放、部署和回收入口。
- `tests`：CPU回归；`docs`：方案和精简实验结果。

`reports/`、`runs_v40/`、日志、模型权重、仿真缓存和本地Python环境不提交。`assets/`下的必要机器人资产、合同和`docs/evidence/`下的小型证据文件保留；不按`.usd`或`.stl`扩展名一刀切忽略资产。

<details>
<summary>早期 V3.x / 并联腿实验记录</summary>

下面保留的是较早的项目说明，包含过时接口和未复核的历史表述。启动当前V4任务请以上面的入口和对应实验记录为准。

轮足(Wheeled-biped)机器人端到端运动控制的训练与部署双仓库。训练端基于
**Isaac Sim + Isaac Lab + rsl_rl**,部署端基于 **ROS2 + ros2_control + ONNX Runtime**,
两仓以一份冻结的策略合同(35D 观测 → 6D 动作)为唯一接口权威。

核心设计:训练中保留腿部并联结构与气弹簧的物理形态,通过单一 policy 实现
end-to-end 的多任务盲走控制(平移 / 小陀螺 / 冲刺 / 变高),sim2real 依赖
合同对齐 + 域随机化 + 延迟建模,不做补偿策略。

```text
训练:Isaac Lab 200Hz 物理 × 4 ──► 50Hz 策略 ──► PPO(adaptive-KL)
部署:500Hz PD 闭环 ──► 50Hz ONNX 推理 ──► MuJoCo sim2sim / 真机串口
```

## 架构说明

三包架构,依赖严格单向(完整文件级树见 [docs/project_tree.md](docs/project_tree.md)):

| 包 | 职责 | 关键内容 |
|---|---|---|
| `wheeled_world` | 机器人物理形态 | ArticulationCfg(并联腿+气弹簧+armature 折算)、实测曲线电机模型 |
| `wheeled_tasks` | 环境与课程 | DirectRLEnv(35D/43D/6D)、腾空-落地/台阶/坡面状态机、特殊模式指令桶、延迟 ring、域随机化、rough 高度场、跳跃稠密轨迹族 |
| `wheeled_algo` | 算法插件层 | 共享 PPO 循环 + 五个分支、runner_class 分发、ONNX exporter、实验注册表 |

两条训练通道并存:

- **rsl_rl 集成(main 主路径)**:分支实现为 rsl_rl ActorCritic 子类,经
  `class_name` 注入由官方 OnPolicyRunner 训练——与生态工具(分布式/日志/续训)天然兼容;
- **自研 ExtPPOLoop(`self-impl` 分支)**:紧凑的采集→GAE→更新循环 +
  extra_loss/surrogate_penalty 两个钩子,五分支各 ~120 行,本机 CPU 可收敛实证——
  教学与快速迭代用。

```text
一个控制步的数据流(训练端)
policy action ─► 解码(腿位置目标+轮速度目标)─► 动作延迟(20-60ms)
  ─► 200Hz 物理内环(弹簧施力/PD)─► 观测延迟(20-80ms)+噪声
  ─► 35D 观测拼装 ─► reward 表(exp 核跟踪+惩罚+跳跃轨迹族)─► GAE/PPO 更新
```

## 创新点

相对已验证的赛季开源实践,本仓库在**工程形态**上做了六件事:

1. **双训练通道**。赛季方案只有一条与 rsl_rl 深耦合的路径;本仓库把"算法机制"
   (自研 ExtPPOLoop,可读可改可断言)与"生产形态"(rsl_rl 官方 Runner,可上服务器
   全规模)分离为两条分支,同一份环境合同、同一组分支语义,学习路径与产出路径互不污染。

2. **实验体系产品化**。控制变量实验从"散落的 cfg 副本"升级为注册表
   (`ExperimentSpec`)+ 一键运行 + 断点续训 + `metrics.jsonl` 落盘 + 跨实验对比表/CSV。
   一组对比实验之间强制单变量(exp003 vs exp004 仅 cost limit 不同),实验史可复现。

3. **测试即规格**。7 套件 / 659 行纯 torch 回归(赛季方案为 0 行测试),三层断言:
   行为断言(delay lag 语义、FSM 转换)、收敛断言(每个算法分支必须证明 toy 收敛)、
   性质断言(轨迹边界条件、扭矩限幅 droop、桶互斥)。全部组件无需 Isaac Sim 即可验证。

4. **合同工程化**。35D→6D 冻结合同独立成文(CONTRACT.md)并配校验器
   (名称/shape/dtype/零输入前向四道检查),合同变更走五处同步流程;
   critic 侧特权观测(43D)可自由扩展而不触碰部署面。

5. **延迟与指令的向量化工程**。逐 env 延迟缓冲为预分配 ring(无每步分配)、
   特殊模式指令桶与按地形指令覆盖全向量化(无逐 env Python 循环)——
   4096 env 规模下避免隐式同步与 O(N) 解释器循环两类隐形税。

6. **机制文档化**。每个机制的语义、调参入口、常见坑(migration.md 五大坑按踩中概率
   排序、sim2real 排查表按症状→首查/次查组织)沉淀为 10 篇 docs,而不是散在注释里。

## 核心机制速览

| 机制 | 要点 | 详见 |
|---|---|---|
| 观测合同 | 35D = 指令3+高度1+IMU6+关节12+上帧动作6+模式标志7;critic 另含特权流+DR 回读 | [CONTRACT](../isaac_wheeled_rl_deploy/CONTRACT.md) |
| Reward | exp 核速度/高度跟踪 + 力矩/加速度/动作率惩罚 + 跳跃全轨迹族;稀疏奖励与裸辅助力是已知陷阱 | docs/environment.md |
| 指令课程 | spin(2π–4.5π)/dash(2–3 m/s) 按迭代数分批启用,桶互斥 | docs/environment.md |
| 域随机化 | 质量/COM/材质/PD/摩擦,startup+reset(720 步门控)两档,采样值回读进 critic | docs/environment.md |
| 延迟 | obs 20–80ms / act 20–60ms 逐 env 重采样;不足则实机抖动,过大则定点静差 | ../isaac_wheeled_rl_deploy/docs/timing.md |
| 算法分支 | HIM(一步特权估计)/ DreamWaQ(VAE)/ NP3O(约束)/ GRU(记忆)/ FrameStack(对照) | docs/algorithms.md |
| 辨识 | real2sim:真机 bag 回放 vs MuJoCo 同轨迹,曲线对齐;轮电机直接辨识 | ../isaac_wheeled_rl_deploy/docs/sim2real.md |

## 快速开始

```bash
# 本机验证(无 Isaac Sim):CPU torch + rsl-rl-lib==2.3.3
PYTHONPATH=src python tests/test_mdp.py
PYTHONPATH=src python tests/test_rsl_rl_smoke.py
PYTHONPATH=src python tests/test_algorithms.py

# 实验(本机 toy 模式)
python scripts/run_experiment.py --list
python scripts/run_experiment.py --exp exp001_him_latent16 --iterations 60
python scripts/compare_experiments.py runs/*

# 服务器训练(Isaac Lab 环境)
./isaaclab.sh -p scripts/train.py --task WheeledBiped-Flat-v0 \
    --num_envs 4096 --max_iterations 20000 --headless
./isaaclab.sh -p scripts/export_onnx.py --checkpoint logs/*/model_final.pt --output policy.onnx
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 三包架构、依赖方向、设计决策 |
| [docs/environment.md](docs/environment.md) | 环境机制逐项详解 |
| [docs/algorithms.md](docs/algorithms.md) | 算法分支指南 |
| [docs/experiments.md](docs/experiments.md) | 实验工作流 |
| [docs/migration.md](docs/migration.md) | 自有机器人迁移清单 |
| [docs/project_tree.md](docs/project_tree.md) | 两仓库文件级架构树 |
| [docs/server_setup.md](docs/server_setup.md) | 服务器训练环境搭建(版本矩阵/验证/云端工作流) |

## 硬件与已知限制

- 训练硬件实测参考:RTX 5070 Ti 16GB / 云端 4090,flat+rough 全程约 300–500 卡时
- 已知限制:rough 地形 patch 级难度课程、云台系指令模式、wheel_forward_scan 预瞄未实现;
  辅助损失的 rsl_rl 侧接线规划中(完整实现见 self-impl 分支的 ExtTrainer 路径);
  部署侧 RealBridge 帧字节需与固件对齐

</details>
