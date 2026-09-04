# wheeled-biped RL:轮足机器人强化学习训练与部署框架

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

## 硬件与已知限制

- 训练硬件实测参考:RTX 5070 Ti 16GB / 云端 4090,flat+rough 全程约 300–500 卡时
- 已知限制:rough 地形 patch 级难度课程、云台系指令模式、wheel_forward_scan 预瞄未实现;
  ppo_ext 辅助损失的 rsl_rl 侧接线为 WIP(完整实现见 self-impl 分支);
  部署侧 RealBridge 帧字节需与固件对齐
