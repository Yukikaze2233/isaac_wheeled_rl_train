# 架构设计

本文说明训练仓库的三包架构、依赖方向与关键设计决策。行数与文件名以当前代码为准。

## 包结构

```text
src/
├── wheeled_world/               # "世界"层:机器人物理形态
│   ├── assets/__init__.py       #   ArticulationCfg(执行器分组/armature/USD 注入)
│   ├── actuators/m3508_curve.py #   实测曲线电机模型
│   └── terrains/
├── wheeled_tasks/               # "任务"层:环境与课程
│   ├── direct/wheeled_biped/    #   DirectRLEnv 工作流
│   │   ├── env.py               #     观测/动作/奖励/重置 主逻辑(377 行)
│   │   ├── env_cfg.py           #     全部超参(Flat / Rough 两个任务变体)
│   │   └── state_machines/      #     腾空-落地 / 台阶 / 坡面 FSM
│   ├── manager/mdp/             #   可组合环境组件
│   │   ├── commands.py          #     特殊模式指令采样器
│   │   ├── delay.py             #     观测/动作延迟 ring buffer
│   │   ├── events.py            #     域随机化事件
│   │   ├── curriculums.py       #     reward 权重课程 + 辅助力退火
│   │   ├── terrain.py           #     rough 高度场 + per-env flag 映射
│   │   ├── jump_rewards.py      #     跳跃稠密轨迹奖励族
│   │   └── terrain_cmd.py       #     按地形名的指令覆盖
│   └── agents/rsl_rl_ppo_cfg.py #   rsl_rl RunnerCfg(训练超参)
└── wheeled_algo/                # "算法"层
    ├── algorithms/              #   ppo_base + 四个分支
    ├── runners/                 #   runner_class 字符串分发 facade
    ├── utils/exporter.py        #   checkpoint → ONNX
    └── experiments/             #   控制变量实验注册表
```

## 依赖方向(严格单向)

```text
wheeled_world  ←  wheeled_tasks  ←  wheeled_algo  ←  scripts
     ↑                  ↑
   (无依赖)        manager/mdp 不感知 direct;
                    Isaac Lab 只在 direct/ 与 world/ 边界出现
```

- **wheeled_world 不依赖任何上层**,可在无 Isaac Lab 的环境做纯 torch 单元测试(电机曲线模型)。
- **wheeled_tasks 是唯一 import isaaclab 的包**。manager/mdp 下的组件(指令/延迟/课程/状态机)
  刻意保持纯 torch——它们只消费 tensor、不接触 sim 句柄,因此 659 行测试里大部分无需 Isaac Sim。
- **wheeled_algo 与仿真完全解耦**:算法分支只通过 VecEnv 协议(元组 obs + critic extras)
  与环境交互,toy 环境与 Isaac Lab 环境对它不可区分。
- **scripts 是薄壳**:train/play 各 ~100 行,不含任何逻辑,逻辑在三包里。

## 关键设计决策

### 1. 合同即边界(direct env 的 35D 输出)

`env.py::_get_observations` 的 cat 顺序逐段对应部署仓库 `CONTRACT.md` 的索引表。
policy 流(35D)是三方冻结接口;critic 流(43D = 35+3+1+4 DR 回读)可以自由扩展,
因为 critic 不参与部署。改观测必须同时改 CONTRACT.md、env、sim2sim 三处——这是唯一的
多点同步点,其余一切改动都是单点的。

### 2. 算法即插件(钩子式分支)

所有算法分支共享 `algorithms/ppo_base.py::ExtPPOLoop`(采集→GAE→minibatch+adaptive-KL),
分支只覆写三处:

| 钩子 | 用途 | 使用者 |
|---|---|---|
| `act()` 内部的历史编码 | 当前观测 + 记忆(latent/hidden)进 actor | 全部分支 |
| `extra_loss(obs, priv, hist, priv_hist)` | 辅助损失(估计器 MSE / VAE recon+KL / BarlowTwins) | HIM / DreamWaQ / NP3O |
| `surrogate_penalty(obs, hist, priv)` | 进入 PPO surrogate 的约束惩罚(Lagrangian × cost) | NP3O |

新分支 = 继承 HistoryRoller + 写差异 + 注册表加一行,共享循环不碰。

### 3. runner_class 字符串分发

`runners/on_policy_runner_ext.py` 维护 `TRAINER_ALIASES` 名册,
`ExtOnPolicyRunner(env, train_cfg)` 按 `train_cfg["runner_class"]` 构造,
对外暴露与 rsl_rl OnPolicyRunner 相同的 `learn/save/load/get_inference_policy`。
这使 run_experiment(本机 toy)与服务器 Isaac Lab 走同一调用形状。

### 4. 配置与逻辑分离

所有数值(reward 表、DR 范围、延迟步数、弹簧参数、指令包络)都在 env_cfg / registry 数据层。
实验通过 `ExperimentSpec.overrides` 改数据,不改代码;调参史不进 git diff 之外的地方。

### 5. 可测性边界

| 组件 | 测试方式 |
|---|---|
| delay / commands / curriculums / state_machines / terrain flags | 纯 torch 单测(无 sim) |
| 算法分支 | toy env 收敛断言(improvement 阈值) |
| PPO/rsl_rl 栈 | toy VecEnv 冒烟 + checkpoint 往返 |
| 电机曲线 / 跳跃轨迹 / 指令覆盖 | 数值性质断言 |

state_machines 位于 isaaclab 面向的包内但本身纯 torch,测试用 importlib 按文件路径加载绕开包 __init__ 的导入链。

## 与单包方案相比的取舍

三包的成本是 import 路径更长、需要三个 editable install(或统一 src 布局);
收益是:world 层可整体替换(换机器人)、tasks 层可并行开新任务目录(rough/jump/云台各一目录)、
algo 层可独立发版。当前规模(3.6k 行)下收益已兑现——四轮大重构均未跨包破坏。
