# isaac_wheeled_rl_train

轮足(Wheeled-biped)强化学习训练框架,采用三包架构
(world / tasks / algo),Isaac Sim + Isaac Lab + rsl_rl 同栈。
所有代码为自研实现(架构与参数参照已验证的赛季实践),每层均带本机可跑的回归测试。

## 架构总览

| 包 | 内容 |
|---|---|
| `wheeled_world/` | 资产与物理:assets(ArticulationCfg)、actuators(实测曲线电机模型)、terrains |
| `wheeled_tasks/` | 环境与课程:direct/wheeled_biped(env + cfg + state_machines)、manager/mdp(commands/delay/events/curriculums/terrain)、agents(RunnerCfg) |
| `wheeled_algo/` | 算法层:algorithms(ppo_base + HIM/DreamWaQ/NP3O/GRU)、runners(runner_class 分发)、utils/exporter、experiments(控制变量注册表) |

## 文档

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 三包架构、依赖方向、设计决策 |
| [docs/environment.md](docs/environment.md) | 观测合同/动作管线/延迟/DR/课程/状态机/地形 逐项详解 |
| [docs/algorithms.md](docs/algorithms.md) | 五算法分支机制、钩子接入点、新分支写法 |
| [docs/experiments.md](docs/experiments.md) | 实验定义/运行/续训/对比 工作流 |
| [docs/migration.md](docs/migration.md) | 迁移到自有机器人的完整清单与常见坑 |

## 结构

```text
src/
├── wheeled_world/               # "世界":资产与物理
│   ├── assets/__init__.py       #   ArticulationCfg(USD 路径经 WHEELED_RL_ASSETS_DIR)
│   ├── actuators/               #   数据驱动电机模型扩展点(M3508 类曲线模型)
│   └── terrains/
├── wheeled_tasks/               # "任务":环境与课程
│   ├── direct/wheeled_biped/
│   │   ├── env.py                   # DirectRLEnv:35D policy / 43D critic / 6D act
│   │   ├── env_cfg.py               # 全部超参(Flat / Rough 两个任务变体)
│   │   └── state_machines/          # 腾空-落地 / 台阶 / 坡面 FSM
│   ├── manager/mdp/             #   commands / delay / events / curriculums / terrain
│   └── agents/                  #   RslRlOnPolicyRunnerCfg(调优数值)
└── wheeled_algo/                # "算法":rsl_rl 插件层
    ├── algorithms/              #   ppo_base(ExtPPOLoop) + him / dreamwaq / np3o
    ├── runners/                 #   runner_class 字符串分发 facade(ExtOnPolicyRunner)
    ├── utils/exporter.py        #   checkpoint → ONNX(合同自检)
    └── experiments/             #   Exp0xx 注册表 + frame-stack 对照分支
scripts/
├── train.py                    # gym.make → RslRlVecEnvWrapper → rsl_rl OnPolicyRunner
├── play.py                     # 加载 rsl_rl checkpoint 回放(自动反推网络维度)
├── export_onnx.py              # 导出 ONNX [1,35]→[1,6] 并自检合同
├── run_experiment.py           # 一键实验:本地 toy 实跑 / 服务器模式打印 Isaac Lab 命令
└── compare_experiments.py      # 跨实验对比表 + CSV 导出(读 runs/*/metrics.jsonl)
tests/
├── test_mdp.py                 # delay/commands 单测(纯 torch)
├── test_env_features.py        # 状态机转换 + 课程推进 单测(纯 torch)
├── test_algorithms.py          # HIM/DreamWaQ/NP3O 三分支 toy 收敛测试(CPU 实跑)
├── test_rsl_rl_smoke.py        # 官方 rsl_rl 2.3.x 在 toy env 上必须收敛
└── toy_env.py / toy_env_ext.py # rsl_rl 协议 / 扩展历史流 toy 环境
```

## 环境合同(与部署仓库一致)

- **观测 35D**(policy 流):指令3 + 高度指令1 + 角速度3(×0.5)+ 投影重力3 +
  腿关节位置4 + 轮位置槽2(恒0)+ 腿速度4(×0.1)+ 轮速度2(×0.1)+ 上一帧动作6 +
  7D 模式标志。精确索引见部署仓库 `CONTRACT.md`。
- **Critic 39D** = policy 35D + 真实线速度3 + 真实高度1(非对称 actor-critic,
  经 `RslRlVecEnvWrapper` 的 obs_groups 映射到 rsl_rl)。
- **动作 6D**:4 腿位置目标(scale 0.5)+ 2 轮速度目标(scale 10)。

## 环境特性(flat 任务)

| 特性 | 实现 |
|---|---|
| 时序 | 200 Hz 物理 × decimation 4 → 50 Hz 策略 |
| 气弹簧 | prismatic joint + 逐步主动力(400→600 N 线性 + ±50 N 逐 env 随机) |
| 延迟 | obs 20–80 ms / act 20–60 ms,逐 env reset 重采样 |
| 域随机化 | 质量(×0.9–1.3 / ×0.9–1.1)、PD 增益(×0.75–1.25)、轮/腿摩擦、reset 姿态 |
| 指令课程 | spin_low/spin_mid/dash 桶,按迭代数(3000/4000/2000)分批启用 |
| rough 地形 | `WheeledBiped-Rough-v0`:TerrainImporter 高度场(台阶/反向台阶/坡面/反向坡),per-env patch → stair/slope flags 喂状态机 |
| Reward | exp 核速度/角速度/高度跟踪 + 力矩/加速度/动作率/接触/终止惩罚表 |

## 使用

```bash
# 1. 资产:URDF→USD(见 assets/README.md),然后
export WHEELED_RL_ASSETS_DIR=/path/to/your/usd_dir

# 2. 在 Isaac Lab 环境内(Isaac Sim 4.5/5.1 + Isaac Lab 2.1/2.3 + rsl-rl-lib 2.3.x)
python -m pip install -e .
./isaaclab.sh -p scripts/train.py --task WheeledBiped-Flat-v0 \
    --num_envs 4096 --max_iterations 20000 --headless

# 3. 回放 / 导出
./isaaclab.sh -p scripts/play.py --checkpoint logs/*/model_final.pt
./isaaclab.sh -p scripts/export_onnx.py --checkpoint logs/*/model_final.pt --output policy.onnx
```

### 实验工作流(迭代与实验性分支)

赛季方案 Exp0xx 式控制变量实验,注册表统一定义、一键运行、自动对比:

```bash
# 列出全部注册实验(每条只改一个变量)
python scripts/run_experiment.py --list
#   exp000_framestack5   [ppo_hist ] Exp060 对标:纯 PPO 吃 5 帧展平历史(无估计器)
#   exp001_him_latent16  [him      ] Exp063 对标:HIM 一步特权估计 + 16D latent
#   exp002_dreamwaq_kl1  [dreamwaq ] CENet-VAE 隐式估计,recon+KL
#   exp003_np3o_limit05  [np3o     ] 约束 PPO,cost limit 0.5
#   exp004_np3o_limit15  [np3o     ] 控制变量:仅放宽 cost limit 至 1.5
#   exp005_rough_curriculum [ppo_hist] 服务器专用:Rough 任务 + 课程

# 本机 toy 实跑(任意机器,CPU),metrics 逐迭代写入 runs/<exp>/metrics.jsonl
python scripts/run_experiment.py --exp exp001_him_latent16 --iterations 60

# 迭代续训(从最新 checkpoint 继续,迭代计数不断)
python scripts/run_experiment.py --exp exp001_him_latent16 --resume

# 跨实验对比(尾窗均值表 + CSV)
python scripts/compare_experiments.py runs/* --csv results.csv

# 服务器模式:打印对应的 Isaac Lab 命令与 runner 集成说明
python scripts/run_experiment.py --exp exp005_rough_curriculum --backend isaaclab
```

实验分支与服务器 rsl_rl 集成的对应关系写在每条 spec 的 `isaaclab_note`
(runner_class 字符串,赛季方案 `rsl_rl_ppo_cfg.py` 的分发模式)。

### 本机(无 Isaac Sim)验证

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu rsl-rl-lib==2.3.3
PYTHONPATH=src python tests/test_mdp.py           # delay 语义 + 指令采样器
PYTHONPATH=src python tests/test_rsl_rl_smoke.py  # 官方 rsl_rl 冒烟(toy 任务收敛)
```

rsl_rl 版本说明:Isaac Lab 2.3.x 配套 rsl-rl-lib 2.3.x(get_observations 返回
`(obs, extras["observations"]["critic"])` 元组)。pip 最新版已改成 3.x API
(TensorDict + actor/critic 键名),与 IsaacLab 2.3 不匹配 —— 训练机按 IsaacLab
配套版本装,本机测试用 `rsl-rl-lib==2.3.3`。

## 技术栈

- **仿真**:Isaac Sim 4.5/5.1 + Isaac Lab 2.1/2.3(DirectRLEnv)
- **算法**:rsl-rl-lib 2.3.x(OnPolicyRunner + PPO,adaptive-KL)
- **导出**:ONNX(opset 13,静态 shape)
- **测试**:纯 torch + rsl-rl 本机单测

## 已知限制 / 扩展点

- USD 资产不入库;需自行转换 URDF。
- algo_ext 三分支为练手级核心机制实现(本机 toy 实证收敛);上服务器接 Isaac Lab
  历史流后走 赛季方案 式 runner_class 集成。
- rough 地形难度课程(patch level)与 wheel_forward_scan 预瞄未实现;云台模式未实现。
- critic 维度 43 = 35 policy + 线速度3 + 高度1 + DR 回读4(部署合同只冻结 35D
  policy 流, critic 任意扩展不影响部署)。
