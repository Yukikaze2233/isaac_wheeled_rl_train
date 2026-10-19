# Wheeled-Legged RL

面向轮腿底盘的强化学习训练仓库，基于 **Isaac Sim + Isaac Lab**，提供 PPO 训练、仿真评测和 ONNX 导出。

正式训练代码维护在 `main`。运行使用固定提交的源码、配置和机械资产，日志与检查点保存在独立运行目录。

同时在开发基于历史观测的 **Transformer 控制策略**，相关网络与训练实现见 [transformer_rl](https://github.com/Yukikaze2233/transformer_rl)。

## 功能

- **运动训练**：平地行进、前后加减速、停车换向、原地旋转和站高控制。
- **混合课程**：固定工况复习、连续命令、停走序列、非稳态接稳和小型粗糙地形。
- **恢复接管**：姿态判断与自起轨迹衔接同一个策略，脚本和混合控制区间排除 PPO 更新。
- **物理模型**：闭链机构、气弹簧、反馈 PD、力矩限幅与实测执行器响应。
- **域随机化**：质量、重心、执行器参数、地面材料、观测噪声与通信时延。
- **训练工具**：TensorBoard、完整学习状态检查点，以及支持暂停、继续和保存退出的终端控制台。
- **部署导出**：确定性 ONNX actor、接口定义和输入输出校验向量。

## 网络架构

采用非对称 Actor–Critic。两个网络使用独立 MLP，隐藏层为 **256 / 128 / 64**，激活函数为 **ELU**。

```text
Actor     35 → 256 → 128 → 64 → 6
Critic    78 → 256 → 128 → 64 → 1
```

Actor 使用单帧传感器观测和上一帧动作。训练时采样对角高斯动作，部署时输出均值；无循环网络、无帧堆叠、无运行均值方差归一化，输出层为线性层。

| Actor 输入索引 | 内容 |
| --- | --- |
| 0–2 | 车体系速度命令 `[vx, 0, yaw_rate]` |
| 3 | 站高命令，乘 5 |
| 4–6 | 车体系角速度，乘 0.5 |
| 7–9 | 车体系单位重力方向 |
| 10–15 | 相对名义关节位置，腿角包裹至 ±π，轮位置置零 |
| 16–21 | 关节输出侧速度，乘 0.1 |
| 22–27 | 上一策略帧裁剪后的动作 |
| 28–34 | 命令上下文 |

Critic 在训练中额外使用真实速度、高度、执行器增益与力矩、信号年龄、接触和物理参数。Critic 不参与实车推理。

ONNX 接口为 `obs: float32[batch, 35] → actions: float32[batch, 6]`。策略轴序为 `[左髋, 左膝驱动, 右髋, 右膝驱动, 左轮, 右轮]`，膝驱动轴角与机构内部膝夹角分别处理。

## 软件环境

已验证的软件环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| Isaac Sim | 6.0.0.1 |
| Isaac Lab | 6.1.14 |
| PyTorch | 2.11.0+cu128 |

使用配置好的 Isaac Sim / Isaac Lab Python 环境，在仓库根目录安装：

```bash
python -m pip install -e .
```

仿真和学习依赖由该环境提供；本项目安装不会替换其 Isaac、PyTorch 或 CUDA 包。

## 训练

训练入口为 `scripts/train_chassis.py`，训练内容由 `configs/` 中的 JSON 配置指定。

| 配置 | 数值 |
| --- | --- |
| 并行环境 | 4096 |
| 每次更新采样 | 每环境 24 帧 |
| PPO | 5 epochs / 4 minibatches |
| 学习率 | adaptive，初值 1e-4 |
| KL 目标 / PPO clip | 0.01 / 0.2 |
| γ / λ | 0.99 / 0.95 |
| entropy / value loss 系数 | 0.005 / 4 |
| 策略频率 | 50 Hz |
| 物理与反馈 PD | 200 Hz，4 个子步 |
| 求解器 | TGS，12 / 6 |

查看训练入口参数：

```bash
python scripts/train_chassis.py --help
```

训练启动参数包含课程配置、运行目录、检查点和预算账本。已配置的训练主机使用 `v7ctl` 管理这些参数，并在启动过程中核对配置、资产与学习状态：

`pip install -e .` 不会安装 `v7ctl` 快捷命令。新环境可在仓库根目录运行 `python tools/v7ctl.py --help`，并将下文的 `v7ctl` 替换为 `python tools/v7ctl.py`；主机运行配置通过 `--config /path/to/config.json` 指定，默认读取 `~/.config/v7ctl/config.json`。

```bash
v7ctl plan       # Show the bound training command
v7ctl start      # Start or continue training
v7ctl status     # Show progress and checkpoints
v7ctl pause      # Save and pause at an update boundary
v7ctl resume     # Resume the resident worker
v7ctl stop       # Save and exit
v7ctl logs -f    # Follow the training log
```

直接运行 `v7ctl` 打开终端面板：`p` 暂停、`r` 继续、`s` 保存退出、`q` 退出面板。实现位于 `tools/v7ctl.py`，主机配置位于 `~/.config/v7ctl/config.json`。

驻留暂停保留仿真场景；保存退出后恢复网络、优化器、随机数状态和训练进度，并重新初始化物理回合。TensorBoard event 写入对应运行目录。

## 评测与导出

`scripts/evaluate_flat_policy.py` 使用固定检查点和评测协议，覆盖固定工况、停走换向、恢复接管与 rough 地形：

```bash
python scripts/evaluate_flat_policy.py --help
```

评测绑定模型、训练配置、资产 manifest、控制参数和场景设置，输出回合记录及汇总指标。

导出 ONNX 与部署接口：

```bash
python scripts/export_chassis_candidate.py \
  --checkpoint /path/to/run/model_final.pt \
  --output /path/to/new_bundle
```

父运行目录中的 checkpoint seal、训练配置、资产和冻结源码需要一并保留；导出目标目录应为新目录。部署包附带观测与动作定义、校验值及数值校验工具。

已封存的模型按版本保存在 [models/releases/](models/releases/)，仓库只保留 ONNX 文件。每个文件的 `deployment` metadata 内嵌该版本的观测、动作、控制增益和资产标识，可通过 ONNX Runtime 的 `get_modelmeta().custom_metadata_map` 读取。历史模型按自身 metadata 对接；下节描述当前模型的部署接口。

## 部署接口

策略以 **50 Hz** 更新目标，实车反馈 PD 以 **1000 Hz** 使用最新传感器数据计算力矩。

- 腿动作裁剪至 ±3，经 0.25 rad 比例和名义位置解码；同侧双轴成对选圈，使用 `paired_branch107_v1`。
- 轮动作裁剪至 ±9，经 10 rad/s 比例生成轮侧速度目标。
- 腿反馈参数为 Kp=160、Kd=2.5，输出侧力矩限幅 ±40 N·m；轮速度反馈增益为 0.6，轮侧力矩限幅 ±4.5 N·m。
- ONNX 输出动作目标；反馈 PD、航向外环和自起状态机由部署控制器执行。

## 项目结构

```text
src/
├── wheeled_tasks/chassis/    # Observations, rewards, terrain and recovery
├── wheeled_tasks/agents/     # PPO configuration
└── wheeled_algo/            # Rollout, checkpoints, export and control
configs/                    # Current configuration and calibration inputs
scripts/                    # Training, evaluation and export
tools/                      # Training CLI and geometry support
tests/                      # Current control and training-interface checks
references/                 # Pinned dependencies and license notices
```

模型资产以配置中的 `asset_directory` 和 manifest 为准。训练日志、检查点、临时诊断和对接材料存放在运行或交付目录。
