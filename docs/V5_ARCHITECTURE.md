# V5 训练主线架构

本文对应`contracts/v5_scut35_integrated_v1.json`与`scripts/train_chassis.py`。
用V5自己的闭链资产和执行器参数，参考华南虎V14普通PPO训练单策略；
课程推进、行为验收、恢复产物和远端进程具有独立职责。

## 1. 入口与运行时

| 入口 | 职责 |
|---|---|
| `launch_v5_remote.py` | 冻结源码、验证传输hash、建立Kaiser独立tmux会话、生成launch回执 |
| `chassis_remote_job.py` | 管理训练编排子进程，运行中自主封存，退出后完整打包 |
| `run_full_chassis.py` | 按依赖顺序执行阶段，验收通过后交接已接受模型 |
| `run_chassis_blocks.py` | 交替执行训练块和固定评测，维护candidate/best/accepted选择 |
| `train_chassis.py` | 装配Isaac环境与RSL-RL runner，执行PPO、保存checkpoint、发布诊断 |
| `evaluate_chassis.py` | 固定案例、固定seed下运行actor并按行为指标判定 |

当前正式运行环境为Isaac Sim 6、Isaac Lab 3 beta、RSL-RL 5.5.1。
训练入口复用`agents/v40_ppo_cfg.py::V40PPORunnerCfg`，由物化合同覆盖rollout、初始std和学习率等设置；
类名中的V40不代表当前机械资产。算法是官方`rsl_rl.runners.OnPolicyRunner`，主线没有使用自研`ExtPPOLoop`。

## 2. 资产与控制边界

资产位于`model/纯底盘_v5/urdf/`：

- `manifest.json`：资产身份、控制关节顺序、标称姿态及依赖hash。
- `model_spec.json`：刚体、惯量、关节、安装变换和闭链约束。
- `gas_spring_binding.json`、`fit_10mpa.json`：气簧坐标定义与力曲线来源。
- USD及视觉/碰撞网格：供PhysX装配和可视化使用。

`ChassisEnv`负责Isaac/PhysX适配：装配场景、绑定关节索引、写入力矩、推进物理步、读取接触与刚体状态。
`V5Control`负责纯张量计算：

1. 六个policy动作映射至四个腿位置目标和两个轮速度目标。
2. 每个物理步计算PD与速度相关轮电机限幅。
3. 用移动副坐标计算气簧压缩量，施加轴向关节内力。
4. 计算耦合机构工作余量，避免对同一机械极限重复计罚。

物理200Hz、policy50Hz，decimation=4。气簧、电机、质量和惯量使用V5参数。
当前部分参数仍是研究先验，见资产标定文档；训练收敛不代替参数辨识。

## 3. 观测、命令、奖励与完成条件

| 模块 | 职责 |
|---|---|
| `scut_observation.py` | 可部署传感器与指令 → 固定35D actor输入，处理动作顺序映射 |
| `skill_commands.py` | 根据采样量和episode时间生成指令，管理特殊重置及旋转参考系 |
| `performance_curriculum.py` | 按完成episode统计原始误差，调整训练核宽/权重/转速并序列化课程状态 |
| `motion_limits.py` | 根据V5轮半径/轮距检查或投影指令包络 |
| `rewards.py` | 仿真状态与目标 → 底盘分项奖励密度与驻留目标；参考来源在模块注释中记录 |
| `task.py` | 相位跟踪、fall持续确认、地形表面、走廊mesh与环境组分配 |
| `full_tasks.py` | 跳跃稠密参考、离地/质心/落地事件与完成谓词 |
| `robustness.py` | 观测与动作延迟、观测噪声 |
| `torque_monitor.py` | 实际电机/气簧作用量的分组统计 |
| `episode_metrics.py` | 分case累计误差、终止原因、暖机后的样本及回合结果 |

Actor只读取可部署35D输入；81D privileged critic可使用仿真真值。
真实接触、被动膝、气簧、地形和真实根速度不进入actor。
最后7维是命令上下文，不是接触驱动的真实相位，详见[传感器合同](V5_SCUT35_SENSOR_CONTRACT.md)。

奖励、任务完成、验收是不同边界：高reward不自动意味着任务完成，完成事件也不能代替固定评测。
普通运动使用基座原点速度进行reward/评测；跳跃另外观察整机质心，防止仅伸腿抬高机身被计为跳起。

## 4. 合同物化与课程组织

`full_curriculum.resolve_plan`解析继承、覆盖和技能插入，保留技能目录。
`stage_contract`统一组织基础合同与两种物化路径：

- 专项路径：`skill_curriculum.py`定义单技能采样、固定案例与前序回归。
- 综合路径：`integrated_curriculum.py`在准备好的基础合同上装配多技能分布及验收案例。

综合模块不反向调用合同组织器；命令调度位于`skill_commands.py`，不依赖计划物化器。
依赖方向为：入口脚本 → 合同组织器 → 技能/综合配置；环境 → 指令/控制/奖励/指标。
rollout期间不读取或改写课程计划文件。

V5.1在同一边界增加修复采样与`StationaryAnchor`，性能课程与奖励函数分离。
固定评测禁用训练中的转速上限与核宽进度，仍执行原目标；新训练不通过降低门槛宣称过关。
核宽与权重在每环境reset时锁存，已完成episode窗口及等级随checkpoint恢复。
V5.2完整计划见`V52_REPAIR_TRAINING.md`：核宽分级带独立误差门槛；训练fall持续确认与固定评测单帧判据分离。

当前7个阶段为flat、landing、rough、spin_translate、high_speed、jump、mixed_robust。
各环境组共享同一actor；后续阶段保留早期速度档和高度端点回归。

预算以4096环境×24步为基准：

- 更新上限：`ceil(reference_updates × 4096 / num_envs)`。
- 命令调度时钟：`training_transitions / (4096 × 24)`。
- checkpoint/评测间隔使用实际PPO更新数。

物化器整理后已与此前保存的7个阶段合同逐项比较，内容完全相同。

`evaluation.py`逐例判断速度、高度、漂移、姿态、完整时长或任务成功。
预热只影响误差采样窗口，失败回合仍计入。预算耗尽但未过门时保留待验收状态。

## 5. 进程与恢复边界

```text
tmux session
  chassis_remote_job.py             autonomous sealing and final delivery
    run_full_chassis.py             accepted-stage handoff
      run_chassis_blocks.py         sequential training / evaluation
        train_chassis.py            PPO worker
        evaluate_chassis.py         fixed evaluation worker
```

训练不依赖本地SSH、ControlMaster、TensorBoard或回收watcher存活。
TensorBoard独立运行，只读取event文件；GUI位姿镜像也不向物理世界写入控制量。

`chassis_checkpoints.py`先写临时目录，再原子发布不可变checkpoint目录。
保存actor、critic、优化器、学习率、更新数、采样数与随机数状态。
恢复时PhysX回合重新初始化，不是完整物理世界的逐位复现。
完整编排器也支持`--resume <sealed-checkpoint> --start-stage <stage>`，从同阶段恢复优化器，
后续阶段仍只接收通过验收的模型；计数从不可变completion元数据读取，避免监督进程额外加载PyTorch。

`chassis_batch_export.py`只封存已发布快照和已退出写入进程的训练/评测单元，生成逐文件SHA-256。
`sync_chassis_batches.py`增量下载并校验归档及成员，拒绝路径越界或已存在不可变文件的内容冲突。
`watch_chassis_artifacts.py`负责结束后的完整回收。

| 产物 | 含义 |
|---|---|
| 周期快照、`model_final.pt` | 最近训练状态，可能未通过行为验收 |
| `best_candidate.pt` | 按评测排序保留的候选 |
| `model_best.pt` | 阶段内通过评测的模型 |
| `accepted_policy.pt/.onnx` | 已通过阶段交接门槛的策略，由selection记录身份 |

运行身份由基准commit、显式overlay hash、源码归档hash和合同hash共同确定。
本地后续提交不会修改正在运行的远端源码副本。

## 6. TensorBoard与验证层次

训练图表统一使用TensorBoard event。训练入口显式选择`logger="tensorboard"`。
新增指标接入RSL-RL logger或`extras["log"]`，约定稳定tag、单位及step；静态图由event导出。
JSON继续用于合同、评测判定、checkpoint选择、恢复和归档回执，不另建重复的图表数据管线。

验证分为：

1. 纯Python/PyTorch：合同预算、命令包络、观测布局、奖励性质、门控与归档。
2. 物化结果比较：架构整理前后的7阶段合同完全相同。
3. 仿真探针：真实PPO更新、闭链残差、内存余量与ONNX数值一致性。
4. 恢复探针：封存checkpoint恢复优化器与课程时钟后继续PPO。
5. 行为验收：固定案例、独立seed与前序技能回归。

CPU CI与正式Isaac运行时使用不同的RSL依赖集合，CPU测试不能替代远端仿真验证。
实际启动身份、TensorBoard访问与恢复命令见[运行手册](V5_INTEGRATED_TRAINING_20260921.md)。

## 7. 历史模块与维护方向

- `direct/wheeled_biped`、`manager/mdp`及自研算法分支是历史V3/实验路线。
- V5复用`wheeled_tasks/v40`中的部分控制函数、HistoryStack及PPO配置。
- `wheeled_world`三包设计仍服务于历史入口，当前V5资产通过`ChassisEnv`直接装配。

主要耦合点是`ChassisEnv`同时承担场景装配与rollout，以及训练脚本对runner的更新/保存回调适配。
后续扩展优先在这两个边界提取有明确归属的组件，并保留合同、物理探针和恢复测试。

性能整理采用缓存/批量闭链误差计算、跳过无关任务的奖励路径，以及每环境累积力矩后在report时分组归约。
仍保留全部物理步样本、float64力矩矩与int64计数；不通过降低物理频率或放宽验收来换取吞吐。
局部遥测微基准与端到端PPO吞吐是不同指标；实际加速以同checkpoint、同合同的顺序对照为准。
