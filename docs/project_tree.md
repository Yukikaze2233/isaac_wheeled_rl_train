# 项目架构层级树

两仓库完整文件树(以 git 跟踪内容为准)。行内注释为该文件的职责一句话。

## 训练仓库 `isaac_wheeled_rl_train`

```text
isaac_wheeled_rl_train/
├── README.md                        # 快速上手 + 架构总览表 + 文档索引
├── pyproject.toml                   # 包定义(src 布局, Isaac Lab 环境内安装)
├── assets/
│   └── README.md                    # URDF→USD 转换步骤 + 自有机器人迁移清单(简版)
├── docs/
│   ├── architecture.md              # 三包架构、依赖方向、5 个设计决策
│   ├── environment.md               # 观测合同/动作管线/延迟/DR/课程/状态机/地形
│   ├── algorithms.md                # 五分支机制、钩子接入点、新分支写法
│   ├── experiments.md               # 实验定义/运行/续训/对比 工作流
│   └── migration.md                 # 自有机器人迁移六步清单
├── scripts/
│   ├── train.py                     # 训练入口: gym.make → RslRlVecEnvWrapper → OnPolicyRunner
│   ├── play.py                      # checkpoint 回放(自动反推网络维度)
│   ├── export_onnx.py               # 导出 ONNX [1,35]→[1,6] + 合同自检
│   ├── run_experiment.py            # 一键实验: 本机 toy / 服务器命令打印 / 断点续训
│   └── compare_experiments.py       # 跨实验对比表 + CSV 导出
├── src/
│   ├── wheeled_world/               # ── "世界"层: 机器人物理形态 ──
│   │   ├── __init__.py              #   AssetPath 解析(WHEELED_RL_ASSETS_DIR)
│   │   ├── assets/__init__.py       #   ArticulationCfg: 执行器分组/armature/USD
│   │   ├── actuators/
│   │   │   ├── __init__.py          #   扩展点说明(数据驱动电机模型)
│   │   │   └── m3508_curve.py       #   实测扭矩-转速曲线 → 扭矩限幅插补
│   │   └── terrains/__init__.py     #   世界侧地形定义占位
│   ├── wheeled_tasks/               # ── "任务"层: 环境与课程 ──
│   │   ├── agents/rsl_rl_ppo_cfg.py #   RslRlOnPolicyRunnerCfg(训练超参)
│   │   ├── direct/wheeled_biped/
│   │   │   ├── env.py               #   DirectRLEnv 主环境(观测/动作/奖励/重置)
│   │   │   ├── env_cfg.py           #   全部超参 + Flat/Rough 任务变体
│   │   │   └── state_machines/      #   腾空-落地 / 台阶 / 坡面 FSM(7D 模式标志)
│   │   └── manager/mdp/
│   │       ├── commands.py          #   特殊模式指令采样器(spin/dash 桶+迭代门控)
│   │       ├── delay.py             #   观测/动作延迟 ring buffer(逐 env 滞后)
│   │       ├── events.py            #   域随机化: 质量/COM/材质/PD/摩擦/root
│   │       ├── curriculums.py       #   reward 权重课程 + 基座辅助力退火
│   │       ├── terrain.py           #   rough 高度场工厂 + per-env flag 映射
│   │       ├── jump_rewards.py      #   跳跃稠密轨迹奖励族(4 项)
│   │       └── terrain_cmd.py       #   按地形名指令覆盖(向量化)
│   └── wheeled_algo/                # ── "算法"层: rsl_rl 插件 ──
│       ├── algorithms/
│       │   ├── ppo_base.py          #   ExtPPOLoop(采集→GAE→更新)+ HistoryRoller
│       │   ├── him.py               #   HIM 分支(自研路径)
│       │   ├── dreamwaq.py          #   DreamWaQ 分支(自研路径)
│       │   ├── np3o.py              #   NP3O 分支(自研路径)
│       │   ├── recurrent.py         #   GRU 分支(自研路径)
│       │   └── ppo_ext.py           #   rsl_rl PPO 扩展骨架(辅助损失接线, WIP)
│       ├── modules/actor_critic_ext.py  # rsl_rl ActorCritic 分支(帧堆叠合同)
│       ├── runners/
│       │   ├── on_policy_runner_ext.py  # ExtOnPolicyRunner facade(自研路径)
│       │   └── branch_runners.py    #   官方 runner 的 class_name 注入分发
│       ├── utils/exporter.py        #   checkpoint → ONNX(维度自动反推)
│       └── experiments/
│           ├── registry.py          #   Exp0xx 控制变量注册表
│           └── ppo_hist.py          #   frame-stack 对照分支
└── tests/                           # ── 7 套件回归(纯 torch, 无 Isaac Sim)──
    ├── toy_env.py                   #   rsl_rl 协议 toy 环境
    ├── toy_env_ext.py               #   扩展历史/特权流 + 帧堆叠模式
    ├── test_mdp.py                  #   delay 语义 + 指令采样器
    ├── test_env_features.py         #   FSM 转换 + 课程推进 + 地形 flags
    ├── test_season_features.py      #   电机曲线 + 跳跃轨迹 + 指令覆盖 + GRU
    ├── test_algorithms.py           #   三分支 toy 收敛断言
    ├── test_rsl_rl_smoke.py         #   官方 rsl_rl 栈冒烟 + checkpoint 往返
    ├── test_rsl_rl_branches.py      #   分支类经官方 runner 训练断言
    └── test_experiments.py          #   实验框架端到端(注册/续训/对比)
```

## 部署仓库 `isaac_wheeled_rl_deploy`

```text
isaac_wheeled_rl_deploy/
├── README.md                        # 部署快速上手 + 架构图 + 文档索引
├── CONTRACT.md                      # 冻结策略合同(35D→6D, 唯一权威)
├── docs/
│   ├── contract_changes.md          # 合同变更五处同步流程
│   ├── timing.md                    # 500Hz/50Hz 双频时序与训练端对齐
│   ├── sim2real.md                  # 上真机核对清单 + real2sim 辨识 + 排查表
│   ├── ros2_architecture.md         # 三层结构/sim-real 切换/扩展路径
│   └── testing.md                   # 两仓验证体系 + 改动→必跑对照表
├── models/
│   └── toy_wheeled_biped.xml        # 玩具 MJCF(部署链路验证用)
├── sim2sim/
│   └── mujoco_sim2sim.py            # 轻量 sim2sim(500Hz PD + 50Hz ONNX, 不经 ROS)
├── tools/
│   └── check_onnx_contract.py       # ONNX 合同校验器(部署入口第一道闸)
└── ros2/src/                        # ── ros2_control 工作区(colcon)──
    ├── controllers/wheeled_rl_controller/
    │   ├── rl_controller.{hpp,cpp}  #   ControllerInterface: 接口声明 + 500Hz 循环
    │   ├── state_machine.hpp(.cpp)  #   fsm 模块: INIT→IDLE→PREPARE(1rad/s)→RL
    │   ├── robot_state.hpp(.cpp)    #   robot_state 模块: 状态缓存 + 35D 拼装
    │   ├── policy_runtime.hpp(.cpp) #   onnxruntime 模块: ORT 会话封装
    │   ├── wheeled_rl_controller.xml#   pluginlib 导出
    │   └── CMakeLists.txt           #   HAVE_ONNXRUNTIME 编译开关
    ├── interfaces/
    │   ├── wheeled_mujoco_system/   #   sim 后端: SystemInterface + 插件内 PD 闭环
    │   └── wheeled_real_system/     #   real 后端: 串口 RealBridge(termios+CRC16)
    ├── middlewares/wheeled_bringup/
    │   ├── launch/bringup.launch.py #   backend:=sim|real 一键切换
    │   └── config/                  #   ros2_control yaml + xacro(接口声明)
    └── tools/wheeled_teleop/
        └── teleop.py                #   键盘遥控(50Hz Twist + 高度指令)
```

## 层间数据流(一图总览)

```text
wheeled_world (资产/电机模型)
      │ ArticulationCfg
      ▼
wheeled_tasks (env × mdp 组件 × 状态机)          wheeled_algo (算法分支)
      │ obs 35D ──────────────── ONNX ──────────►  部署三层
      │                                            controllers → interfaces
      ▼                                            (sim: MuJoCo | real: 串口)
experiments (registry → run → compare)    CONTRACT.md 为两仓唯一接口权威
```
