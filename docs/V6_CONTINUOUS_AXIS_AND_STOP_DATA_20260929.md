# V6 主动轴连续回转与止挡数据核验

2026-09-29。本报告对应用户确认：大腿主动轴没有机械角限位，同腿两个主动输出等增量运动可共同回转；膝内角仍有机械止挡。没有修改封存资产。

## 资产身份与机械事实

核验目录 `model/纯底盘_v6_232mm_stop105/urdf`，完整封装 manifest SHA-256 `875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba`。本次逐项复核 manifest 的 84 个文件哈希。

| 文件 | SHA-256 |
|---|---|
| robot.urdf | `12488abd510c1b6e395b88a786e9f5b542f929d10d4a3f1ce1f7b68699e35635` |
| robot.xml | `770340546b0d6fabdb1dba87dfa03ecfaacaea20a3b5b8de5c1984fb377dff3f` |
| robot.usda | `f9a1e4a9350ff0d98af187ed287f73428a10ed339c9ef7b9de04688fa30f0f98` |

`L_joint1/R_joint1/LL_joint1/RR_joint1` 四个主动腿轴：URDF 全部为 `continuous`，仅有 effort/velocity 参数，没有 lower/upper 位置限位；MJCF 全部 `limited=false`；USD 的四个 `PhysicsRevoluteJoint` 没有 author `physics:lowerLimit/upperLimit`。本机 MuJoCo 3.12.0 编译读取的四轴 `jnt_limited` 全为 false。USD 文本检查与既有封装 PhysX 回执共同保留证据；本次没有新启动 PhysX/GPU。

独立相对膝关节 `L_joint2/R_jonit2` 保留有限区间：左 `[-0.08616095279206137, 1.048303061004253] rad`、右 `[-1.048303061004253, 0.08616095279206137] rad`，通过销心向量投影计算，两端严格对应膝内角 40°/105°。主动轴连续不意味着膝差动可突破这个范围。

本次用封装 FK、SciPy 闭链求解和 MuJoCo forward，分别沿正负方向扫描两圈，共 66 个姿态，四主动轴都相对 nominal 同增量至 ±4π。最大闭链残差 `1.6653e-16 m`；膝内角变化上界 `7.0705e-6 rad`（约 0.000405°），气簧长度变化上界 `3.4591e-7 m`，来自保留的 CAD 销点/轴线微小不共轴。所有编译状态/加速度有限。这证明坐标拓扑不存在主动轴绝对角止挡；这是几何检查，并非含车体自碰撞和接地的无限回转动作验收。

动作目标取值范围、任务期望姿态、跌倒/数值异常终止阈值属于控制和训练合同，不能被解释成新增机械限位。不能因大腿无止挡就删除相对膝角、力矩、速度或真实接触的边界。

## 碰止挡的 bag 如何使用

打到止挡的数据可用，不能整包作废。按实际运动相位分层：自由段用于惯量、摩擦、执行器增益/延迟等辨识；接近/接触/离开止挡段用于有效止挡位置、加载变形、反力、回弹及接触耗散核验；长时间停靠平台用于校零与静载合力约束。接触状态不确定的过渡窗口要单列，并对自由段与接触段分别报告残差、覆盖率和留出结果。

仅凭止挡约束让轨迹 RMSE 变小，不能声称电机/气簧参数已识别。当前 bag 约覆盖膝内角 100–107°，可支持该局部姿态及止挡附近检查，不能证明 40–105°完整活动域的动力学已识别。105°名义止挡、106/107°有效平台与 encoder 零位之间仍相关；不应把同一不确定性重复加到零偏和止挡随机化里。原始 bag/时间戳/输入力矩都应保留，接触段的使用要保持接触模型与反力解释。

上述数据分层是本项目的后续辨识设计，不归因于华南虎源码已有自动化实现。资产原 `training_ready=false`、硬件辨识待完成状态保持原样；允许仿真研究训练的运行合同应另行记录假设和验收条件，不能改写历史物理验收结论。

## 华南虎源码实际做法

只读核验本地 `scut-wheeled-legged-rl` HEAD `b8ff79f3df855faf9dc92f4a282bd80c42649466`：

- `source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py:2774` 解码位置目标；`:2871` 对目标按配置 clamp。`env_cfg.py:400` 默认 ±3.14 rad。这是命令范围，不能拿来认定自有机械轴有 ±π 止挡。代码还提供 sin/cos 角编码、相对当前角 unwrap，但不能只看到 unwrap 就忽略随后 `_apply_action` 的目标 clamp。
- 同一 `env.py:3435` 的 `_get_rear2_rear1_joint_limit_terms` 对相对关节软边界分别计算位置、向外力矩、向外速度惩罚；`:4911` 终止主要按碰撞、姿态、数值/观测异常和时长处理。V14 继承 V13，再继承该基类；V14 配置 `env_cfg.py:569` 禁用旧 front/rear 差值罚项，改用 rear2 相对关节范围（-1° 到 68.5°）。这些数值属于其机械体，不移植为本机的膝内角。
- `wheelbipe_V13/env_cfg.py:203` 等对主动、被动及轮关节摩擦/armature 做域随机化；`source/agent_world/agent_world/assets/wheelbipe_V13.py` 定义执行器 PD/力矩/速度及阻尼先验。源码存在待标定说明。
- 在 `source/`、`scripts/` 和 README 搜索 system identification、least_squares、curve_fit、rosbag、calibration、辨识、标定，没有找到碰止挡 bag 的系统辨识优化器；README 对 Sim2Real 系统辨识仍列待办。不能声称华南虎已有一套能直接代替本项目辨识的优化器。

可借鉴的是分开机械边界、命令域、软边界代价与随机化，再做本机自己的验证；不是以任何软件位置 clamp 推翻用户确认的无机械止挡事实。
