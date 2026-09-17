# 新两级四杆模型：训练仓库内的交付

## 位置

所有交付文件均位于 `/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train/` 内：

- 完整模型目录：`model/纯底盘/urdf/`
- 压缩包：`model/纯底盘/chassis_closedchain_20260917.zip`，约 5.9 MiB
- URDF：`model/纯底盘/urdf/robot.urdf`
- Isaac 入口：`model/纯底盘/urdf/robot.usda`
- MuJoCo 入口：`model/纯底盘/urdf/robot.xml`

模型目录及 ZIP 与已验证候选逐文件一致，压缩包完整性检查通过。
必要模型资产进入版本管理；重复的分发 ZIP、运行日志与训练权重按 `.gitignore` 保存在本地。

## 关节轴系与物理属性

模型包含 **15 个独立刚体、14 个树转动关节、4 个闭合球约束**。
每个刚体都有局部 frame、visual/collision、质量、质心及惯量；每个树关节都有 parent/child、
origin、axis 和限位语义。总质量 **12.752 kg**，共有 36 个碰撞代理。

每侧保留髋、膝、轮三个等效控制坐标，新增曲柄 O、下连杆 A、摇杆 P、上连杆 B 四个被动轴。
下连杆与摇杆在 D 点闭合，上连杆与小腿在 E 点闭合。USD 与华南虎采用相同表达方式：
闭合球约束设置 `physics:excludeFromArticulation=true`，由物理引擎求解。

**标准 URDF 只表达树**，单独导入 `robot.urdf` 不会得到闭链；必须额外应用 `constraints.json`。
在 Isaac 中直接加载 `robot.usda` 即可保留这四个闭合约束。

拆分杆件物性是均匀密度积分、按源每侧质量归一后的研究估计，并非实测 CAD 惯量。
真实电机/链传动到关节的映射待确认；当前六轴为等效输出控制坐标。
当前禁用机器人自碰撞，保留地面/外界碰撞；硬件部署状态仍为未就绪。

## 已完成验证

静态及 MuJoCo 证据随模型打包：183 个姿态的 FK/闭合核对、逐杆正定惯量、质量守恒、
真实动力步进及移除闭合约束对照。MuJoCo 闭合约束 Jacobian 秩为 8，保留 6 个内部独立坐标。

后续 GPU PhysX 验证使用 Sim 6.0.0.1、Lab 3 beta2.patch1、本机 RTX 4060 Laptop：

- 求解器实际读到 15 刚体、14 坐标，总质量 12.7520008 kg。
- 200 Hz 物理、TGS 位置/速度迭代 16/4；每轴阻尼 0.002 Nm·s/rad、armature=0。
- 地面为独立测试用静态长方体，顶面 z=0、摩擦 0.5。
- 自由重力 4 s 最大闭合误差 **0.7051 mm**；小力矩 4 s 为 **0.7261 mm**。
- 被动轴初始扰动后，引擎在 0.1 s 内将对应点误差从 0.4049 mm 降至 0.000248 mm。
- 禁用四个闭合约束，0.3 s 后误差达到 **110.9685 mm**。
- 每步无几何闭合器调用、无被动关节姿态回写。

独立单环境 PPO 工程试训完成 **16 次成功更新 / 768 个策略步**，actor/critic 参数均实际改变且有限，
最大闭合误差 **0.4532 mm**。固定 0.32 m 指令、2 s 回合，3 个结束回合均为时间到期。
仅验证了真实闭链、观测控制与官方优化器能够联通，样本不足以判定站立能力。
此模型还没有替换本分支默认训练入口的七刚体资产，也没有改动 Kaiser 上的旧模型续训。

## 证据与指纹

后续运行证据已一并放入本仓库本地目录：

- `reports/chassis_physx_20260917_gpu_01/report.json`
- `reports/chassis_physx_20260917_gpu_ablation_01/report.json`
- `reports/chassis_physx_20260917_ppo_01/report.json`
- `reports/chassis_physx_20260917_ppo_01/ppo/`：配置、TensorBoard、checkpoint。

`ppo/experiment.json` 是启动时快照；最终完成状态看外层 `report.json`。
资产包是后续 PhysX 试验之前的冻结快照，包内 `isaac_validation=not_run` 保留生成时状态。
上述报告按 manifest SHA256 关联同一资产，没有通过回写验证结果改变模型身份。

| 文件 | SHA256 |
|---|---|
| ZIP | `ceaf7df719b1c4e366400ca3799da5f294e2c25498c53ff54542557154b111d3` |
| manifest.json | `715f8e5bf8259543aacdb4d29fc97962f37a943ea1157d22a09ef4bbdff92016` |
| robot.urdf | `9ea0b4a76c3da4e81ecf229d61881bd77813c7290414ee7d996143f9bd5bf445` |
| robot.usda | `9aa957ff3fd9821a480aa6e9f23d465c4c0491e313bc4ffc5dffbfa0faa44d41` |
| 试训 model_final.pt | `b86b6b548a64ef5151f4b95631adcbd00d518064721574cc66f394b4e15d3f7e` |
