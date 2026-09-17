# 纯底盘：两级四杆闭链研究候选

这是独立完整的 `coupled_fourbar_research` 包。15 个真实刚体（含 base），14 个树转动关节，
4 个球形点闭合约束，共 18 个连接；6 个等效输出控制坐标、8 个被动坐标。
`robot.urdf` 是标准树；**仅导入 URDF 不会得到闭链**，必须同时应用 `constraints.json`。
可直接用 `robot.usda`（USD Physics）或 `robot.xml`（MuJoCo）导入闭链。

## 坐标与驱动

- 根 frame：X forward / Y left / Z up；原 canonical 六轴 origin、axis、raw q 保留。
- 控制顺序：`L_joint1,L_joint2,L_joint3,R_joint1,R_jonit2,R_joint3`，保留历史拼写。
- 每侧树：base→C2→shank→wheel；C2→C0@O→C1@A；C2→C3@P→C4@B。
  闭合为 C1–C3@D、C4–shank@E。新增 child 原点位于其父连接轴；visual/collision/COM 同步换帧。
- source active q=0 对应左膝内角约 42.261°、右约 44.937°，**不是 68°**。
  新增 passive q=0 采用该源零位精确闭合；微米级拟合调整见 `joint_mapping.json`。
- `nominal_joint_pos` 含全部 14 轴，六主动值不变；base 名义高度 0.32 m。
  USD 直接 author 该名义姿态；MJCF 使用 `mj_resetDataKeyframe(model, data, 0)`。
- hip/wheel/passive 为 continuous，无 ±π stop；仅两膝保留原 raw 坐标下 35–80° 限位。
- URDF/USD 无 drive。MJCF 有六个默认零输入的 unit-gear torque actuator，非硬件电机映射。
  C0 实际驱动、编码器/传动映射仍 pending；`hardware_deployment_ready=false`。

## 质量、惯量与碰撞

总质量 **12.752 kg** = 10.8 + 2×(0.326 + 0.45 + 0.2)，不是旧 ZIP 文档的 13.404 kg。
base/shank/wheel 复用 canonical 原惯量（未实测、未自动翻转非对角项符号）。
C0–C4 按独立完整精度体积积分，每侧共同密度归一至 0.326 kg；这是
**uniform-density research prior，非 CAD 真值**，没有重复计入旧 aggregate 惯量。
与原 aggregate COM 的差约 3.449/4.237 mm，惯量 Frobenius 差约 38.803%/40.121%。
完整来源、原 tensor、逐杆 COM、单位密度积分和 C2 极小浮点接缝诊断见 `inertial_sources.json`。

15 个 visual STL 顶点保持源值、不缩放；USD 原生 mesh 同值。动态 collision 使用本地 convex
研究代理和原轮 cylinder（R≈0.06 m，半宽≈0.0125 m，轴偏移±0.02035 m），没有动态 triangle collider。
base/shank 沿用组件 convex，新增杆用各自 convex hull，超过 255 点时用保守 PCA box。
这些代理填充孔洞/凹陷，不代表精确实体接触。来源见 `collision_sources.json`。
新候选默认关闭全部 self-collision，外界/地面接触保留：USD 使用真实 FilteredPairsAPI
及 `physxArticulation:enabledSelfCollisions=false`，MJCF 使用互斥机器人/地面 bitmask，
URDF 提供研究用途 `collision_filters.srdf`。**新 self-collision 审核未通过**，未继承旧 6-pair 批准。
USD 闭合为 `UsdPhysics.SphericalJoint` + **`physics:excludeFromArticulation=true`**。
平面 hinge 树使 4 个三维点约束的独立秩为 8，保留六个内部独立自由度。

## 文件与复验

- `manifest.json`：训练交接接口、全文件 SHA256（不含 manifest 自身）、状态。
- `robot.urdf`, `robot.usda`, `robot.xml`：三种自包含相对路径入口。
- `model_spec.json`, `kinematics.json`：完整换帧模型和确认的纯几何输入；后者的旧 physics_ready=false
  属于源视觉数据标签，不是候选验证结果。候选结果以 validation 文件为准。
- `static_validation.json` 与 `dynamics_validation.json`：分开的静态与 MuJoCo 实测证据。
- `tools/build_chassis_closedchain.py`：完整生成/复验源码。

依赖已有 Python + numpy/scipy/trimesh/pxr/mujoco。无需 reports、Downloads、训练仓库或环境代码即可复验：

```bash
python tools/build_chassis_closedchain.py --validate-only .
```

动力学验证从名义姿态只初始化一次，后续不写 passive qpos/pose、不调用几何闭合求解器。
包含自由基座重力跌落、小力矩、固定基座重力小力矩及移除约束的对照。
固定基座仅是临时测试 fixture；三个正式模型均为自由基座。MJCF 阻尼 0.002 Nm·s/rad
是显式数值研究先验，USD/URDF 不注入驱动；MuJoCo 通过不等同于 PhysX 稳定性通过。
**Isaac/PPO 尚未运行，后续由主 agent 集成独立 snapshot。**

旧 `model/纯底盘/urdf_V4.0_multilink.urdf` 是保留的他人草稿，其 hip 原点、mesh/COM rebase 和
wheel limits 不适合作为本候选源。有效的新候选入口是**本目录的 `robot.urdf`/`robot.usda`/`robot.xml`**。
