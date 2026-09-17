# 资产:URDF → USD

## V5.0 含气弹簧候选

原始导出在 `model/纯底盘_v5/source/`，含19个刚体。静态审计和10 MPa气弹簧拟合保存在同级目录。
完整训练设计及接入前待核实项见 [V5方案](../docs/V5_FULL_TRAINING_DESIGN.md)。

## 新两级四杆底盘

完整模型在 [`model/纯底盘/urdf/`](../model/纯底盘/urdf/README.md)，本地交付包为
`model/纯底盘/chassis_closedchain_20260917.zip`。15 个刚体、14 个树关节轴、4 个闭合球约束，
包含逐刚体质量、质心、惯量和碰撞几何。

- URDF：`model/纯底盘/urdf/robot.urdf`，闭环约束另见同目录的 `constraints.json`。
- Isaac：使用已含闭环约束的 `model/纯底盘/urdf/robot.usda`。
- MuJoCo：`model/纯底盘/urdf/robot.xml`。
- 交付、物性来源和后续 PhysX/PPO 验证见[记录](../docs/CHASSIS_CLOSEDCHAIN_20260917.md)。

## 早期外置资产流程

以下保留早期模型的外置资产约定。`wheeled_world/assets/__init__.py`
通过环境变量 `WHEELED_RL_ASSETS_DIR` 定位 USD:

```text
$WHEELED_RL_ASSETS_DIR/
└── wheeled_biped/
    └── wheeled_biped.usd
```

## 转换步骤(Isaac Sim 内)

1. `File → Import → URDF`(或用 `omniverse` 的 urdf importer 扩展):
   - Root Link: `base_link`
   - Merge Fixed Joints: **关闭**(保连杆结构,便于调质量)
   - Self Collision: 关闭
   - 惯量:保留 URDF 原值(勿勾选自动估算)
2. 导出为 `.usd`,放到上面目录结构中。
3. 核对关节名 —— env 依赖以下命名约定(或同步修改 env 与 assets 配置):
   - 主动腿关节:`{left,right}_front1_joint` / `{left,right}_rear1_joint`
   - 被动闭链关节:`*_front2/3/4_joint`、`*_rear2_joint`、`*_spring1_joint`
   - 气弹簧移动关节:`*_spring2_joint`(prismatic)
   - 轮:`*_wheel_joint`
4. 用 USD viewer 检查:默认站姿、质心、惯量、碰撞体。

## 换成自有机器人的清单

- [ ] `assets/__init__.py`:init_state 关节零位 / spawn 高度 / 各执行器组参数
      (Kp/Kd/力矩限幅/armature=转子惯量×减速比²)
- [ ] env_cfg:`height_range`、`default_height_cmd`、`leg/wheel_action_scale`
      (按关节行程)、`max_wheel_vel`
- [ ] mdp/events.py 调用处:body/joint 命名(若不一致)
- [ ] 部署仓库 `CONTRACT.md` 与 sim2sim/ROS2 的关节映射同步
- [ ] real2sim 辨识后回填:关节摩擦、弹簧曲线、延迟范围
