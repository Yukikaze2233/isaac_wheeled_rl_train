# 资产:URDF → USD

本仓库不含模型资产。`wheeled_world/assets/__init__.py`
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
