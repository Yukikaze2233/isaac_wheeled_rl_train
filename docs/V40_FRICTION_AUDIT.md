# V40 摩擦、材质绑定与滑移可观测性审计

日期：2026-09-13。范围：Kaiser WSL 已部署 `v40-live-snapshot03`、五高度评估的现存USD缓存、本地对应源码与历史遥测；只读、CPU USD解析，未启动仿真或训练。

## 1. 可靠结论

1. **当前V40没有证据启用`multiply`摩擦合成。** 五份真实评估USD的左右轮碰撞体均未绑定physics material，走场景默认材质。实际代码链中，场景默认与地面均为`μs=0.5, μd=0.5, restitution=0`；Sim6 combine配置为`None`，已安装PhysX schema回退为`average`。据此推导的**名义轮地有效μs/μd均为0.5，恢复系数0**，不是0.25，也不是关节参数中的0。
2. **这比“只看配置数字”多了一层实际USD绑定检查，但还不是solver材质表实测。** robot缓存不含运行时`/World/ground`、`/physicsScene/defaultMaterial`或完整session层；进程已结束，不能从这些缓存假装读回当时的完整轮地material pair。下面明确分开“USD已见事实”“源码推导”和“尚缺读回”。
3. **零速漂移不等于打滑。** 最终五高度CSV没有轮速、轮world twist、轮地接触对/法向力，无法区分正常滚动、轮胎纵滑、横滑、离地空转。0.28m的非轮净力也不能证明某个轮正在接触地面。
4. **速度/yaw不能各自取上限再随意组合。** 按本机研究资产的轮接触几何中心名义轮距约0.4373m、半径0.06m和当前未测量电机曲线，5m/s直行已接近曲线零力矩速度；5m/s+2rad/s外轮可用力矩近零；5m/s+120rpm外轮超过该先验曲线速度域。它们都不是已通过的硬件速度指标。
5. 第三轮应先围绕**effective μ约0.5**做窄范围摩擦对照/DR，明确定义材质绑定和combine，不应默认“μ越大越好”，也不应将摩擦变化同时混入新的网络/奖励基线后再凭漂移判断根因。

小型机器可读证据：[v40-friction-audit-kaiser.json](evidence/v40-friction-audit-kaiser.json)。

## 2. SSH结果与版本身份

- 源码：`/home/kaiser/robot-rl-sim60/v40-live-snapshot03`，dirty快照基于HEAD `4ca58112c1ab23bea805a4d40ce707e61d244ef6`；四个相关文件SHA在JSON中记录。
- Lab：`ffff603eafc6b74264a5261cc0183d6a65390d78` / `v3.0.0-beta2.patch1`。材料配置、地面生成与PhysX默认材质四个Lab源文件，本地与远端SHA逐项一致。
- 只读检查时没有V40训练/回放进程，也没有tmux server。已知smoke仍为2env、3次PPO更新；五高度批次五个exit code均为0。本次没有产生Round3训练结果。
- USD读取使用既有runtime中的`pxr.Usd/UsdPhysics/UsdShade`，USD版本0.25.11；没有`AppLauncher`、`SimulationContext`或physics step。该CPU解释器不能直接import `PhysxSchema`，所以combine fallback通过**已安装扩展的generatedSchema.usda原文**核对，未为此启动Kit或安装依赖。

## 3. 实际材质解析链

### 3.1 V40自己的代码没有设置轮/地摩擦DR

- [`env.py:270–274`](../src/wheeled_tasks/direct/v40_serial/env.py#L270)：`spawn_ground_plane(..., cfg=GroundPlaneCfg())`，没有显式摩擦或combine override。
- [`env_cfg.py:24`](../src/wheeled_tasks/direct/v40_serial/env_cfg.py#L24)：`SimulationCfg(dt=.005, render_interval=2)`，采用场景默认物理材质。
- [`assets/v40.py:255–287`](../src/wheeled_world/assets/v40.py#L255)：URDF/可选USD生成没有设置`physics_material`；实际五高度使用run-local重新转换，不使用外部USD seed。
- [`assets/v40.py:235–243`](../src/wheeled_world/assets/v40.py#L235)中的`friction=0.0`是**actuator/关节摩擦参数**，不是轮胎与地面的库仑摩擦系数。
- 仓库另有[`manager/mdp/terrain.py:95–96`](../src/wheeled_tasks/manager/mdp/terrain.py#L95)设置地面static/dynamic=1.0，但当前Direct V40的`_setup_scene`没有调用这个manager terrain构造器；不能把闲置分支的1.0当成本次地面参数。
- [`robot.urdf:209–237,344–371`](../assets/urdf_v40/robot.urdf#L209)的轮碰撞体是半径约0.06m的Cylinder；`<material><color>`仅描述visual。不能将visual material、关节摩擦、接触摩擦混用。

### 3.2 已读取的五份USD事实

现存缓存均位于远端评估目录：

`/home/kaiser/robot-rl-sim60/round2-height-20260912T155346Z/h028…h032/usd_cache/robot/robot.usda`

对每份执行`Usd.Stage.Open()`并在真实CollisionAPI prim上调用`UsdShade.MaterialBindingAPI(...).ComputeBoundMaterial('physics')`，结果完全相同：

| 左/右碰撞体路径（前缀`/own_v40/Geometry/base_link`） | 类型 | resolved physics material | resolved binding |
|---|---|---|---|
| `/L_link1/L_link2/L_link3/L_link3_cylinder` | Cylinder + PhysicsCollisionAPI | 无 | 无 |
| `/R_link1/R_link2/R_link3/R_link3_cylinder` | Cylinder + PhysicsCollisionAPI | 无 | 无 |

左右轮各自没有直接material relationship，也没有解析到继承physics binding。七个`/own_v40/Materials/material_1…7`均无PhysicsMaterialAPI、无physics/friction属性，是视觉材质。**未发现左轮/右轮绑定不同摩擦材质。**

五个cache SHA不同，但上述绑定结果一致；具体SHA见JSON。每个已用layer集合只有对应robot文件及CPU打开时产生的空anonymous session层，不包含原运行scene/ground层。

### 3.3 默认材质、地面绑定与combine

下列路径相对所核对的IsaacLab commit：

| 来源 | 实际默认/行为 |
|---|---|
| [`RigidBodyMaterialBaseCfg:82–89`](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/sim/spawners/materials/physics_materials_cfg.py#L82) | static=0.5、dynamic=0.5、restitution=0 |
| [`PhysxRigidBodyMaterialCfg:185–202`](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab_physx/isaaclab_physx/sim/spawners/materials/physics_materials_cfg.py#L185) | friction/restitution combine默认`None`，不是显式multiply |
| [`SimulationCfg:250–259`](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/sim/simulation_cfg.py#L250) | 未指定物理材质的刚体使用场景默认材质 |
| [`PhysxManager:655–663`](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py#L655) | 创建`/physicsScene/defaultMaterial`并绑定scene |
| [`GroundPlaneCfg:210–231`](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py#L210) | 使用默认RigidBodyMaterialCfg |
| [`spawn_ground_plane:212–225`](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py#L212) | 创建`/World/ground/physicsMaterial`，绑定地面下第一个Plane collider |
| [`bind_physics_material:809–878`](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/sim/utils/prims.py#L809) | `materialPurpose='physics'`，默认`strongerThanDescendants` |

材料writer只在非None的solver-specific字段存在时应用对应PhysX schema；不能把`None`当作“无摩擦”或“采用乘法”。已安装扩展
`omni.usd.schema.physx-110.1.11+110.1.1.lx64.r.cp312.u7f4/plugins/PhysxSchema/resources/generatedSchema.usda`：

- 第754行：`physxMaterial:frictionCombineMode = "average"`。
- 第761行：`physxMaterial:restitutionCombineMode = "average"`。
- 文件SHA256：`a3ef3e946005a122734c2665c6ee27cc13304249a4605458a3b1f7ed3bb76639`。

因此当前nominal pair的源码/资产联合推导是：

```text
wheel  = scene default:  μs=0.5, μd=0.5, e=0, mode=average (fallback)
ground = ground material: μs=0.5, μd=0.5, e=0, mode=average (fallback)
effective μs = (0.5+0.5)/2 = 0.5
effective μd = (0.5+0.5)/2 = 0.5
effective e  = (0+0)/2     = 0
```

PhysX的combine优先级为`average < min < multiply < max`；两端模式不同使用较高优先级，再分别合成静摩擦、动摩擦或恢复系数。选择multiply时0.5与0.5才得到0.25，但它不是当前V40的默认链。

**证据边界：** 这不是对已经结束的PhysX材质表、接触patch实际摩擦利用率的读回，也不是材料实验测得的轮胎μ。完整运行USD/session层与solver材料读数均未保存。最终验收仍应保存真实scene绑定和物理读回，而不是只记录这个推导值。

原Round2冻结代码`453cd1c`也使用默认GroundPlaneCfg/SimulationCfg；本地保留的Lab2.3.0 `physics_materials_cfg.py:38–58`默认同样是0.5/0.5/0与显式average。没有证据表明Sim5.1→Sim6时V40默认μ从0.25变成0.5；底层cooking/contact差异需另证。

## 4. 上游启用内容与R3摩擦范围

### 上游参考不能只抄系数

- 华南虎commit `b8ff79f3df855faf9dc92f4a282bd80c42649466`：
  [V14轮材质DR](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py#L160)启用static[0.5,1.2]、dynamic[0.4,1.0]、restitution[0.02,0.2]、consistent。
  [flat祖先地面](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env_cfg.py#L267)显式static/dynamic=1、restitution=0、combine=multiply。
  对这套绑定链，乘以地面1才让轮系数保留相同数值；不能移到V40的地面0.5后继续声称effective范围相同。multiply优先于average，亦不能对不同mode简单各算一半。
- 复旦commit `8204e853dfd2ed06d85a322e1a998c3d20a3be2c`：
  [plane配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot_config.py#L50)的地面static/dynamic=0.5、restitution=0.5；asset shape摩擦DR[0.6,1.4]、restitution[0.6,1.0]在
  [创建shape时应用](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot.py#L438)。这些是shape/plane配置数字；本次没有其IsaacGym runtime pair材质读回，不把[0.6,1.4]直接称为effective轮地μ。

### R3建议（只设计，不实施）

1. 冻结当前nominal对照：effective静/动μ约0.5、恢复系数0，左右轮一致；显式记录combine及最终绑定，不以visual材质名替代。
2. 初始有效μ目标可围绕 **[0.4,0.6]**，保留足量nominal样本，再用独立较低摩擦留出场景检验泛化。当前average、ground0.5时，若只随机化wheel，要实现effective[0.4,0.6]，wheel应为 **[0.3,0.7]**。这只是数学映射，不是实物材质校准。
3. 如果以后选择ground=1、multiply，就需另设wheel系数来保持相同effective baseline；必须标记合成策略变更。不要同时改ground和wheel而忽略其组合分布。
4. 保持`μd<=μs`；先不引入复旦式高回弹。先做窄范围、可解释的摩擦变化，再决定是否分开静/动系数或左右非对称DR。
5. **大μ不一定解决漂移。** 无滑移滚动也能持续漂移，策略的零速偏差、动作饱和、接触几何都可能相关；增大μ还可能加大横向约束、轮地力矩、内部接触反力和数值敏感性。尤其0.28m的持续非轮净力应先定位pair，不能靠增大轮地摩擦掩盖。

## 5. 现有遥测能否计算打滑

| 数据 | 已有内容 | 能否形成滑移验收证据 |
|---|---|---|
| Round2最终五height CSV | root位置、body系root COM速度、wz、gravity_z、非轮净力、diagnostics | **不能**：没有轮dq、轮world twist/姿态、轮地pair、接触点/法向力；只能量化漂移 |
| 最终GUI/teleop CSV | 类似root速度和命令，非轮净力 | **不能**；action norm/velocity target不是实测轮速 |
| TensorBoard训练曲线 | 跨环境/rollout聚合跟踪误差、净力候选、reward | **不能**：丢失每环境左右轮的同一时刻状态和支持接触关系 |
| model_4200 `fixed_command.records.json` | q/dq、root quat/COM twist、wheel-axis midpoint、两轮净力模 | 可做带URDF/FK假设的**运动学proxy**；无ground pair/真实接触点与轮world twist，而且不是最终checkpoint |
| Kaiser 3update live viewer | 七body姿态、低频墙钟采样、command/action，post-step可能已reset | 可做有限的姿态差分检查，不能验收最终策略滑移：轮角高速采样混叠、接触gate缺失、reset边界不同 |

对应源码：[`env.py` snapshot](../src/wheeled_tasks/direct/v40_serial/env.py#L161)、[`play_v40_onnx.py` CSV](../scripts/play_v40_onnx.py)、[`live_view.py:314–324`](../src/wheeled_tasks/v40/live_view.py#L314)。snapshot内存里有q/dq，不代表它们已保存在最终CSV中；不能从过去丢弃的字段补造数据。

现有contact sensor甚至显式拒绝`filter_prim_paths_expr`、`track_contact_points`、`track_friction_forces`，并使用空`filter_patterns`：见[`contact_sensor.py:12–14,43–45`](../src/wheeled_tasks/direct/v40_serial/contact_sensor.py#L12)。**净力>阈值只能是contact candidate，不是轮地支持gate。**

## 6. 应如何定义与记录滑移

### 6.1 先计算接触点相对地面的速度

对每个经pair身份确认的轮地接触点，全部使用同一世界坐标系与同一物理时刻：

```text
v_rel(p) = v_wheel_COM_world
         + omega_wheel_world × (p_contact_world - x_wheel_COM_world)
         - [v_ground_COM_world + omega_ground_world × (p_contact_world - x_ground_COM_world)]
```

静止平地的方括号项为0。`omega_wheel_world`必须是**整个轮刚体的总角速度**，不能只用编码器相对关节dq替代，后者没有包括root/髋/膝运动。r是接触点到**轮COM**的矢量，不是固定`[0,0,-0.06]`，也不是到wheel-link origin：本资产轮COM/圆柱中心均有轴向偏置。

取地面法向n，将轮轴a投影到地面切平面得到横向方向，再由叉积得到纵向滚动方向，并用机体forward统一正负号。计算：

```text
slip_long_m_s = dot(v_rel, t_long)
slip_lat_m_s  = dot(v_rel, t_lat)
```

只在ground pair有效且法向支持力/冲量超过定义阈值时统计支持滑移；离地期间单列空转/失去支撑时间，不能当0滑移。阈值应按正常载荷、传感器噪声及接触滞回制定，不能把现有净力1N原封不动当ground支持判定。

若需要ratio，先定义统一符号和速度参考：常见proxy为`(v_roll-v_center_long)/max(abs(v_roll),abs(v_center_long),v_floor)`。**静止附近必须优先报告m/s，不报告被近零分母放大的百分比**；设速度有效域后再统计ratio、valid frame比例。横向滑移可报m/s，侧偏角也应有低速gate。多点接触不应把相反符号速度先平均而抵消；可按正常载荷加权绝对值并同时给P95/峰值。

### 6.2 左右轮符号与旋转工况

当前canonical URDF nominal FK中，L_joint3正轴约`+Y`，R_joint3正轴约`-Y`：

```text
近直立平地、忽略腿运动时：v_roll_L ≈ +0.06*dq_L
                         v_roll_R ≈ -0.06*dq_R
```

这来自[`robot.urdf:231–236,366–370`](../assets/urdf_v40/robot.urdf#L231)的关节axis经整个链的变换，不是仅凭关节名猜测；实物encoder零点/方向尚未确认。

旋转时不能把两个轮都与root vx比较。对轮中心中点参考的理想差速模型：

`v_L = v - yaw_rate*b/2`，`v_R = v + yaw_rate*b/2`。

更一般必须用`v_reference + omega_root × offset + leg_Jacobian*dq`得到各轮中心速度；root COM与轮中心中点并非同一点。纯旋转时root vx≈0而两轮速度相反是正常现象，不是两轮同时打滑的证明。

### 6.3 最少需要增加的遥测（交由主方案决定实现）

1. **材质身份**：实际碰撞shape与ground collider路径、physics material解析路径、μs/μd/restitution、两端combine、最终pair nominal值；逐环境DR实际采样值；episode reset后若随机化则记录更新。
2. **左右轮运动**：具名q/dq、wheel COM世界位置、轮姿态/轴方向、COM世界线速度、总世界角速度；root完整pose/twist及腿q/dq以交叉校验，明确xyzw/wxyz及COM/link坐标含义。
3. **支持接触**：ground-pair身份、接触点p、法向n、normal force/impulse、friction/tangent force或impulse、dt；逐轮接触有效性及法向载荷。solver若只提供冲量，要除以对应physics dt，不用policy dt混算。
4. **执行能力**：policy原输出、执行wheel target、实际applied torque、当前速度对应的力矩上界与饱和flag。不能用policy target反推实际轮速。
5. **时间与质量**：physics tick、policy tick、episode id、pre-reset状态、command、reset原因；接触history每样本时刻。基于同一physics时刻联合计算，不能将history最大力与另一时刻的dq任意配对。
6. **指标输出**：纵/横滑移速度MAE/P95、接触有效占比、离地空转比例、friction utilization `norm(F_t)/F_n`与转弯/加减速分层；漂移继续独立统计。`F_t/F_n`也不是材料μ的直接估计，未达摩擦极限时二者不会相等。

## 7. 本机几何与高速/yaw预算参考

以下全部是**当前研究资产与电机先验的理想化算术参考，不是实机确认，也不是已执行高速测试**。

### 7.1 用圆柱中心轮距，不误用关节原点间距

CPU对当前URDF按v2 nominal q做FK：

- 左圆柱几何中心base坐标：`(-0.0128894, +0.2179684, -0.2599715) m`。
- 右圆柱几何中心：`(-0.0128894, -0.2193363, -0.2600040) m`。
- **名义轮距b=0.4373047m**；轮body/joint origin间距仅0.3966051m。差值来自左右轮约20.35mm轴向偏置，不能把两者混为一谈。
- 两轮轴略有约0.00455rad倾斜；严格接触patch轮距会随姿态/触点变化。此处用圆柱中心轮距作差速预算，不声称是实物尺量或瞬时触点轮距。
- 半径R取0.06m。与现有[`v40_design_geometry.json:256–266`](../assets/urdf_v40/evidence/v40_design_geometry.json#L256)记录一致。

### 7.2 电机曲线与牵引限制

[`own_v40_v2.json:58–64`](../contracts/own_v40_v2.json#L58)的轮曲线为未测量的线性研究先验：ratio11、motor侧999.063rad/s时扭矩降至0；映射至wheel joint：

```text
omega_0 = 999.063 / 11 = 90.82391 rad/s
v_0     = R*omega_0    = 5.44943 m/s（零可用力矩边界，不是可达稳态速度）
tau_max(v_i) = 3.8376866 * max(0, 1 - abs(v_i)/5.44943) Nm
```

这是[`core.py:187–231`](../src/wheeled_tasks/v40/core.py#L187)执行的速度相关扭矩包络；表中扭矩均是**轮关节输出侧**，不是转子侧。`velocity_limit_sim=1e9`只是撤掉URDF占位限速，不提供无限物理速度能力。

| 中点v m/s | yaw rad/s | 左/右轮地面速率 m/s | 左/右轮关节可用扭矩上界 Nm |
|---:|---:|---|---|
| 2 | 0 | 2.000 / 2.000 | 2.429 / 2.429 |
| 2 | 2 | 1.563 / 2.437 | 2.737 / 2.121 |
| 5 | 0 | 5.000 / 5.000 | 0.317 / 0.317 |
| 5 | 2 | 4.563 / 5.437 | 0.624 / **0.00854** |
| 0 | 12.566（120rpm） | -2.748 / +2.748 | 1.903 / 1.903 |
| 5 | 12.566（120rpm） | 2.252 / **7.748** | 2.252 / **0** |

仅考虑曲线速度域，理想差速约束为`abs(v)+abs(yaw)*b/2 < 5.44943`，还必须留出扭矩余量。5m/s时此几何上界允许yaw约2.055rad/s，但靠近它的外轮已经几乎没有可用力矩。

当前URDF总质量12.752kg。**若**平地左右轮对称承载全部重量，Fn每轮约62.55N；μ=0.5时单轮纵横向合成牵引上界约31.27N、等效轮扭矩约1.876Nm。低速电机包络可高于牵引界，可能先受附着约束；5m/s时又可能先受电机余量约束。实际Fn随高度、平衡、转弯和非轮接触变化，不能把这个假设用于证明0.28m工况支持正确。

平地近稳态转弯的COM侧向加速度近似`abs(v*yaw)`，需要与纵向加速度共享附着预算：

`sqrt(a_long² + (v*yaw)²) <= μ*g`（理想化必要条件，非充分条件）。

- v2/yaw2：侧向4m/s²，最低μ约0.408；μ0.5下留给纵向加速度的理想余量仅约2.84m/s²。
- v5/yaw2：侧向10m/s²，最低μ约1.019，已超过当前名义0.5。
- v5/yaw120rpm：侧向约62.83m/s²，最低μ约6.405，而且外轮已越电机曲线域。
- v5、μ0.5时，仅这一侧向条件给出的yaw上界约0.981rad/s，比上述曲线几何上界更小。

纯旋转v=0不表示“不需要摩擦”：还需要左右轮牵引形成yaw力矩、克服旋转惯量/损耗，可能发生横滑和载荷转移；也应考虑高位姿态的倾覆/平衡裕量。电机损耗、真实转速-力矩、电压、电流、轮胎形变和实际COM均未确认，不应据表直接提高R3速度或实物上限。

## 8. 收尾

只新增本文及小JSON证据。核查过程中只读远端文件、CPU解析既有USD、做URDF FK和预算计算；没有改训练代码/材质/合同，没有运行GPU physics或长测试，没有commit/push。最优先的下一步证据是**实际材料绑定/solver读回 + 有ground支持gate的左右接触点滑移速度**，不是从已有零速漂移反推摩擦不足。
