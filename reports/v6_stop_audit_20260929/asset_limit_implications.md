# V6 名义 105° 止挡：资产与辨识影响审计（2026-09-29）

用户新确认：**105° 同时是原标定姿态的膝内角和名义机械上限；实际误差可能使有效上限达到 106°／107°。** 这覆盖先前“105° 与 110° 尚待一致性确认”的物理先验；不追溯改写旧资产和旧验证记录。

本审计保持 final_r3 全部内容及 manifest SHA `9542c788afd6e1b7f0af5e396062f123e511f63579cd8807087068e82e401c5c` 不变。只新增本目录文档、`check_stop_mapping.py`、`mapping.json`。没有硬件动作、仿真 rollout、训练或 runtime 修改。

## 1. 被动 DOF 与几何内角的精确映射

在已验证的装配分支、公共轴法平面的 pin-center 角度定义下：

```text
β0 = π − 2.3573 = 0.7842926535897932 rad = 44.93665895381104°
βL = β0 + q[L_joint2]
βR = β0 − q[R_jonit2]
```

`L_joint2`／`R_jonit2` 是实际被动膝铰链，不是 `LL_joint1`／`RR_joint1` 主动膝输入。主动输入到 β 的关系仍是闭链非线性 FK。

40° 下限不变时，左范围为 `[rad(40)-β0, rad(βmax)-β0]`，右范围为其反序取负。具体数值：

| 几何上限 βmax | L_joint2 upper (rad) | R_jonit2 lower (rad) | USD 左 upper／右 lower (deg) | 闭链 AB (mm) | 压缩 (mm) |
|---:|---:|---:|---:|---:|---:|
| 105° | +1.048303061004253 | -1.048303061004253 | ±60.063341046189 | 212.377347 | 19.622653 |
| 106° | +1.065756353524196 | -1.065756353524196 | ±61.063341046189 | 213.209105 | 18.790895 |
| 107° | +1.083209646044139 | -1.083209646044139 | ±62.063341046189 | 214.035706 | 17.964294 |
| 原 110° | +1.135569523603969 | -1.135569523603969 | ±65.063341046189 | 216.482680 | 15.517320 |

左 lower 固定 `-0.08616095279206137 rad`，右 upper 固定 `+0.08616095279206137 rad`；USD 对应 `−4.936658953811°`／`+4.936658953811°`。**USD 角限位采用度，但数值是关节坐标角，不能直接写 105。** URDF/MJCF/spec 均采用弧度；PhysX `get_dof_limits` 读回采用弧度。

本目录脚本在两组髋姿态（左 -0.42 和 +1.6 rad，右镜像）、两腿、四个候选上限共 16 个闭链姿态复核；独立 pin-center FK 内角误差 <1e-8°，最大闭链残差 `7.76e-14 m`。同时检查当前 spec、URDF、MJCF 的原范围一致，读取并记录 USD 原度数。气簧 232 mm 全伸、80 mm 行程、安装点和力曲线无需因上限变化而改变；被允许访问的角域/行程子域会缩小。

## 2. 新版资产需同步更新的地方

下列是新版生成/验证清单，不是修改 final_r3 的授权或补丁。

| 路径（训练仓库内） | 现状、下一版处理 |
|---|---|
| `model/纯底盘_v6_232mm/inputs/mechanical_limits.json:3–5` | 唯一显式几何先验当前为 `[40,110]`、旧用户来源。新版输入应明确 `[40,105]` 名义域、2026-09-29 用户确认及有效上止挡 105–107 的研究不确定域；二者不能含糊地共用一个数组。 |
| `model/纯底盘_v6_232mm/tools/build_v6_closedchain.py:133–150` | 已按输入参数转换左右 passive range，无需改变公式。生成新版时用新输入；没有把主动膝轴直接设为105的依据。 |
| 同 builder `:464–467` | manifest 的限位值/来源已参数化，但 `rmcs_105_deg_record` 仍硬编码“未授权改变40–110”，新版必须更新这条语义。根 tools 与导出 bundle/tools 副本同步由生成流程处理。 |
| 新 bundle `mechanical_limits.json`、`model_spec.json:207,249` | 输入快照及真正的左右被动限位必须一起变化；否则 runtime 与物理引擎将不一致。 |
| 新 bundle `robot.urdf:524,532`、`robot.xml:82,118` | 由 spec 再生成两膝范围；质量、惯量、轴、mesh、闭链 anchor 不因本次上限确认变化。 |
| 新 bundle `robot.usda:1370–1385` | 再生成度数限位。原 exporter `mechanism_export_legacy.py:558–560` 做 rad→deg；没有额外减 nominal。 |
| 新 bundle `manifest.json`、`build_manifest.json` | 更新几何上限、来源、仍未实测的 uncertainty 描述及全套文件哈希。新的资产版本/路径与 parent identity 明确，旧 hash 保留可复算。 |
| `tools/validate_v6_asset.py:116–129`、新版 `cpu_validation.json` | 端点与75姿态域从 manifest 读取，不存在需手改的110循环常量；新输入自动变成40–105。另把106/107作为有效止挡候选覆盖，不能将它们混称名义合法域。 |
| `validation.json`、`dynamics_validation.json`、`physx_validation.json`、`physx_input_manifest.json`、`tools/seal_v6_asset.py:47–65` | 新物理限位使输入 hash 改变，原 MuJoCo/PhysX 动力学成功回执不能原样继承成“新版通过”；重跑适当边界探针并再封装。旧 input manifest 只能留为历史来源。 |
| 根及 bundle `README.md:25,80` | 40–110 当前描述是历史真实记录。新版文档改为名义40–105、有效上限研究域105–107，明确标定与止挡不确定性。 |

**不应改的历史来源**：原始下载及 `source/urdf/urdf_v5.0.urdf`、`source_urdf_v5.0.urdf` 是 CAD 快照；本次限制由 mechanical_limits 输入覆盖，不是改原始 CAD。`inputs/v5_232mm_reference_spec.json` 和 bundle 同名 reference_spec 的110限位属于冻结旧对比模型，也不能为使搜索“没有110”而抹改。其旧限位不用于新版 limit authority。

## 3. 资产外的消费方与缺口

- `scripts/audit_v5_motor_coupling.py:30–31` 当前硬性只接受 `[40,110]`，且 `:88` 域采样写死110。`identify_pair_dynamics.py` 和反馈采样 ablation 调用它，因此真正加载新105资产会失败。新版工具应根据经过验证的 manifest/spec 检查范围并采样域，而不是删除所有输入检查；旧V5测试保持原合同语义。
- `src/wheeled_tasks/chassis/v5_control.py:23,73` 从 spec 建立 `knee_bounds`，用于工作余量风险；`env.py:354–362` 从 spec 建立物理/终止边界并检查 PhysX 读回，`env.py:1118` 的硬终止另容许 **0.03 rad≈1.719°**。这个容差不等于用户确认的上限，也不应被用来代替105–107不确定域。
- 当前 `knee_bounds` 设计是共享左右2×2数组/标量 dict。如果未来每环境随机止挡，必须同步 PhysX 每环境限位、控制器余量、终止判定和回执；仅随机物理属性或仅扩大 termination 会让物理与验收矛盾。这里没有实施随机化。
- `contracts/v5_height_workspace_v2.json` 冻结的是旧 spec hash 和40–110域；`scripts/probe_reference_takeoff.py:33` 还有110根搜索上界。新高度域、参考动作、起跳/落地与自起轨迹须在新资产/新限位上重新核验。不能因为表里有同一高度数值就沿用旧可达性和关节余量风险。
- `inspect_v5_activation.py` 的 split/extra helper 默认40–110，但主要调用已传 manifest；105标定相关测试与旧110机械端点区分应保留为历史合同测试。新增新版测试验证105是名义上限，并分别覆盖106/107候选。
- 现 `tests/test_v6_asset_bundle.py` 检查 hash/坐标/闭链/执行器，但未显式对照四种导出格式的两膝端点。新版最小补充：manifest→spec→URDF/MJCF/USD一致、左右符号正确、FK端点正确、运行时实际限位读回一致，以及针对上止挡的有限步接触探针。无需重造庞大验证流程。

## 4. 与既有 API 零位及静载的关系

前次独立审计 `reports/v6_zero_pose_audit_20260929`：API q=0 按现 offset 映射后，左 β=105.021671°、右105.021166°，已经很接近名义105；初始静载左106.984128°、右105.732977°。这与“有效上限可能106/107”相容，但不能从相容性推出左右真实止挡就是上述两个数值。

| replay 中设定的有效上限 | 左初始几何超出量 | 右初始几何超出量 |
|---:|---:|---:|
| 105° | +1.984128° | +0.732977° |
| 106° | +0.984128° | -0.267023°（域内） |
| 107° | -0.015872°（域内） | -1.267023°（域内） |

新固定105边界连 API 零位先验本身都约有0.02°轻微穿透，不能以“一帧限位穿透”判定硬件校准错了。原始 q 初始化需保留并记录初始穿透、limit force/impulse、最大穿透、退出接触时间、闭链误差和速度；不要先夹紧 q 到界内后把降低的 replay RMSE 当成验证改善。被动 stop 是引擎的有限刚度/数值约束，jnt_range=105 不保证积分时绝无超过105的状态。

父任务计划在独立 MuJoCo replay 对象仅改 `jnt_range` 的105/106/107 bounded对照是合适的机制诊断。它不改变 sealed资产，不证明测得真实止挡，也不宜同时再调offset/摩擦/气簧倍率使效应混合。保持提交力矩、参考、重力、原初态等相同并分项报告。若加入原110对照，明确它仅复现历史假设。

## 5. 如何避免止挡与 encoder offset 重复随机化

设录包映射的名义主动坐标为 `q_nom`，未知真实零偏为 `b`，真实坐标为 `q_true=q_nom+b`，实际内角为闭链函数 `β=f(q_true)`。接触条件为：

```text
f(q_nom + b) = β_stop_physical
局部近似：β_stop_effective_in_nominal_coordinates ≈ β_stop_physical − Jβ b
```

因此在只有电机录包、没有独立膝角/销距测量的情况下，**有效坐标止挡与物理止挡角、encoder零偏不能唯一分离**。已有“在止挡姿态设零”的流程还会使实际止挡误差和零位误差相关，不能独立均匀抽两份同一来源的不确定性。

推荐两层做法：

1. **当前机制研究**：固定现有offset/sign，以“名义坐标里的有效上止挡”作为唯一此类自由量，先做105/106/107离散对照；不另加造成膝内角偏移的 encoder differential bias。105–107是用户给出的可能域，不是测得的概率分布，不能声称均匀分布已校准。左右分别分析，暂不强迫同一个上限。
2. **未来训练/独立标定**：若有独立物理角度、销距或多姿态标定证据，再分别参数化物理 stop 与 encoder bias，并用共同 latent calibration/条件关系采样。例如零位来自同一次触碰止挡，必须同时满足 `f(q_zero_nom+b)=β_stop_physical`，然后推导名义坐标边界；不可再独立随机一个 effective stop。每次参数抽样在整段 episode/设备实例保持固定，不能每物理帧移动边界。

同侧两个主动轴的共模 bias 主要转动整条腿，几乎不改变内角；差模 bias 才显著改变 β。当前静载处 `Jβ` 左约 `[-0.881818,+0.881819]`、右约 `[+0.880129,-0.880126]`，所以不能把“±1° encoder误差”机械地当成“±1°膝角误差”。共模 bias 还影响重力/大腿方向，应该由独立姿态证据约束；不能由膝止挡数据凭空确定。

## 6. 复现与本次验证

```bash
OPENBLAS_NUM_THREADS=1 python reports/v6_stop_audit_20260929/check_stop_mapping.py
python -m py_compile reports/v6_stop_audit_20260929/check_stop_mapping.py
```

本次通过所有 frozen文件 SHA、当前三种弧度导出一致性、四个候选的16姿态 FK/闭链检查；final_r3 manifest前后相同。该脚本提供可复用的纯CPU `passive_range_for_inner_limits(40, upper)`，只返回研究候选范围，不改源 spec 或资产。
