# V40 canonical 资产链与离线静态验收

## 当前选择：独立串联等效研究模型已获明确批准

用户在 `ask_user_question`（id=`v40-equivalent-research-scope`）明确选择：
**“同意，先用串联等效研究模型训练”**。这项新决定仅在下面独立研究范围内
替代此前暂停决定，不更改真实CAD，也不撤销左右髋材料/装配待审结论。
**本次仅创建变体记录和验证，未上传、训练、运行Isaac或做实机部署。**

|层|文件|collision_validation.passed|含义|
|---|---|---|---|
|原始材料审查|`manifest.json`|false，字节不变|两髋材料/装配与body归属未确认，不能当实物模型认证|
|明确用户批准|`research_model_approval.json`|不是碰撞修复报告|仅允许六对直接关节连接体内部接触不建模|
|独立研究变体|`research_manifest.json`|true，仅新范围|串联等效研究模型的已有静态检查通过；并非Isaac/硬件就绪|

研究 `scope_id=own-v40-equivalent-serial-internal-contact-v1`；碰撞检查范围严格是
`equivalent_serial_research_with_explicit_internal_joint_exclusions`。仅省略本文件
后文列出的6对内部接触，其它自碰撞与地面碰撞仍启用，膝35°～80°保留。
质量、惯量和link刚性归组明确作为**研究动力学先验**，不是已校准实物真值；
原inspection的1Nm诊断motor也不是训练或硬件性能批准。

### 研究批准与原失败状态分层

研究清单每对保留原 `geometry_review_supported`（两髋仍false），另添加
`research_exclusion_approved=true`、`policy_status=user_approved_joint_internal_contact_exclusion`；
原策略保存于 `raw_policy_status`。这表示“明确选择不建模这些内部接触”，
**不表示源材料交叠已修复**。原880点、独立右髋材料证据和73条历史接触不动。
原raw层的整体禁训文字作为历史保留；不得将其覆盖成材料审查通过，也不能
将历史暂停误读为撤销了这次单独明确批准的研究模型。

批准记录来源为 `approval_source.kind=explicit_user_decision`，带上述question_id
与完整选择文本；它是SHA绑定的用户决定记录，不是数字签名。
原manifest SHA256为：
`ec93ff2bab06b817796338268d9640754849c49796cec88138b1d24b5a6ce196`。
批准文件 SHA256为：
`a11358cd9703715a46b00ac5cc50b5ef70fa7a9989ac191bc2854ca328944c73`。

`research_manifest.research_model` 包含 `scope_id`、`approval_file/approval_sha256`、
`raw_manifest_file/raw_manifest_sha256`、`approval_source`、`dynamics_prior`、
`hardware_deployment_approved=false` 与当前研究生成器指纹。

散列依赖无循环：原manifest仍只覆盖其原66个文件，**不回填新增研究文件**；
研究清单 `files_sha256` 为原66项 + `manifest.json` + `research_model_approval.json`，
共68项，不包括研究清单自身。原全部67份raw文件保持字节不变。原provenance
的生成器指纹属于历史构建记录，不重写；新增研究清单另记录当前emitter版本。
URDF、MJCF、STL、凸件、质量与姿态全引用同一原文件，无第二套几何或路径外依赖。

已有SHA绑定的CPU静态记录与独立结构/解析净空复核提供10项门：非邻接最小
净距+6.50853687mm、两轮代理地面距离−27.8/−60.3µm（显式仅轮0.1mm容差）、
硬膝限位、原站姿、质量/坐标、精确6对排除及其他碰撞保留等均通过。
`isaac_cooking_tested=false`、`server_adjacency_filter_verified=false`、
`hardware_deployment_ready=false`；没有全工作域、动态平衡或实机认证。

### 显式生成与可复现性

```bash
python -B tools/prepare_v40_assets.py \
  --raw-assets assets/urdf_v40 \
  --research-approval assets/urdf_v40/research_model_approval.json \
  --out assets/urdf_v40
```

没有显式 `--research-approval` **绝不生成研究变体**；它必须与 `--raw-assets`
成对使用，不能从有限无见证样本推断授权。普通 `--source ... --out 新目录`
仍生成原始raw审查层（双髋pending、passed=false），不会自动携带研究批准。

研究记录必须与其不可变raw快照同目录，以保持相对引用。若要在新目录重现，
先逐字节复制原raw快照，再显式提供同一批准文件。生成器校验原manifest SHA、
所有被绑定原文件、精确6对/原geometry flag与批准边界；不自动重定向批准hash。
相同研究记录可幂等重现，不同已存在记录拒绝覆盖。仅生成JSON，不导入模拟器。
即使审批有效，只要非邻接净空、地面或硬限位等真实门失败，研究passed仍false。

研究变体最终回归：现有MuJoCo环境 **45 passed（12.44 s）**；系统Python
**44 passed、1 skipped（12.52 s）**，跳过的是该环境无MuJoCo的原静态运行时项。
两环境研究JSON均逐字节重现，覆盖缺审批/错raw SHA/增减排除对/抹除原geometry
失败标志/放宽地面或非邻接门等拒绝案例，并证明emitter不导入模拟器或改原文件。

调用方应**显式选择研究清单**并校验批准，再为Isaac按名称接线6对窄过滤；
这属于父代理的contract/导入实现。不能用全局关闭自碰撞代替，不能因研究
静态门通过就绕过服务器cooking和实际过滤生效检查。

## 原始 raw 审查状态（历史与材料结论保持不变）

`assets/urdf_v40/manifest.json` 保持 `collision_validation.passed=false`。
新只读对照在右髋也找到了源材料内部见证，**原实际 nominal 姿态亦存在**；
不能继续用旧有限样本中“右侧未发现”自动批准 base_link / R_link1 过滤。
在该raw审查记录中 **L_joint1 与 R_joint1 均为 pending material/assembly review**，
当时用户选择先修CAD/明确部件归属、暂不V4训练；该整体暂停已由上方新决定
仅在独立等效研究范围内替代。左右台阶不同不等于只有左侧重叠，
也尚不能判断哪侧 CAD 正确或哪一部分应该删除。

此前的纯状态同步及本次独立研究记录均未修剪、镜像、布尔切料、修改几何、
质量、惯量、限位、姿态或六对过滤的物理实现。49 个 URDF/MJCF/SRDF/STL/OBJ
物理文件与5份既有静态/历史诊断逐字节保留，旧880点与73条历史数据不丢弃。

`isaac_cooking_tested=false`、`server_adjacency_filter_verified=false`、
`hardware_deployment_ready=false`。未运行训练、Isaac、GUI、动力学积分、SSH、
安装或源码编译；仅现有 numpy/scipy 与 MuJoCo 3.12.0 CPU 静态 `mj_forward`，
仿真时间及步数均为 0。测试通过不会更改门禁。

## 来源与可移植性

- 输入是已确认 `own-v40-knee35-80/robot.urdf`，SHA256 为
  `835352595029eccc1dee951d4192f6dca4b00674d54e77290f79778cdb399431`。
- 生成器固定输入 URDF 和 7 个 STL 的散列。`source.urdf` 与 `meshes/*.STL`
  均保留源字节；未动父仓旧代码、旧资产、Downloads、质量或惯量。
- `evidence/` 保存源方向/硬件合同及既有设计证据。
  `evidence/revision1_static_validation.json` 是本仓上一版报告的逐字节副本，
  **73 条未过滤穿透诊断全部保留**；相应旧 manifest、碰撞审计也已存档。
- `evidence/revision2_adjacency_review.json` 冻结旧880点审查和当时右髋有限采样
  推断；当前报告保留这些原数值，但将其与新的材料/装配待审结论分开。
- `evidence/hip_material_review_hold.json` 是源 URDF/STL hash 绑定的可移植摘要，
  记录右侧原 nominal 见证、切片复核、上游报告散列及 CAD-first 决定。该摘要
  同时嵌入生成器，不需要原报告/临时目录或 `--evidence` 才能保持两髋 pending。
- `evidence/hip_review_status_sync_baseline.json` 保存此次纯状态同步前的49份
  物理文件与保留诊断散列，用于验证本轮没有偷偷改形状或覆盖历史。
- 当前 `robot.urdf`、`inspection.xml` 是相对路径的活动模型；
  `inspection_whole_hulls.xml` 与 `inspection_component_mesh_baseline.xml`
  仅作历史静态对照。旧轮凸件移入 `evidence/legacy_wheel_hulls/`，不作为活动
  模型的 collision 使用，不能拿历史对照 XML 启动训练。
- `files_sha256` 覆盖整个产物目录，排除 manifest 自身；`provenance.json`
  记录输入、生成器和证据散列。重建不依赖父目录 Python 模块或 Isaac SDK。

## 坐标、质量与名称

CAD 前 −Y、左 +X、上 +Z；canonical 是 `Xforward_Yleft_Zup`。仅将
`Rz(+90°)` 左乘到根 visual/collision/inertial origin 及两根髋关节 origin，
不重复转子 link，不改局部轴或约 0.261° 安装倾斜，也不是只改 spawn quaternion。
根 inertial origin 携带旋转，原惯量数值系数不再重复旋转。

质量仍 **12.752 kg**。根 COM 为 `[-0.005,0.0003,-0.053] m`，有效惯量 kg·m²：
`[[.178,-.000264,-.004123],[-.000264,.126,-.00049],[-.004123,-.00049,.255]]`。
32 组带多圈髋/轮的 URDF FK/COM/inertia 协变误差 <1e−12；MuJoCo FK <1e−12，
COM 误差 0，fullinertia 编译重构误差约 5.96e−9。无几何自动质量估算。

动作序：`L_joint1,L_joint2,L_joint3,R_joint1,R_jonit2,R_joint3`。
**保留拼写 R_jonit2；关节、qpos、dof、actuator 都按名字解析，不依赖导入索引。**
髋与轮 continuous，原 ±3.14 标签不是硬位置止挡；仅两膝 revolute：

- 左 q ∈ [−0.1267274153917777, 0.6586707480056704]。
- 右 q ∈ [−0.6119707480056704, 0.1734274153917776]。
- 机械内夹角 β=π+origin_rz+axis_z*q，范围35°～80°，伸直为180°。

`nominal_base_height_m=.32`；`nominal_joint_pos` 未改：

```json
{"L_joint1":0.41526541073209217,"L_joint2":0.44890796255584864,"L_joint3":0.0,"R_joint1":-0.42250891703703075,"R_jonit2":-0.4129956195280641,"R_joint3":0.0}
```

该候选在新膝限位内，但仍 blocked。不抬高到 .36/.40 m（旧两姿态越限），
也不改变 q 去掩盖本轮问题。MJCF `nominal_candidate` keyframe 保存该 raw q；
默认零关节角不是站姿。1 Nm motor 仅作检查格式，不是训练或硬件额定值。

## 轮圆柱：测量值与地面微小包络误差

活动碰撞共34个非轮组件凸 OBJ + 2个 URDF/MJCF cylinder。原 visual 与 inertia
未改。非轮组件按完全相同顶点连通，焊接容差0，逐组件 Qhull 包络，不删件、
不缩 box、不 joggle；每个源顶点/三角形均登记覆盖，越界 <1e−9 m。

两轮以原关节局部 ±Z 轴为圆柱轴；STL 外环拟合检查轴心一致，半径取**全部
源顶点最大径向距离**，轴宽和中心取原顶点轴向极值，保证不缩小：

| 轮 | 半径 m | 轴宽 m | 中心局部 z m | 原 .32m 姿态的圆柱最低 z m |
|---|---:|---:|---:|---:|
| 左 | .060000003250648436 | .02500000037252903 | +.020350000821053982 | −.0000278138164663 |
| 右 | .06000000315456387 | .02500000037252903 | −.020350000821053982 | −.0000603223657797 |

MJCF `size` 使用半轴宽，不误用全长。当前 collision 不再烹饪720顶点轮 mesh；
活动非轮凸件均通过保守255顶点离线筛查，但仍须服务器验证 PhysX cooking。

原视觉轮面仍与地面相切；包络圆柱覆盖轮胎倒角和安装倾斜，产生27.8/60.3µm
的微小轮地穿入。审计明确采用**仅轮代理0.1mm**地面容差，非轮仍1µm；
该容差、实际负距离及理由均写入 manifest，没有抬高、缩形或声称无穿入。
六对过滤后只有这两个轮地接触，非相邻最小几何净距 **+6.50853687mm**。

## 六对邻接过滤及实际几何审查

`adjacent_collision_filter_pairs` 在 manifest 按名列出，另输出
`collision_filters.srdf`。当前 MJCF 显式设置这六对 exclude，并**关闭隐式
parent filter**，从而单独验证六对表的效果：

1. base_link / L_link1 — L_joint1
2. L_link1 / L_link2 — L_joint2
3. L_link2 / L_link3 — L_joint3
4. base_link / R_link1 — R_joint1
5. R_link1 / R_link2 — R_jonit2
6. R_link2 / R_link3 — R_joint3

没有过滤 base–shank、左右腿之间、非邻接零件或地面，所有 collision 掩码1/1，
只有视觉0/0。**URDF 标准本身不携带 SRDF 排除规则**：Isaac/PhysX 适配器必须
按名字应用这六对并在服务器验证；不能假定导入 URDF 就自动生效。
在原raw材料审查层，两髋对均标记 `proposed_pending_material_review` 且 `geometry_review_supported=false`；
其余四对只保留旧 nominal 局部代理检查结果，不是完整工作域或真实机构批准。
原raw六对表自身不是训练过滤批准；独立研究层的许可来自新明确用户决定，
不是把这些材料审查flag改为true。右髋旧0个
占据见证仅记录在 `limited_sample_no_unexplained_witnesses` 与历史快照中，不再
转译为当前过滤审批。

不能只看 MuJoCo 接触点。`adjacency_review.json` 还求解完整凸半空间交集：
左右髋代理重叠沿关节径向达186.8/188.6mm，膝达69.94/71.21mm，轮约60.02mm
（圆柱诊断采用外包128边形，上界误差<19µm）。**并非全部只落在小轴孔内。**

上一轮在22组相交凸件中，各用固定seed生成40个内部/极值见证，共880点，
其中877点至少一侧为空材料、左髋另3点两侧绕数均≈1；这些原始记录保留不变。
这只是有限采样结果，**右髋旧0个见证并不证明无材料重叠**。旧左三点为：

| 见证到髋轴半径 mm | 到base源三角面距离 mm | 到L_link1源三角面距离 mm |
|---:|---:|---:|
|25.9473|1.5696|.8696|
|25.3890|1.0864|.8696|
|26.6730|1.7628|1.2239|

三个不同方向的独立射线奇偶，对两侧三点均为 `[1,1,1]`（内部）。这些点确在
髋安装区，但不是亚微米表面毛刺，也不能仅凭“相邻”断言有意装配简化。
**这不等价于断言实机一定干涉**：可能是CAD装配/部件归属简化，仍需机械
证据解释；当前代理没有足够授权将其标成纯伪碰撞后放行。具体位置、绕数、
表面距离和射线奇偶直接保存在 manifest details 与完整审查报告。

源开边/非流形边继续披露，有限采样不冒充完整实体布尔证明或工作域认证。
历史73条诊断存档不变，并以组件轮mesh baseline再次复现；新圆柱的未过滤
诊断另记，不覆盖旧证据。

## 新只读左右对照：右髋也不能自动批准

完整依据见 [V40_HIP_COMPARISON.md](V40_HIP_COMPARISON.md) 及
`reports/hip_comparison/reliability.json`、`sections_comparison.json`。本轮仅
读取这些文件，没有修改别代理的报告或对照脚本；摘要保存其实际 SHA256。

独立源截面搜索每侧每姿态14616点，零位与等角 nominal 的四组，各选12点
再以完整三维绕数、三独立射线及最近源面复核，均得到双内部数值见证。
候选数不是体积、概率或全域证明；部分纳米近边界样本不能过度解读。

- 右零位 R0-4：base / R thigh 源面内部深度约 **.569580 / .869587 mm**。
- 右零位 R0-6：约 **1.001241 / .555825 mm**。
- 右 Rm-10 在**原实际 nominal**（qR=−.42250891703703075，不是等角替代值）
  再复查：canonical base坐标 `[-.0144627212179471,-.19341950642903252,
  .018116959671021265] m`，base / R thigh 内深 **.068718869 / .073536275 mm**，
  两侧绕数≈1、射线均 `[1,1,1]`。这些证据独立于旧880点，未伪造加入旧样本。

左右组件000的轴向台阶/大孔朝向相反，不能据此判定应镜像哪侧。也没有直接
证明整件或相同三角片被重复划入base/thigh；组件编号不是CAD body身份。
thigh中5个SW实例的静态合并并不证明真实刚性归属、联动或膝传动形式。
需要带实例/body ID的CAD总装、link映射、mate/活动铰和多姿态证据先作复核，
不能靠删环、切料、改零位、放宽碰撞或重算质量让门通过。

## 静态门范围与 GPU 保守净空

`collision_validation.scope` 仅定义为固定初态 + 六对具名邻接规则的离线检查，
不涵盖工作域、动力学平衡、Isaac cooking或硬件。当前两髋材料/装配与部件
归属解释均未完成，原raw记录中的CAD-first/暂停决定作为历史保留；即便旧
地面与非邻接检查通过，raw综合 `passed` 仍为 false。原始raw重建不能靠有限
无见证样本清除此双髋待审状态。新的等效研究清单仅按上方独立明确审批与
新范围生成，不清除此原材料审查失败，也不声称几何修复。

新增 `base_visual_bounds_m` 为根 canonical frame 的 `[min_xyz,max_xyz]`：

```json
[[-0.28349998593330383,-0.2526317238807678,-0.17299328744411469],[0.28349998593330383,0.2525971829891205,0.07804601639509201]]
```

它覆盖根全部原视觉STL（包括较高部分），不是只取最大连通凸件。GPU净空门
应把此AABB的8个角按实际基座姿态变换到世界后取最低点；不能只用中心高度，
也不能将根AABB当成整机器人安全包络。

## 独立生成与测试

生成器 import-safe：导入不解析参数、写文件、启动进程/模拟器或加载Isaac。
现有numpy/scipy用于构建，MuJoCo仅在显式静态检查时导入；缺失、失败或
`--skip-mujoco` 都关闭静态门。输出必须是新目录，不覆盖任何旧资产：

```bash
python -B tools/prepare_v40_assets.py \
  --source assets/urdf_v40/source.urdf --out /tmp/own-v40-new
# 可选 --skip-mujoco；--evidence PATH 可重复保存历史证据

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=disable \
  python -B -m pytest -q -p no:cacheprovider tests/v40/test_assets.py
```

上轮几何版本使用现有 `.venv_mj314/bin/python` 实测21项通过。本次状态同步
另加两髋待审、独立重建与旧样本/物理散列保留回归。父代理关于 NumPy/BLAS
微小诊断差异的调整原样保留：物理文件仍 byte-equal，frame_validation 的4个
max_error 分别断言 `<1e-12`，不要求跨环境诊断浮点文本完全相同。

前次纯状态同步的历史实测：现有 MuJoCo 环境 **24 passed（10.95 s）**；系统 Python
**23 passed, 1 skipped（10.42 s）**，跳过项是该环境未安装的 MuJoCo运行时分支。
两环境均验证独立重建的49份物理文件 byte-equal、双髋仍pending，以及父代理
保留的4个 `<1e-12` 诊断误差断言。

退出码2表示所选清单范围的门关闭，0仅表示对应范围静态门通过：原raw构建
当前仍为2，明确批准的独立研究变体当前为0；二者绝不互相覆盖失败状态。
测试覆盖原链路全部散列/视觉/惯量/限位/FK/重建，以及测量圆柱、六对显式
过滤、旧73条证据字节保留、880个源材料见证及3点独立交叉检查、地面微小
代理误差、canonical根边界。普通Python缺MuJoCo时跳过运行时分支，不冒充已验。
