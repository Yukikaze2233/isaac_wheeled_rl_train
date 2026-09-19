# 连杆与气弹簧建模工具链

这些工具把原始 CAD/URDF 导出转换成可检查、可复现的闭链研究模型。
当前机构适配为 V5：两级四杆、两套气簧，生成19刚体、16转动轴、2移动轴和6个闭合约束。
几何运算与导出可复用；具体部件名、闭合孔轴和清理规则在 V5 适配器中定义，不假设任意 URDF 都是这种拓扑。

## 入口

| 工具 | 输入与职责 | 主要输出 |
|---|---|---|
| `audit_v5_source.py` | 原始 URDF/CSV，查质量、惯量、树结构与安装轴线 | `source_audit.json` |
| `v5_mechanism.py` | 原始 STL，圆孔截面拟合、组件归属、FK | `metrology.json`；可导入的几何函数 |
| `fit_bkb_gas_spring.py` | 可编辑的目录读图点及目标压力 | 单调力曲线、数值表、拟合图、来源哈希 |
| `build_v5_closedchain.py` | V5原始包、气簧曲线 | 相对路径 URDF/USD/MJCF、约束、惯量/网格来源、manifest |
| `validate_v5_dynamics.py` | 生成的模型包 | MuJoCo真实动力步进及禁闭环对照 |
| `publish_v5_model.py` | 候选包及匹配的MuJoCo/PhysX验证回执 | 不覆盖的正式模型目录、ZIP与SHA256 |
| `inspect_scut_springs.py` | 本地华南虎实际USD | spring转轴/移动副/闭合关系读回 |
| `analyze_v5_spring_limits.py` | 交付模型安装点与气簧行程 | 膝角可达区间、行程余量及完整闭链交叉验证 |
| `audit_v5_installation.py` | 源STL身份、接头与气簧独立固体 | 销孔／网格端面／配合投影尺寸，区别于待确认的图纸基准 |
| `review_chassis_progress.py` | 远端运行回执 | 只读日志快照、TensorBoard CRC、训练窗口与代表姿态 |
| `../scripts/preview_v5_springs.py` | 生成的模型包 | 原生Isaac整机运动、气簧力开关和遥测 |
| `../scripts/compare_v5_spring_load.py` | 同一模型的有／无气簧构型与三种腿姿 | PhysX轮端承重对照、逐步实际电机力矩、稳态窗口和数值质量检查 |

运行时力/势能函数在 `src/wheeled_tasks/chassis/gas_spring.py`。
气簧不是固定杆，也不是直接加在机身上的外力：它有真实移动副，轴向推力经两端安装约束传递。

## 环境

静态生成使用 Python、NumPy、SciPy、trimesh、USD Python bindings（pxr）和MuJoCo。
Isaac预览使用项目已验证的 Sim6 / Lab3 环境。普通 `--help` 不启动仿真。
下面从训练仓库根目录执行，`python` 指向相应环境；本机可用 `/home/yukikaze/isaacsim60-venv/bin/python`。

## 典型流程

### 1. 保留原件、审计并拟合气簧

V5原件存放于 `model/纯底盘_v5/source/`，包含完整URDF/CSV/mesh。
建模产物写到其它目录，生成器会拒绝把输出放进原件目录。

```bash
python tools/audit_v5_source.py
python tools/v5_mechanism.py
python tools/fit_bkb_gas_spring.py \
  --data model/纯底盘_v5/gas_spring/catalogue_points.json
```

拟合输入保留“人工近似读图”及单位/型号假设；当前插值实现支持9–12 MPa之间的目标压力。
安装基准、目录本体长度、额定行程、压缩量分别记录，不能互相代替。

### 2. 构建一个新候选

```bash
python tools/build_v5_closedchain.py \
  --source model/纯底盘_v5/source \
  --spring-data-dir model/纯底盘_v5/gas_spring \
  --output reports/v5_candidate_new
```

输出目录必须不存在。工具会：

1. 依据形状、体积和装配世界位置识别V5导出混入的右上连杆副本，仅选择保留的原STL三角记录。
2. 保留源质量；对无效惯量使用带来源标签的网格均匀密度研究估计，保留原始数值。
3. 建立气簧移动副、四杆与气簧端点闭合；处理导出轴线舍入，逐项记录修订。
4. 求解一致初始状态，导出 URDF、USD、MJCF，验证闭合及约束Jacobian秩。
5. 保存配置、生成源码、源URDF/CSV及资产SHA256。

V5适配规则集中在 `build_spec()`；`v5_mechanism.py` 负责几何/FK，旧导出器作为版本化共享实现复用。
新机构应新增自己的部件映射、孔轴依据和验收样本，避免修改V5规则去兼容不相关拓扑。

### 3. 验证、展示

```bash
# 静态/FK/自由度核对，不改包内容。
python tools/build_v5_closedchain.py --validate-only reports/v5_candidate_new

# 有/无气簧、小力矩、移除约束对照；输出在包外。
python tools/validate_v5_dynamics.py reports/v5_candidate_new \
  --output reports/v5_candidate_new_dynamics.json

# 12秒模拟时间的固定基座PhysX往复验证。
OMNI_KIT_ACCEPT_EULA=YES python scripts/preview_v5_springs.py \
  --bundle reports/v5_candidate_new --headless --seconds 12 \
  --output reports/v5_candidate_new_physx

# 原生整机双侧展示，持续到手动关窗。
OMNI_KIT_ACCEPT_EULA=YES python scripts/preview_v5_springs.py \
  --bundle reports/v5_candidate_new --view whole \
  --output reports/v5_candidate_new_gui
```

展示只有主动轴的目标/力矩随时间变化，从动杆和气簧由物理引擎求解。
详情与华南虎对照见 [`docs/V5_SPRING_INSTALLATION.md`](../docs/V5_SPRING_INSTALLATION.md)。

### 4. 发布、测试

```bash
python tools/publish_v5_model.py \
  --candidate reports/v5_candidate_new \
  --physx-report reports/v5_candidate_new_physx/report.json \
  --mujoco-report reports/v5_candidate_new_dynamics.json \
  --destination model/纯底盘_v5/urdf_new \
  --archive model/纯底盘_v5/v5_closedchain_new.zip

PYTHONPATH=src python -m pytest -q \
  tests/test_v5_model.py tests/test_gas_spring.py tests/test_chassis_task.py
```

发布检查回执是否对应候选资产、包内依赖哈希和ZIP逐文件一致性。
测试中的标准交付目录为 `model/纯底盘_v5/urdf/`；新候选可先用命令行验证，再经审查发布为新版本。

## 版本管理

提交原始必要资产、生成模型、来源记录、脚本和小型验证摘要。
运行日志、视频、checkpoint、重复ZIP和完整临时报告保存在本地忽略目录。
原始CAD质量疑点、研究惯量、行程基准推断及自碰撞状态均写入模型记录，不因测试通过而自动改成实机标定完成。
