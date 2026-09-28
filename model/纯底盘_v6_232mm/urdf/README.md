# V6.0 独立闭链研究资产

本目录从 2026-09-28 审查的新下载源重新生成，未修改 V5 或 V5_232mm。
可移植入口是 `urdf/robot.xml`（MuJoCo）、`urdf/robot.usda`（USD/Isaac）。
`robot.urdf` 仅表示树，单独导入不会获得六个闭环约束。

源文件内部仍叫 `urdf_v5.0.urdf`；以版本身份和 SHA 区分，不能依文件名判断。
其 SHA-256 为 `2c061fc49e1cf375b32c19002495aa03c2856941b326688f36cd81d7102e095d`。
`source/` 是只读下载快照，`source_provenance.json` 保存每个输入和冻结 V5 工具的来源哈希。
生成器拒绝覆盖已有输出，并在临时目录完成导出、MuJoCo 编译和约束秩验证后才发布。

## 坐标和闭链

- 根坐标使用新源自带的 X 前、Y 左、Z 上，**没有额外 root Rz(+90°)**。
- 旧测孔点通过局部 Rz(+90°) 转入新 link frame。新源关节轴按各侧共同铰轴重新做舍入校正，改变量逐轴记录。
- 19 个刚体、16R+2P 树、6 个树外球闭合约束；18 个点约束方程秩为 12，内部独立自由度为 6。
- 6 个主动轴顺序为 `[L_joint1, LL_joint1, L_joint3, R_joint1, RR_joint1, R_joint3]`。
  相对 V5_232mm，四个腿主动轴的 q、dq、力矩同时反号，两个轮轴不变。
  完整被动轴/滑块映射见 `coordinate_mapping.json`；它不包含实车编码器零点标定。
- nominal 从新闭链源零位连续求解，六个主动值约为
  `[-0.42, 0.137422825954, 0, 0.42, -0.137415576258, 0]`。
  base 高度 0.32 m 是初始化配置，不是断能平衡高度。
- 膝角定义为髋销—膝销、膝销—轮轴的两方向投影到共同铰轴法平面后的内角。
  沿铰轴错层的销中心直接计算 3D 夹角会混入横向偏移。
  平面零位内角是 44.936658953811°，原 40–110° 边界由 FK 复核。
  RMCS 辨识配置的 105° 记录仍需单独核对，未擅自替换用户冻结的机械上限。
- 源髋/膝输入销轴有约 0.55–0.56 μm 横向舍入错位，原点没有被静默搬移。
  同侧两电机等增量测试保留并报告由此产生的亚微弧度误差。

## 质量、惯量与网格

总质量为 **13.886 kg**。新源的 RR_link2 为 **0.084 kg**，与左侧同形件一致，
替代旧源 0.84 kg；这是 CAD、对称性与网格体积支持的修订，尚不是称重结果。

新源四个张量违反主惯量三角不等式，另六个使用人工 `1e-8 kg·m²` 极小主惯量。
这十件统一采用其封闭、自洽定向的 owned mesh、源质量做均匀密度体积积分，
生成 COM 和合法张量。其余九件保留 CAD 值。逐件源值、生成值、特征值、原因及
网格体积见 `inertial_sources.json`；不使用 MuJoCo `balanceinertia` 隐藏不合法输入。
所有修订仍是未辨识的研究先验。

视觉网格保留新下载三角面；所有权审查结果见 `mesh_ownership.json`。
碰撞使用分连通件凸包/确定性 OBB，轮使用半径 0.06 m 的圆柱。自碰撞关闭。
L_link2 相比冻结 V5 多 12 个面、无新顶点的来源差异不被悄悄抹掉。
凸包可能填洞，本轮有限检查不等于接触、恢复或全工作域碰撞验收。

## 气簧与执行器接口

采用已确认 **AB 全伸 232 mm、行程 80 mm、全压 152 mm**；
205 mm 簧体长度加 27 mm 有效接头偏置延续已确认安装定义。
下载 USD 的 109–172 mm / 63 mm 方案没有被使用。

滑块 `L_spring_slide/R_spring_slide` 是树内可观测、可施力 DOF，正方向增加 AB：

```text
AB_CAD_zero = 0.16227281990887452 m
q ∈ [-0.01027281990887452, +0.06972718009112547] m
compression = 0.06972718009112547 - q
F = 280 + 122.73591849113129 u² + 53.96980440930969 u³ N
u = compression / 0.080
```

nominal 两侧 AB 约 179.909639 mm、压缩约 52.090361 mm。
力曲线是既有 10 MPa 目录拟合先验；0–72 mm 为原拟合域，72–80 mm 为显式研究外推。
75 个几何样本的压缩范围约 15.517–72.795 mm，40° 端点略进入外推区。
压力、温度、老化、密封摩擦、滞回和阻尼尚未由这次资产导出确认。

MuJoCo 的六个 motor actuator 施加主动广义力矩，另外两个 `_gas_force` actuator
仅施加内部气簧力。载入 XML 本身不会自动施加力曲线；使用者必须每步计算并施力。
USD 不含隐式驱动；Isaac runtime 同样每步给滑块施力，不重复添加另一份弹簧补偿。

PC 力矩模式主动轴默认阻尼为 0，MuJoCo 显式覆盖旧 exporter 的全局 `.002`；
被动铰轴 `.002 N·m·s/rad`、滑块 `.002 N·s/m` 是沿用的未辨识先验，PhysX runtime
须保持一致。辨识新的被动项时先扣除/替换这些先验，避免重复计算。
导出关节的 effort/velocity 为解析器占位信息，不是电机能力或硬件允许边界。
轮减速比 **15.8** 已记录；策略接口仍是轮端 rad/s，不再乘一次传动比。

## 验证和边界

`validation.json`：MuJoCo 编译、拓扑/秩、零位与 nominal 闭合。
`cpu_validation.json`：75 姿态、40–110° 膝域、frame/FK 映射、惯量、USD 单位/关节/
质量、弹簧虚功与等增量耦合。所扫髋角 `[-0.8, -0.42, 0, 0.42, 0.8] rad`
只用于几何验证，不声明为硬件安全 workspace。
`dynamics_validation.json`：有/无气簧及小力矩的有限 MuJoCo 动态检查，以及关闭约束后
销点分离的负对照。检查是否有限和闭合，不用它证明被动站稳。
`physx_validation.json`：1 环境、固定底盘、1 ms 步长、3.02 s、2° 小摆；
运行时不覆盖被动坐标，也不调用 IK。fixture 的增益/限幅属于验证控制器，不是辨识结果。

资产哈希和回执由 `manifest.json` 固定。`build_manifest.json`（如存在）保留验证前的
原构建身份；验证回执中的输入 manifest SHA 指向这个冻结版本，最终 manifest 包含回执。
即使这些有限检查通过，`training_ready` / `hardware_deployment_ready` 仍为 false：
接触/自碰撞、实车参考端/零位、执行器与气簧辨识、真实台阶/落地边界仍须单独验收。

## 重建

Kaiser CPU 环境已有 numpy、scipy、trimesh、MuJoCo、USD；导出不需要启动 Kit/GPU。
从仓库运行（输出目录必须不存在）：

```bash
source /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh
python model/纯底盘_v6_232mm/tools/build_v6_closedchain.py --output /tmp/v6-rebuild
python model/纯底盘_v6_232mm/tools/validate_v6_asset.py /tmp/v6-rebuild --output /tmp/v6-cpu.json
python model/纯底盘_v6_232mm/tools/validate_v6_dynamics.py /tmp/v6-rebuild --output /tmp/v6-dynamics.json --steps 2000
```

PhysX 检查需单独空闲的 GPU 和 IsaacLab runtime：

```bash
python model/纯底盘_v6_232mm/tools/preview_v6_springs.py --bundle /tmp/v6-rebuild --output /tmp/v6-physx --headless --device cuda:0 --seconds 3 --swing-deg 2
```
