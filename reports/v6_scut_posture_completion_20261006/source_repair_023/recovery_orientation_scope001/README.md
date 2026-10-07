# REC v21 quaternion correction: occupied geometry scope

结论：这次四元数顺序错误不支持“当前 REC 实际碰撞地形发生实质改变”的判断，
没有仅因此提前迁移或重启当前 REC 的几何依据。这个结论限于当前 v21 REC 的
真实水平场地，不扩展到 slope/cross_slope，也不是 PhysX 同轨迹或硬件认证。

## 绑定的来源和实际执行

- 冻结源：`/tmp/scut-entropy-frozen-023-av6rrek4`。
- 原提交：`6f07ec298e01d2686216787cfba34b394151f1aa`。脚本实际执行只读
  `git show`，将 recovery worker、campaign、env、task、contact_domain
  五份冻结文件与该提交逐字节核对通过。
- 修正提交：`b0d2800a338fa11099b355ece48c0baaccbb1b02`。修正表达式直接从
  提交内的 `env.py` AST 提取，不读取任意未提交实现。
- REC 原有首 PPO 回执中的 worker 命令确认 `4096` 环境、seed `290927`。
  此次不通过 SSH 查询运行进程；原回执及全部源文件均记录 SHA-256。
- 安装 API 位于
  `/home/yukikaze/Documents/workspace/robot_rl/.deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab/isaaclab/`。
  `sim/utils/transforms.py` 的实际 `standardize_xform_ops` 在 CPU 内存 USD 中执行；
  `spawn_cuboid` 的 Cube size=min(size)、scale=size/min(size) 几何规则按已读源码重建。
  三份安装 API 文件哈希见 `stdout.json`。
- 最终实际运行：`execution.json`，退出码 **0**，**3.2934 秒**；实际 stdout
  是 `stdout.json`，stderr 为空。脚本 `inspect_geometry.py` SHA-256：
  `19d27d66d87062bf6e1170a7a6602875057e196dcebfee03d3132d64a9b2f8ef`。

执行仅在 CPU 上构造 USD 和运行查询，显式隐藏 CUDA。没有启动 Kit、PhysX、GPU、
PPO、optimizer，没有修改源码/合同或操作远端训练。输入文件执行前后哈希保持一致。

## 实际人口和完整实体比较

| 场地 | 组数 | 环境数 | 原生几何分支 |
|---|---:|---:|---|
| flat | 75 | 3116 | 水平三角网格，提前 continue，不调用 Cuboid orientation |
| step_up | 10 | 490 | 每环境 2 个水平实心 Cuboid，共 980 |
| stairs | 10 | 490 | 每环境 4 个水平实心 Cuboid，共 1960 |
| 总计 | 95 | 4096 | 3116 个网格 + 2940 个 Cuboid |

真实 step_up 为 200 mm；双级台面累计高度为 200/450 mm，踏面长度为 .8/.6 m。
`solid_step_platforms=true`，每条场地的实心底部为最低支撑面减 .1 m。
未启用 release-stop 重复地形或 terrain runout。所有实际 Surface 的 slope 和
cross_slope 均为零；合同中的一般 slope_deg 配置没有使本域产生坡面。

脚本按真实 scene_groups 配额、各组 limits/level、原 Surface 构造和接触域随机化
枚举全部人口。每个 Cuboid 比较完整 **8 顶点、6 面**，没有以单个 local +z 面
代表实体。相同 USD 构造参数缓存后实际构造 1541 对 old/corrected USD 对象，
每个具体场地 collider 都消费相应比较结果。95 组逐组几何、人口、材质范围和
比较结果保存在 `stdout.json`。

对于实际水平 Cuboid，原 Surface 四元数为 wxyz `(1,0,0,0)`。按错误 xyzw API
解释时，它表示绕 X 轴转 180°；修正后是单位旋转。两者中心、尺寸完全相同。
以方块中心为原点，旧变换 `(x,y,z) -> (x,-y,-z)` 只置换八个角点和六个外表面。
这对薄地板和任意厚度的水平实心平台都成立，不会改变实体上下边界或台阶高度。

### 保留真实 USD 的非逐位相等结果

初次直接断言 USD 顶点集合逐位相等确实失败。`attempt001_*` 保留原脚本、退出码
1 和 traceback。没有修改或放宽几何容差来掩盖这个负结果。

原因是 USD XformCache 的 180° 旋转矩阵包含 `sin(pi)` 量级残差：所有 2940 个
Cuboid 的 USD 顶点/面浮点集合都不是 bitwise equal。按实际四元数所确定的
顶点置换逐一对应后，最大坐标差 **2.7755575615628914e-16 m**；完整顶点和面
集合的代数相等另行验证通过。`attempt002_*` 保留第一次正确区分代数与浮点的
通过回执；最终版本又增加 flat mesh 分支和三个地形构造函数的 AST 同一性核验。

因此应表述为“代数实体集合相同，USD 读回仅有已记录的浮点舍入差”，不能表述
为“所有原生碰撞数据逐位相等”，更不能推出未来 PhysX 轨迹完全相同。

## 材质、接触对象和地面查询

- 实际 ContactDomain 使用固定 seed 的独立 CPU generator；新旧 draw 和有效
  摩擦系数数组逐位一致。每个 collider 的静/动摩擦、零 restitution、整 prim
  材质绑定相对路径和 collision enabled 均比较一致，材质差异数 0。
- 生产配置仍为各向同性标量摩擦、multiply combine，没有 per-face 材质。
  所以面标签置换不会把另一种材质移到接触面。此为源码和 CPU USD 材质读回；
  没有声称执行了实际 PhysX 接触力比较。
- 6056 个地形碰撞筛选路径和各 lane 的 Surface 数保持一致。完整外表面集合
  相同，不会凭这个旋转产生新的支撑面、侧壁或空隙；动态 contact slots/order
  不在本次验证范围内。
- `ground_height` 在原提交与修正提交中的 AST 完全一致。`terrain_surfaces`、
  `corridor_mesh`、`Surface.box` 执行体也相同（仅 box 文档字符串新增），
  flat mesh 生成分支 AST 相同。Surface 张量的名义域与实体最高面相符。
- 对全部 4096 环境执行跨台阶接缝及边界的 **147456** 次旧/新名义地面查询，
  结果逐位相等。原实体真实最高面与名义 Surface 的最大误差同为
  `2.7755575615628914e-16 m`。

旧 local +z 面标签确实翻到了实体底部，最大“标签与名义支撑面差”可达 **.55 m**
（450 mm 台面与 −100 mm 底部），但实体最高面仍在正确台面高度。不能将 .55 m
标签差当作 REC 台阶碰撞面下降 .55 m 的证据。

据此，无需仅为该四元数 bug 提前重启当前 REC。后续包含坡地或横坡的域仍必须
使用已验证的四元数边界修正；这不撤销先前 v22 横坡的真实几何错误结论。
