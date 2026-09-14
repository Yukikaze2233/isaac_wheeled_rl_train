# V40 Round3：A 已实现，分阶段长期训练计划

更新：2026-09-14。基础分支：`isaac60`；A训练使用冻结版本`6ed83408d1d1cb02ea687e77fa80e5c20932f36c`。用户要求增加轮次后，总预算细化为**100000次更新**；A已完成10000次，B–E属于待实现、待验证的课程。

后续机构核查确认了两级四杆随动表示和质量拆分问题。**修复后从零训练的下一轮以[Round4规格](ROUND4_REPAIRED_TRAINING_SPEC.md)为准**；本文件保留旧A/B1基线及长期课程的历史设计，不表示修复后的动力学资产已可用。

## 1. 本次范围与长期目标

用户长期目标为高速直行、高速旋转、姿态控制、冲击恢复和长期运行。**A只实现前置基础能力：0.29–0.32m多高度站立、停止/起步与训练域内行驶/旋转；不声称已启用高速、推扰、地形或摩擦DR。**

Round2已实测存在零速漂移；0.28m有持续非轮净力，来源未确认。A暂不采样0.28m。机器人仍使用现有双侧2R等效与研究资产；膝35–80°硬界、原资产/传动未确认事项保持。

## 2. 已实现的A合同

文件：[contracts/own_v40_round3_a.json](../contracts/own_v40_round3_a.json)。

```text
contract_id: own-v40-jointspace-h5-v2
round3.stage: A
semantic SHA256:
e31ba1538be4f26d5a5c6452ca8b06e1d09f41db999118957909d13f7248e73e
file SHA256:
91ede15d5f5ae2de7d381e1f3a66f229005f213973f71084aca9a241335676c3
```

ID保留v2观测/动作语义，**semantic SHA不同，不能当同一合同普通resume**。现有v1/v2合同文件未修改。

| 项目 | A实际实现 |
|---|---|
| Actor/critic/action | 125历史观测 / 29 critic / 6动作，原尺度、顺序、history/noise、PD、电机曲线均保留 |
| 时钟 | 200Hz物理、100Hz策略；20s episode、原v2持续失稳终止保持 |
| 正式task | `--stage locomotion`；不要省略后落入CLI默认stand子任务 |
| 速度桶 | 每次速度重采样：30%全零速度，10%纯旋转，60%行驶；桶互斥、比例是全部环境的概率 |
| 行驶范围 | vx∈[-2,2]m/s、wz∈[-2,2]rad/s；纯旋转vx=0，wz仍在[-2,2]均匀采样 |
| 速度时钟 | 保留每3s重采样 |
| 高度 | [0.29,0.32]m；下端点15%、上端点15%、剩余70%连续均匀 |
| 高度时钟 | 独立每环境500–800个policy tick（含两端）重采样，即5–8s；reset只重置指定env时钟 |
| 新奖励 | 替换现有`zero_command_translation`槽位：当abs(vx_cmd)<0.05，rate为`-1*(abs(vx)+abs(vy))`；每步再乘原policy_dt=0.01 |
| 未叠加项 | 无额外平方零速项、固定关节姿态项、height-enhance、alive bonus或新的termination penalty |
| 材质 | 左右轮和地面显式μs=μd=0.5、restitution=0、friction/restitution combine均average；无DR |

### strict验证

新增`round3`仅接受`stage`、`spin_probability`、`height_endpoint_probability_each`、`height_resample_seconds`、`physics_material`五个字段。

- 所有新增数值必须finite，拒绝bool冒充数值；未知字段、未知stage拒绝。
- standing/spin概率各在[0,1]且和≤1；每端点概率在[0,0.5]。
- 高度period为两个正、有序、≤episode长度的数，必须对应整数policy ticks；例如5.001s拒绝。
- A高度必须在0.29–0.32内；A材质严格为本次nominal字典，不能写一个未落地的DR开关。
- 对v2基线做结构比较：除明确允许的采样/高度/L1槽位及`round3`元数据，任何观测、动作、PD、奖励、终止、资产或网络变化都拒绝。
- 验证函数保持stdlib-only；Torch仅在命令采样与proxy计算时导入。

## 3. 命令时序与可维护性

`round3.py::Round3Commands`只维护独立高度计数器及速度桶。现有速度命令计数器保留，未创建另一套env或policy接口。

1. `_get_rewards()`使用本次动作所对应的旧command，且在新tick内只推进两类计数器一次。
2. `_get_observations()`在reward完成后重采样到期的速度/高度，然后向原history追加新观测。
3. 普通高度变化不reset、不清空history；相同tick重复读obs不额外采样、不改变缓存的actor输入。
4. 部分env reset只重采样该子集；其它env的高度时钟、command和history不受影响。
5. 固定evaluation override仍最高优先级，覆盖高度时钟；自动reset继续使用同一固定命令。

训练日志除原`Command/standing_fraction`，增加`Command/spin_bucket_fraction`、`Command/height_low_endpoint_fraction`、`Command/height_high_endpoint_fraction`，可检查实际命令覆盖。桶比例是抽样概率，不保证小batch每次精确达到30/10/60。

## 4. 材质显式绑定与真实读回

A启动时在clone完成后给每env两个命名轮Cylinder collider绑定`/World/v40Round3WheelMaterial`；地面绑定`/World/ground/physicsMaterial`。只新增physics-purpose材质绑定，没有更改碰撞geometry、过滤对、PD或关节属性。**`assets/v40.py`已提交的外部USD seed拒绝继续保留。**

`env.check_round3_materials()`为显式调用的诊断方法：

- 用USD解析真实physics-purpose binding、material系数和combine。
- 用当前SDK的`PhysXUnitTests.get_materials_paths(collider)`核对**物理引擎已解析shape使用的材质路径**；不是仅检查cfg对象。
- 对四个动态轮body创建精确路径的RigidBodyView，读取`get_material_properties()`真实张量，核对顺序、shape和系数。
- 静态地面没有伪造tensor读回：其证据是**PhysX解析后的material身份＋该material的USD系数**。当前tensor接口只返回轮shape的系数，不返回combine；combine依据显式USD及已解析shape绑定。返回值明确写出这些限制。

这比此前仅缓存USD推导名义μ更强，但仍不能把材质系数称作实测牵引极限。无真实ground-pair接触时，不能从净力推出实际附着利用率。

## 5. 滑移可观测性：已落地proxy，未伪造接触pair

`V40EnvCfg.wheel_slip_diagnostics=False`默认关闭；开启时限定最多64env，只为evaluation snapshot计算，**不增加训练actor/critic输入、不参与reward、不在默认大批训练中读回轮world状态**。

开启后每个pre-reset snapshot原有真实q/dq/applied torque外，新增`wheel_diagnostics`：

- 具名L/R轮link世界位置、xyzw姿态、COM世界位置、COM世界线速度与**总角速度**。
- 由原Cylinder轴向偏置、半径0.06m、半宽0.0125m构造最低点候选，包含轮轴轻微倾斜时的端面位置修正。
- `v_point_proxy = v_COM_world + omega_total_world × (p_lowest_proxy - p_COM_world)`。
- 按轮轴切向基底分解`longitudinal_velocity_proxy_m_s`和`lateral_velocity_proxy_m_s`，不采用近零分母的滑移比例。
- `geometry_valid`及`net_contact_candidate_not_ground`；**`ground_contact_confirmed`始终false**，因为当前sensor没有ground-pair/真实接触点证据。
- 语义字符串：`lowest_cylinder_point_proxy_not_ground_confirmed; stationary_flat_ground_assumed`。

这里的轮COM和总角速度来自实际PhysX状态，不拿policy轮目标或仅相对dq冒充。左右编码器符号不进入此公式，避免漏掉root/腿运动。其值是候选接触点运动学诊断，不能当真实滑移率；离地、非地面接触、轮轴近竖直或接触点不在候选最低点时限制尤其重要。

**尚未实现**：官方filtered ground-pair contact points/normal/friction impulses、支持接触gate和经验证真实纵横滑移验收。这些是后续小范围评估工作，不阻塞A基础训练，也没有以空开关声称开启。已有`play_v40_onnx.py`默认CSV不自动加入这个嵌套结构；需要主agent的评估入口设`cfg.wheel_slip_diagnostics=True`并保存snapshot，本次bounded探针已这样做。

## 6. Kaiser有界物理诊断：完成，不是站立验收

有效独立源码快照：

`/home/kaiser/robot-rl-sim60/v40-round3-a-probe-20260913T073729Z-02`

以已提交HEAD094fa33源码归档为底，覆盖本任务六个源码/合同文件，未带入另一agent正在编辑的train/warm-start代码。未覆盖旧`v40-live-snapshot03`或任何旧run。

| 项目 | 结果 |
|---|---|
| 任务 | 2env、seed42、固定命令(0,0,0.30)，40 policy steps / 80 physics steps，0.4仿真秒 |
| 动作来源 | 已核对hash的Round2最终ONNX，仅作为诊断动作源；新A合同与其观测/动作/PD逐项兼容 |
| PPO | **0次更新** |
| 四轮PhysX材质张量 | shape `[4,1,3]`，每轮 `[0.5,0.5,0]`；顺序static/dynamic/restitution |
| 四轮parsed shape材质 | 全部`/World/v40Round3WheelMaterial` |
| 地面parsed shape | `/World/ground/GroundPlane/CollisionPlane` → `/World/ground/physicsMaterial` |
| 两种材质USD combine | friction=average，restitution=average |
| 真实关节limit读回 | shape `[2,6,2]`；四continuous及两有限膝界通过 |
| 轮world twist | `[2,2,6]`；q/dq/applied torque均`[2,6]`，记录值全部finite |
| 纵/横向proxy绝对峰值 | 0.031041 / 0.002904 m/s；**不是ground-confirmed滑移验收** |
| 运行边界 | 专用tmux `v40-r3a-probe-02`，timeout180s、kill-after20s；约14.83s墙钟完成，已自然退出 |

源ONNX SHA：`7d4a93f0d0692f914dc769fb94c88550de62e44a0b3553d95b20d80d406b700e`。
资产manifest SHA：`df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886`。

证据：

- 远端`<snapshot>/probe02/{receipt.json,preflight.json,snapshots.json}`；完整日志`<snapshot>/probe02.log`。
- 本地`reports/round3-a-probe-20260913T073729Z/probe02/`保存同一回执和40帧snapshot，不含checkpoint。
- receipt SHA：`81f6eb19ac8551c36db1c24879e56f620529cbd8df046bba94088c0cff4fd48b`。
- snapshots SHA：`4498911cd8a2164eb165b4e14e39124c06262595888e6b404f091fe9069056ab`。
- 初次独立probe01在0步退出：错误地向`PhysXSceneQuery`调用该SDK实际位于`PhysXUnitTests`的material查询；修正后在新快照probe02完成。失败回执保留，没有删除或改写成成功。

该验证只证明A场景/材质/诊断线路可运行，未验证A训练收敛、长时站立、停止起步质量或冲击恢复。高度5–8s时钟的边界和部分reset行为由CPU真实源码方法回归验证，不能说0.4秒探针已覆盖其完整物理周期。

## 7. CPU验证

独立CPU环境：`/tmp/opencode/v40-sim60-ci/bin/python`。

```bash
/tmp/opencode/v40-sim60-ci/bin/python -m pytest \
  tests/v40/test_round3.py tests/v40/test_core.py \
  tests/v40/test_round2.py tests/v40/test_unified_commands.py \
  tests/v40/test_snapshot_diagnostics.py tests/test_onnx_replay.py -q
```

**137 passed**，2条既有ONNX弃用提示。覆盖新合同/stdlib-only校验、非法字段/概率/clock、PD/reward漂移拒绝、50k样本桶/端点比例、异步height old-reward/new-obs、同tick缓存、部分reset、固定evaluation优先级、纯yaw下L1惩罚、原v1/v2语义、理想无滑移/已知纵横滑移proxy及退化轮轴。

## 8. 长周期预算与晋级

### 8.1 明确的100000更新总计划

表中预算统一按**1024env、48步rollout**制定；它是同一观测/动作语义下逐阶段训练一套策略的预算，不是五个独立策略。候选速度/扰动是课程目标，尚不构成能力或实机参数确认。

| 阶段 | 更新预算 | 累计预算 | 核心任务 | 实现/执行状态 |
|---|---:|---:|---|---|
| A：站立与起停 | 10000 | 10000 | 0.29–0.32m高低站立、减小慢漂、起步/停车、原±2速度域 | 已实现且正式训练中；结束后自动四高度评估和回传 |
| B：摩擦与控制不确定性 | 20000 | 30000 | 有效摩擦、温和质量/PD/力矩变化、动作延迟，保持站立与行驶 | 待实现物理随机化、读回及验证 |
| C：高速直行与快速旋转 | 25000 | 55000 | 逐步扩大直行速度、原地旋转速度，再训练物理可行的组合命令 | 待实现命令课程、组合采样及分速度评估 |
| D：冲击与快速恢复 | 30000 | 85000 | 高低站立、行驶和旋转中的多方向冲击、连续起停恢复 | 待实现有时长/清零的扰动及恢复计时 |
| E：综合巩固 | 15000 | 100000 | 混合前述任务，补最弱高度/速度/摩擦桶，长期与留出测试 | 待前序评估结果确定最终分布 |

当前A的10000次包含在总100000次内；此前Round2的20000次不计入本轮预算。每次更新是一次PPO rollout与优化，不是一个episode，也不是一次梯度minibatch更新。

若A在10000次后仍有明确改善趋势、但尚未达到基础目标，可从总预算外增加5000次同合同续训。若连续三次同条件独立评估均无有效改善或出现退化，先定位命令覆盖、接触、奖励权衡与动作饱和原因；不能只因机器有空闲就机械延长同一分布。后续阶段也采用5000次增量复核，总预算可扩展，但每次扩展记录具体短板和前后评估结果。

### 8.2 B：摩擦与控制随机化

- 根据[华南虎/复旦实际实现对照](ROUND3_FRICTION_UPSTREAM.md)，摩擦首先采用**startup分环境分配64材质桶**，reset不重采；约30%环境使用名义材质，其余使用随机材质。保存桶表和env映射，精确恢复时一并恢复。初期同env两轮共用一组材质，左右差异和空间材质变化后置为独立对照。
- 首先围绕已读回的名义有效μ=0.5，训练有效动摩擦[0.4,0.6]、有效静摩擦[0.5,0.7]，采样后保证动摩擦不高于静摩擦；通过后候选扩到动[0.3,0.8]、静[0.4,0.9]。必须记录两端材质、合成模式和实际shape读回。这些是V40候选范围，不是上游实测范围。
- 保持地面0.5且采用average时，wheel μ∈[0.3,0.7]才得到effective μ∈[0.4,0.6]；扩至effective [0.3,0.8]对应wheel [0.1,1.1]。不能把wheel采样值直接称作有效轮地值。
- 质量和对应惯量、实际手写PD/输出力矩增益先从名义值±5%逐项引入；实际输出仍通过速度相关力矩上界限幅。动作延迟先比较0/5/10ms，再进行episode级0–10ms采样；延迟缓冲必须明确物理步单位、reset填充和清空语义。
- 摩擦名义环境在不同回合继续保留，其余控制随机化也至少保留30%名义样本。先逐项对照，再混合；启动配置含DR字段不等于DR已经作用到物理状态或`compute_torques`。
- 晋级前补齐真实轮地接触身份、支持接触条件及纵横向滑移统计。当前最低圆柱点proxy可辅助排错，但不能替代接触点测量。慢漂与打滑分别验收。

### 8.3 C：高速命令采用分轴课程

- **直行**：在原2m/s范围基础上，候选按2.5→3→4m/s扩大，正反向均覆盖；高速下先小yaw。
- **原地旋转**：vx=0，yaw候选按2→4→6→8→12.57rad/s扩大，末档约120rpm，正反转均覆盖。每档只有在姿态、跟踪、载荷与滑移评估通过后才增加。
- **组合命令**：不能独立均匀采样所有速度上限的笛卡尔积。使用实际轮中心轮距、当前高度与电机曲线检查左右轮速/剩余力矩，并依据轮地附着给组合命令留余量。
- 当前研究资产nominal轮距约0.4373m、轮半径0.06m，近似轮地速度为`v_left/right = vx ∓ wz*b/2`。5m/s叠加120rpm时外轮约7.75m/s，已经超过当前研究电机曲线约5.45m/s的零力矩边界；**5m/s与120rpm不能同时当成必须达到的训练指令**。5m/s纯直行也仅剩很小力矩余量，先作为后续单独可行性评估，不纳入C的承诺指标。
- 保留至少40%回合在A/B已学范围，且全零速度桶至少20%。高速动态平衡允许必要的姿态变化，不用固定关节角或全速度恒定姿态目标把动作锁死。

上述速度是研究课程候选，不是修改物理关节硬界或电机上限。更大训练预算不能越过真实执行器和接触约束。

### 8.4 D：冲击恢复

- 第一种训练扰动采用明确的水平速度增量，episode稳定至少2秒后，间隔5–10秒，幅值先0.1→0.25→0.5m/s；方向覆盖前后、左右及斜向，正反运动都要抽到。
- **另一种施扰方式作为留出测试**：有限时长的外力脉冲，持续0.1–0.2秒，施于明确位置并在结束时清零。按当前回合实际质量计算`J=m*delta_v`，不把event采样间隔当脉冲时长。
- 名义12.752kg时，delta_v=0.25m/s对应冲量3.188N·s，持续0.1秒对应31.88N；只是实验换算，真实运行应使用随机化后的质量。
- 至少50%回合无额外冲击，混合低/高站立、直行、旋转以及高度转换。先测一次扰动恢复，再引入间隔足够长的多次扰动，不同时加最大速度、最低摩擦和最大冲击。
- 初始研究目标：0.25m/s量级扰动后约2秒回到预定义误差带，并连续保持1秒。零命令场景候选误差带为平移速率≤0.05m/s、高度误差≤0.01m、roll/pitch绝对值≤10°；移动场景另按目标速度分档设置跟踪误差带。带宽在首次评估前冻结，同时记录恢复率、恢复时间P95、过冲、饱和、滑移和失败，不能只报平均值。

### 8.5 E：巩固与最终验收

E维持同一策略，候选回合比例为40%名义基础任务、30%摩擦/延迟/高速任务、30%受扰恢复；每个大桶都覆盖目标高度，并额外增加此前最弱场景的采样。不是把每个回合都设置成最难参数组合。

- 每阶段每1000次更新进行一次固定评估，每5000次更新做较完整的分桶评估；评估seed、指令序列和参数不参与训练课程选择。最终另保留一组未用于日常晋级的测试。
- checkpoint保留周期为100更新；同时维护按固定评估挑选的best，不能默认last最好。快评开始前复制并核验稳定checkpoint，避免读到写入中的文件。
- 高度至少覆盖0.29/0.30/0.31/0.32m及高度转换；静止目标仍为高度MAE≤5mm、P95≤10mm、平均平移速率≤0.02m/s，同时报告XY漂移和姿态。0.28m须先定位持续非轮力，几何/碰撞证据通过后再恢复采样。
- 现已部署的A结束评估只包含四高度、每高度累计60秒且含20秒reset。后续需补**无timeout重置的连续60秒站立**及更长混合场景；延长episode仅作用于显式评估配置，失败仍记录，不能把reset当恢复。
- C/D/E增加分速度、分高度、分摩擦的跟踪、滑移、恢复率/耗时与执行器饱和统计。回传SHA成功、episode超时、净力阈值或reward上升均不能单独作为效果通过。
- 地形另立后续任务，不为了凑轮次同时引入尚未验证的地面高度reference。

### 8.6 样本、时间和阶段衔接

`transitions = num_envs * 48 * updates`。在1024env下：

| 范围 | transitions |
|---|---:|
| A 10000次 | 491520000，约4.92亿 |
| 全部100000次 | 4915200000，约49.15亿 |

按当前约1.7–1.9s/update，全预算的同负载纯训练参考约47–53小时；DR、冲击、评估和并行任务会改变速度，工程排期先按**3–5天**，不是完成或收敛承诺。每阶段依据实测吞吐重新估时。

2048env×5000更新与1024env×10000更新的transitions相同，但优化次数/batch大小不同，**不构成算法等价**。扩大环境数后必须显式重定该阶段的更新数和样本预算。不能把2048env×10000更新称为相同工作量的加速，也不能用显存余量替代RAM/建场与吞吐测量。

当前A的10000更新和已启动的结束评估/回收继续构成第一段。总100000次是阶段计划，**尚未被追加为运行中的长任务，也没有自动执行B–E**。同合同继续优化可恢复optimizer/iteration；命令、奖励或随机化分布变化时必须建立新的阶段合同与迁移记录。现有`--warm-start`只支持固定Round2父模型→A，不能直接拿它把A→B迁移伪装成已支持。

后续需实现并验证阶段间权重/优化器策略、单GPU串行训练与评估调度、阶段级恢复、best选择及后续课程。将长周期拆成可恢复训练段，段结束按已冻结指标决定继续、晋级或定位问题；不是一次启动100000次后无人检查。当前A的51小时watcher只覆盖A，B–E需各自建立完成评估与回传任务。

上游依据沿用[第二轮及开源审计](ROUND2_FINAL_AUDIT.md)与[摩擦/高速预算](V40_FRICTION_AUDIT.md)：借鉴华南虎实际启用的站立、DR、延迟和阶段迁移；复旦plane不能被描述为push/延迟全开。本表的100000次分配、速度/扰动档位是本机器人研究设计，不是两家上游已经验证的参数。

## 9. 当前部署与下一步

A已完成跨RSL warm-start集成、真实PPO/ONNX短测、1024env容量验证及正式启动。`round3.py`和迁移模块已纳入源码provenance。正式任务使用seed43、10000更新、48小时软预算，运行记录和原命令见[部署交接](ROUND3_A_RUNNING.md)及[runbook](ROUND3_KAISER_RUNBOOK.md)。

代码根目录固定为`/home/kaiser/robot-rl-sim60/isaac_wheeled_rl_train`。本文的长期预算更新不替换运行中的冻结源码，不回写其合同或原始回执。

少env专项诊断可在AppLauncher后创建`V40EnvCfg`，设置`contract_path`、`stage='locomotion'`、`allow_research=True`、`wheel_slip_diagnostics=True`；`env.check_round3_materials()`读回材质，`env.get_evaluation_snapshot()`保存pre-reset全量诊断。它与已部署的四高度CSV评估不是相同的输出接口。

下一阶段优先实现B的材质/控制DR及真实滑移评估，再接C/D的速度和冲击课程。每阶段实现后必须验证参数确实生效、普通样本没有退化、迁移身份正确，才部署对应长训练。B–E目前未执行，不能将长期规划表当作已启用配置。
