# V40 Round3：A 已实现，分阶段长期训练计划

日期：2026-09-13。基础分支：`isaac60`，已提交基线`094fa33139d9f3f0dafa4723f9b7b2c53556688a`；本文描述其上的A增量。

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

| 阶段 | 预算与目标 | 状态 |
|---|---|---|
| A | **10k updates**；0.29–0.32高低站立、零速漂移、停止起步、±2域内行驶/旋转 | 本次已实现，主agent完成warm-start/PPO短测后启动 |
| B | 独立10k–30k预算；effective μ初始0.4–0.6，按真实材质读回与滑移指标扩展；适度模型/延迟DR与训练保留集 | 计划，**未启用DR** |
| C及后续 | 独立20k–60k预算；高速直行/旋转组合、姿态、冲击恢复、必要地形/长时运行；留出扰动和速度边界测试 | 计划，**未启用push/高速扩域/地形** |

总研究预算 **40k–100k updates，可按独立评估扩展**。更新数不是时长保证，也不是自动升级条件；环境数量改变时同时记录transitions。可用持续训练进程/分段checkpoint保持长周期工作，但更换阶段分布/合同必须有可追踪的迁移记录，不因reward或timeouts看起来好就自动进入B/C。

独立评估至少包含：高低位fixed zero command、停止/起步、纯旋转与组合命令；高度MAE/P95、平移速率与逐回合漂移、真实diagnostic、限位与饱和。0.28m须先定位持续非轮力再决定恢复采样。B/C必须加入有支持接触gate的滑移、推扰恢复时间、峰值姿态与未见分布测试。高速预算参考[V40_FRICTION_AUDIT.md](V40_FRICTION_AUDIT.md)，不能同时照抄5m/s与120rpm上限。

## 9. 主agent集成入口与待办

训练入口/跨RSL warm-start由另一agent负责，本任务未修改`train_v40.py`、`v40_job.py`或export。

1. 将本任务源码与新合同合入主agent最终冻结快照；**provenance source list需包含新增`src/wheeled_tasks/v40/round3.py`**，避免只哈希core而遗漏实际采样代码。
2. 完成`--warm-start`：从RSL3的Round2最终模型迁移actor/critic到RSL5新合同，不加载旧optimizer或伪称同合同resume；本次物理探针没有替代这项PPO集成验证。
3. 小规模真实PPO短测成功后，正式A采用：

```bash
source /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh
python -u scripts/train_v40.py \
  --research --headless --stage locomotion \
  --contract contracts/own_v40_round3_a.json \
  --warm-start /path/to/verified-round2-final/model_final.pt \
  --num-envs 1024 --seed 42 --max-iterations 10000 \
  --max-runtime-seconds 43200 \
  --run-dir /new/unique/round3-a-run
```

该命令须在**包含完成后的warm-start代码**的新冻结快照根目录执行，由主agent放入独立tmux并设置外层timeout/产物回传；路径按部署实际替换。是否加live-view、环境数和墙钟预算由短测吞吐决定，本任务未启动此命令。

4. 少env评估可在AppLauncher后创建`V40EnvCfg`，设置`contract_path`、`stage='locomotion'`、`allow_research=True`、`wheel_slip_diagnostics=True`；`env.check_round3_materials()`读回材质，`env.get_evaluation_snapshot()`保存pre-reset全量诊断。

本任务不commit/push。B/C尚未实施；不以此次A基础实现或0.4秒物理探针声称长期目标已经达成。
