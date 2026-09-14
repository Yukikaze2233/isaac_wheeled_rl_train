# Round3-B1：已实现的启动时轮摩擦随机化

日期：2026-09-14。基线commit：`c007bc2f2e6d1ab7b7b0ef67210500169f1b5d24`；B1实现尚未由本任务提交。

## 范围与当前状态

**B1仅增加摩擦DR。** 完整继承A的命令分布、独立高度时钟、L1零速惩罚、125/29观测、6动作、PD/noise、机械界和终止规则；没有加入PD/delay/push/high-speed/terrain。

- CPU相关回归88项通过。
- Kaiser独立实验目录完成 **8env、40policy steps/80physics steps、0.4仿真秒**材质探针，0次PPO更新；检查部分reset后映射及读回不变。
- 本任务未启动256env/500update pilot；该项由主agent集成stage-transfer并测试、提交后部署。
- 未修改、暂停、停止或覆盖A冻结代码目录`/home/kaiser/robot-rl-sim60/isaac_wheeled_rl_train`。
- 执行前只读检查发现A已自然完成：10000/10000 updates、export verified，完成时间UTC `2026-09-13T23:57:20.888252+00:00`（北京时间9月14日07:57:20）。PID446292已不存在。此前“04:11仍运行”的状态已过时；本任务没有等待或促成其退出。

上游依据见[ROUND3_FRICTION_UPSTREAM.md](ROUND3_FRICTION_UPSTREAM.md)：借鉴SCUT/FD的startup64桶，并采用本任务明确指定的窄范围、每env左右一致方案。

## 1. 新合同与strict验证

[contracts/own_v40_round3_b1.json](../contracts/own_v40_round3_b1.json)：

```text
contract_id: own-v40-jointspace-h5-v2
round3.stage: B1
semantic SHA256:
de3da1d5c693e1b42998997fa1a8aa51957542ddd2af99f73f912b63397fda30
file SHA256:
477404d6781a9c256017317aa3d71c470039ed74cdea6cb01b0f4ec1cbfbbedf
```

新增且只允许以下`round3.material_randomization`：

```json
{
  "bucket_count": 64,
  "generator_seed": 1044,
  "nominal_probability": 0.3,
  "effective_static_range": [0.5, 0.7],
  "effective_dynamic_range": [0.4, 0.6],
  "restitution": 0.0,
  "assignment": "per_env_shared_wheels",
  "resample": "startup_only"
}
```

`generator_seed`允许整数`[0,2**63)`，不接受bool；其余分布字段固定为上述已审查B1定义。拒绝NaN/Inf、未知字段、其它桶数/范围/重采样方式、非零restitution及PD等夹带开关。

`validate_round3()`对B1做独立dispatch；B1校验全部已知配置后，将其余合同与A文件逐项严格比较。A仍走原校验逻辑，A semantic SHA保持`e31ba153…7248e73e`。原v1/v2/A合同文件未修改。`Round3Commands`未改，A/B1共用；core只将原A的L1分支扩展到B1。

## 2. 实际采样和USD/PhysX应用

`round3.py::B1MaterialBuckets`使用独立CPU `random.Random(generator_seed)`（MT19937），不消耗Python全局random或Torch command/noise RNG。

1. 采样64个effective三元组：静摩擦均匀[0.5,0.7]；动摩擦均匀[0.4,0.6]后裁到≤静摩擦；回弹0。
2. 先在effective空间处理静/动一致性，再按地面0.5、average转换：`wheel_mu=2*effective_mu-0.5`。因此wheel静摩擦[0.5,0.9]、动摩擦原范围[0.3,0.7]，裁剪后不再独立均匀。
3. 每env分配一个桶ID，独立nominal mask约30%；nominal env明确使用wheel `[0.5,0.5,0]`。不强制小batch恰好30%，也不要求随机桶碰巧等于nominal。
4. 每env两轮共用相同桶或nominal材质。

**没有逐env反复改同一个共享USD Material值。** 启动时创建64个各自独立且之后不变的bucket physics materials，另保留nominal material；按env映射给两个命名wheel Cylinder collider绑定对应路径：

- nominal：`/World/v40Round3WheelMaterial`；
- random：`/World/v40B1Materials/bucket_00`…`bucket_63`；
- ground：仍为`/World/ground/physicsMaterial`，0.5/0.5/0、average。

不同env可共享同一个已固定bucket，但不会产生“后一个env写值覆盖所有env”的问题。只改wheel collider绑定，未绑定/随机化base或腿的shape。episode reset路径完全不调用该采样器或重新绑定；外部USD seed拒绝仍保留。

## 3. 与训练agent约定的report及精确恢复

B1 `V40Env`构造完成时自动进行实际材质核验，成功后提供 **`self.round3_material_report`**，全部为JSON primitive。训练agent可直接deepcopy并写入run manifest；本任务不写训练run目录、不改job或算法模块。

主要字段：

| 字段 | 内容 |
|---|---|
| `stage`, `passed`, `sampling` | B1、读回通过、startup-only |
| `expected_mapping` | schema、num_envs、generator seed/算法、完整配置、64×3 effective桶、64×3 wheel桶、env桶ID、nominal mask |
| `mapping_sha256` | 上述mapping canonical JSON摘要 |
| `bindings` | 各wheel collider及地面实际USD binding、PhysX已解析material路径、USD系数/combine |
| `wheel_body_paths`, `wheel_env_ids` | 实际读取顺序，支持逐env左右核对 |
| `wheel_physx_coefficients` | 实际`get_material_properties()`结果，`[2*N,1,3]` |
| `nonwheel_first_shape_physx_coefficients` | 每个非轮body第一个真实shape的读回；不包含max_shapes padding |
| `ground_scope`, `combine_scope` | 准确说明地面/合成模式的证据边界 |

精确恢复接口：构造环境前设`cfg.round3_material_restore = <完整已保存round3_material_report>`。恢复时：

- 不调用随机generator、不重新生成桶；直接使用保存的表与映射。
- 校验mapping SHA、num_envs、generator seed/config、64×3形状、finite数值、静/动范围与一致性、effective→wheel转换、ID和bool mask。
- 再绑定并重新读取当前PhysX实际值确认。不同num_envs拒绝直接恢复，不能只截取/扩展旧映射。
- 报告深拷贝，不与调用者可变字典共享。

仅传相同seed不作为恢复证据；需要保留并核对实际mapping及读回。digest用于发现意外篡改，父run的整体artifact/manifest身份仍由训练agent验证。映射相同也不意味着跨软件版本的整个物理轨迹逐bit相同。

A的`check_round3_materials()`原返回保持；B1返回上述实际report。地面仍明确为**PhysX解析后的材质身份＋USD系数**，没有伪造静态地面的tensor系数或tensor combine-mode。

## 4. Kaiser真实探针

独立代码根（叶名严格为`isaac_wheeled_rl_train`）：

```text
/home/kaiser/robot-rl-sim60/experiments/b1-probe-20260914T063634Z/isaac_wheeled_rl_train
```

基于已提交c007bc2归档，再覆盖本任务源码及新B1合同；没有将其它agent正在修改的stage-transfer代码半成品用于探针，也没有覆盖A冻结目录。

- Runtime：`/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh`。
- 专用tmux：`b1-material-probe-063634`；timeout180s、kill-after20s，已自然退出。
- 8env，environment seed44；material generator seed1044。
- 使用已完成A最终ONNX作为诊断动作源，未训练B1 policy。A ONNX SHA：`3d8c682cfc94ac0ee1879bdc6fd90c208e123c05e6b48de3f970f42a474c1ad9`。
- 40策略步/80物理步、0.4仿真秒，约14.20秒墙钟；20步后对env0（nominal）和env1（random）执行部分reset。
- 全部保存snapshot数值finite，reset前后expected mapping和实际material读回相同，cached `round3_material_report`未变。

实际nominal mask：`[true,false,false,true,false,false,false,false]`（2/8=25%，没有强制凑30%）。桶ID为`[56,57,7,27,35,63,20,11]`；nominal mask优先于桶ID。

16个真实轮shape读回`[16,1,3]`，每env两轮完全一致；8个env有7组不同材质：

| env | 类型 | 左右轮共同的实际wheel静/动/回弹 | 按ground0.5/average推导effective静/动 |
|---:|---|---|---|
| 0 | nominal | 0.500000 / 0.500000 / 0 | 0.500000 / 0.500000 |
| 1 | bucket57 | 0.842341 / 0.476738 / 0 | 0.671171 / 0.488369 |
| 2 | bucket7 | 0.520525 / 0.374169 / 0 | 0.510262 / 0.437084 |
| 3 | nominal | 0.500000 / 0.500000 / 0 | 0.500000 / 0.500000 |
| 4 | bucket35 | 0.832960 / 0.538166 / 0 | 0.666480 / 0.519083 |
| 5 | bucket63 | 0.557982 / 0.557982 / 0 | 0.528991 / 0.528991 |
| 6 | bucket20 | 0.605323 / 0.376102 / 0 | 0.552662 / 0.438051 |
| 7 | bucket11 | 0.871326 / 0.618937 / 0 | 0.685663 / 0.559469 |

所有dynamic≤static。40个非轮body的首个shape均读回`[0.5,0.5,0]`；代码未改任何非轮绑定。这不是“遍历了所有非轮shape材质”的夸大声明。

本次mapping SHA：`36cc282d8d30e29f0eea77329cc9f1d500797a05ebf53baa19be2681b7285338`。
完整receipt SHA：`76c39fafb29cce9168564afd8526d769fab5da8e7fc54e21445eb325c8f1bfa8`。

证据：远端上述实验父目录`run01/{receipt.json,preflight.json,snapshots.json}`，本地副本在`reports/b1-probe-20260914T063634Z/run01/`。Kit日志及USD cache留在独立实验目录，未混入A run。

**分类：engineering_material_probe_not_policy_quality。** 该探针证明DR已落地、分环境/左右对应正确以及reset不重采，不证明摩擦鲁棒性、长时站立、真实轮地滑移或B1训练效果。

## 5. CPU测试

```bash
/home/yukikaze/isaacsim60-venv/bin/python -m pytest \
  tests/v40/test_round3_b1.py tests/v40/test_round3.py \
  tests/v40/test_round2.py tests/v40/test_core.py -q
```

结果：**88 passed**，2条既有ONNX export弃用提示。覆盖固定seed复现、独立RNG不扰动全局、nominal比例/转换、restore不调用RNG、digest篡改、重签后非法表/ID/mask/env-count/seed拒绝、部分reset不重采、B1与A命令流/L1奖励完全一致以及legacy合同兼容。

## 6. 交给主agent的pilot准备

本任务没有修改`train_v40.py`、`v40_job.py`或算法模块，也没有提交推送。新增B1逻辑全部在现有`round3.py`、`core.py`、`env.py/env_cfg.py`及合同/测试中，**没有新的helper文件名需要加入provenance清单**；现有源文件新SHA仍应按最终提交归档。

主agent完成stage-transfer集成测试和commit后，可在独立实验部署根下启动256env、500update pilot，继续保持代码根叶名`isaac_wheeled_rl_train`，与任何正式A目录隔离：

```bash
source /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh
python -u scripts/train_v40.py \
  --research --headless --stage locomotion \
  --contract contracts/own_v40_round3_b1.json \
  --stage-transfer /home/kaiser/robot-rl-sim60/round3-runs/train-formal01/train/model_final.pt \
  --source-checkpoint-sha256 6d33498d2424bb45305c199d42261d62c39c13a39107a604a89d28ccd19f4af0 \
  --num-envs 256 --seed 44 --max-iterations 500 \
  --run-dir /new/independent/b1-pilot-run
```

源A checkpoint SHA已在远端重新计算并与其completion一致。该命令需在主agent最终新实验快照根目录、专用有界tmux中执行；本任务未执行它。训练agent负责把`env.round3_material_report`写入manifest，并在resume时恢复/核对完整映射。

后续评价应区分nominal/random cohort及effective摩擦档位，复用高低站立、停止/起步、直行/旋转矩阵；不同摩擦不等于不同地形，净力/最低点proxy不等于真实ground-pair滑移。B1不夹带其它DR或高速/冲击目标。
