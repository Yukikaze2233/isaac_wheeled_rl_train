# Round3-A 最终评估失败诊断、恢复与R2对照

日期：2026-09-14。**原四高度失败是场景资产读取失败，未执行策略；修复评估资源入口后，四项均完整完成6000步并已回收。**

## 1. 当前有效结论

- A最终策略在0.29/0.30/0.31/0.32m各完成累计60仿真秒：failure=0、timeout reset=3，全部数值有限，未检测到膝硬界越界、瞬时倾角越35°、低高度或持续失稳。
- **按原研究criteria，仅0.32m全部通过**。0.29m均速略超0.02m/s，0.30m高度/均速未达标且有持续非轮净力，0.31m高度MAE超5mm。
- 相较R2对应高度，四档零速平移速率均下降；但0.29/0.30m高度误差变大，0.30m还出现新的持续非轮净力候选。**不能把“训练10000更新、export verified、零failure”合并成“所有高度站立验收通过”。**
- 非轮净力没有碰撞对手身份，不能冒称地面支撑或打滑。0.30m异常需要后续定位；本次没有修改reward、termination、reset、摩擦或诊断来让它通过。

精简证据：[round3_a_evaluation_recovery_20260914.json](evidence/round3_a_evaluation_recovery_20260914.json)。

## 2. 原自动评估为何四项exit1

原目录：

`/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/audit/after_training/final-evaluation`

读取四份完整log及summary，四项共同情况：

- `policy_steps=0`、`diagnostic_frame_count=0`；四项均没有`telemetry.csv`，不是存在未分析的策略失败轨迹。
- 模型、合同、sidecar与preflight已通过；错误发生在`make_env()`内部、策略循环之前。
- 调用链：`V40Env._setup_scene → spawn_ground_plane → create_prim → add_usd_reference`。
- 每项等待约102–112秒后抛出同一异常：

```text
FileNotFoundError: Unable to open the usd file at path:
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd
```

因此原报告中的“replay failed or policy/contract identity differs”太笼统，**实际不是已证明的policy mismatch、nonfinite或跌倒**。0步summary里初始化为true的`all_numeric_finite`、全零诊断也不构成物理运行证据。

恢复时对同一官方URL读取返回HTTP200、17624字节。日志不足以进一步区分当时的HTTP、网络或OmniClient resolver问题，未把FileNotFoundError武断解释为HTTP404或永久资产缺失。h031另有plugin错误日志，但四项最终异常均止于同一地面URL读取。

旧log/summary及旧failed结果已另行回收到本次恢复目录的`old-failed/`，原远端文件、receipt和原本机`evaluation-return`报告没有覆盖。

## 3. 最小工程修复及源代码隔离

### 只替换资源位置，不编辑地面物理内容

`scripts/play_v40_onnx.py`新增可选：

```text
--ground-usd /path/to/local/default_environment.usd
```

它只接受SHA固定的官方Sim6 Grid USD：

```text
78e9a1e72a8838a13d0f65c49cd487ab92e89233cd128b057730b5b5b4ca2164
```

- preflight前和环境构造时各核对一次文件hash。
- 在该次replay环境构造的上下文内，将ground cfg深拷贝后只替换`usd_path`；原material、size、位置参数等保持。
- 构造成功或异常时都恢复原spawner引用；不patch任何训练源码文件。
- summary保存本地路径、官方URL、SHA和size；默认不传参数仍保留原路径行为，不做不透明fallback。
- 另改善`evaluate_finished.py`失败信息：保存真实replay error、exit code和已执行步数，不再把所有错误统称policy/contract mismatch。

CPU读取该缓存确认：metersPerUnit=1、Z-up、一个Plane collision prim，无外部USD reference。文件未编辑；HTTP Last-Modified为2026-06-02。**原训练未保存地面文件hash，所以这里不是声称已对旧训练时下载的地面字节做了历史哈希比对**；可确认的是固定了相同官方版本URL的当前内容，且没有自行重建/修改地面几何。

### 独立A副本，没有混入B1

新代码根：

`/home/kaiser/robot-rl-sim60/experiments/a-evaluation-recovery-20260914T070110Z/isaac_wheeled_rl_train`

从冻结commit **`6ed83408d1d1cb02ea687e77fa80e5c20932f36c`** 归档，唯一源码补丁是`play_v40_onnx.py`的缓存入口。
逐项对照git对象确认：`env.py`、`env_cfg.py`、`core.py`、`contract.py`、`round3.py`、`assets/v40.py`、`train_v40.py`均与6ed8340逐字节相同。没有使用本地dirty B1代码评估A，也没有修改A原冻结目录。

模型身份：

```text
checkpoint SHA: 6d33498d2424bb45305c199d42261d62c39c13a39107a604a89d28ccd19f4af0
ONNX SHA:       3d8c682cfc94ac0ee1879bdc6fd90c208e123c05e6b48de3f970f42a474c1ad9
contract SHA:   e31ba1538be4f26d5a5c6452ca8b06e1d09f41db999118957909d13f7248e73e
```

## 4. 重评运行与统计口径

Kaiser既有WSL Sim6 runtime，1env、seed44、确定性最终ONNX、固定零速度。

- 四项按0.29→0.30→0.31→0.32串行执行，每项6000步、wall预算580秒，外层timeout620秒。
- 专用tmux：`a-evaluation-recovery-070110`，总外层timeout2700秒；已自然退出。
- 四项均`status=completed`、`stop_reason=step_budget`、exit0；无PPO/新训练。
- 各项3次原生timeout reset，episode长度为1999/1999/1999/3步；**累计60秒，不是连续无reset60秒**。
- 每个episode前2秒单列：初始化603帧/6.03秒；后续5397帧/53.97秒为统一“稳态统计窗口”，这个命名不代表额外证明收敛。
- 高度MAE/P95均为绝对误差，P95线性插值；平移速率为`hypot(vx,vy)`；XY位移按每回合首个已记录pre-reset pose到最后pose，不跨reset相减。

## 5. 初始化与稳态数据

### 每回合前2秒（合计603帧）

| 高度m | 高度MAE / P95 mm | 平移均速m/s | 非轮净力候选帧 | 非轮净力峰值N |
|---|---|---:|---:|---:|
| 0.29 | 3.023 / 13.069 | 0.025307 | 25 | 104.940 |
| 0.30 | 7.393 / 11.186 | 0.018097 | 0 | 0 |
| 0.31 | 7.572 / 9.462 | 0.015198 | 0 | 0 |
| 0.32 | 3.990 / 4.845 | 0.014692 | 0 | 0 |

### 2秒以后（合计5397帧）

| 高度m | 高度MAE / P95 mm | 平移均速 / P95 m/s | 稳态非轮净力候选 | 三个完整回合XY净位移m |
|---|---|---|---|---|
| 0.29 | **1.341 / 1.901** | **0.020668 / 0.036819** | 0/5397 | 0.375 / 0.379 / 0.373 |
| 0.30 | **9.230 / 9.333** | **0.035207 / 0.040868** | **5237/5397（97.04%）** | 0.555 / 0.650 / 0.650 |
| 0.31 | **6.973 / 7.492** | **0.010002 / 0.014307** | 0/5397 | 0.145 / 0.146 / 0.147 |
| 0.32 | **4.064 / 4.683** | **0.016850 / 0.023942** | 0/5397 | 0.198 / 0.191 / 0.191 |

四项稳态高度误差均为正偏差：分别平均高于命令1.341/9.230/6.973/4.064mm。0.30m实际高度范围约0.308050–0.309461m。

0.30m稳态非轮净力均值约13.996N、峰值44.882N；首个候选在step222、episode约2.22s出现，不是只存在于最初接触快照。它是实际诊断异常，但现有CSV没有body-pair身份，不能称为地面支撑、也不能证明轮胎打滑。

所有高度全程`nonfinite`、`knee_limit`、`instantaneous_tilt`、`low_height`、`base_visual_bounds_ground`、`failure_gravity`、`sustained_failure`均0；failure reset=0，timeout reset=3。`termination_flags`与`diagnostic_flags`分别核对，没有互相替代。

## 6. 与R2同高度基线对照

R2来自[ROUND2_HEIGHT_EVALUATION.md](ROUND2_HEIGHT_EVALUATION.md)及其原始精简证据；同Kaiser Sim6、固定零速度、相同分段口径。**R2 seed42，本次A seed44**，因此以下是描述性对照，不冒称同seed多次统计显著性实验，也不能把所有差异归因于单个reward项。

| 高度m | R2→A 稳态高度MAE mm | R2→A 高度P95 mm | R2→A 平移均速m/s | 观察 |
|---|---|---|---|---|
| 0.29 | 0.894→1.341 | 1.541→1.901 | 0.032988→0.020668 | 漂移减小，高度误差略大但仍在高度目标内 |
| 0.30 | 5.977→9.230 | 6.731→9.333 | 0.200596→0.035207 | 漂移显著减小，但高度变差，新增持续非轮净力候选 |
| 0.31 | 8.217→6.973 | 8.988→7.492 | 0.137416→0.010002 | 高度与零速移动均改善，高度MAE仍超5mm |
| 0.32 | 6.253→4.064 | 6.259→4.683 | 0.050161→0.016850 | 两项改善，达到本次预置研究criteria |

R2三个完整回合XY位移范围分别约0.579–0.766、3.924–3.927、2.714–2.747、0.991–0.992m；A对应范围见上表，四档均减小。0.29m初始化非轮候选从287帧降至25帧；但0.30m从稳态0帧变为5237帧，接触不能只看某个高度改善。

## 7. 判定与下一轮建议

保持原判据：稳态高度MAE≤5mm、P95≤10mm，平均平移速率≤0.02m/s，nonfinite/failure=0。没有为本次结果调宽阈值。

| 高度 | 高度精度 | 静止均速 | 组合研究criteria | 待处理 |
|---|---|---|---|---|
| 0.29 | 通过 | **未过**，0.020668>0.02 | 未过 | 不能四舍五入为0.02后判通过 |
| 0.30 | 未过，MAE9.230mm | 未过 | 未过 | 持续非轮净力及高度偏差优先定位 |
| 0.31 | 未过，MAE6.973mm | 通过 | 未过 | 保持减漂优势，改善高度偏差 |
| 0.32 | 通过 | 通过 | **通过** | 每回合仍约0.19m位移，不等于世界XY位置保持 |

下一轮可据此规划，但**不应无条件把A最终模型提升为“全高度站立通过”的父策略**：

1. 主agent结合已回收训练曲线选择少量checkpoint候选，用同一离线地面入口对照0.30m净力/高度与其它高度，避免只按最终reward选父模型。本任务没有重复下载全部checkpoint或扩大此评估。
2. B1可保留为有明确基线的受控pilot，不能声称摩擦DR会自动修复A的净力异常。正式promotion应明确检查0.30m及其它高度，而不是只看导出/回收SHA。
3. 0.30m接触对手身份、真实支持接触与滑移仍缺证据；后续补测不能靠关闭诊断、修改termination或盲目增加摩擦使指标消失。
4. 本次没有高度切换、行驶/旋转、推扰或地形效果结论；长时连续运行也不能由三个timeout回合推得。

## 8. 证据回收、修复测试与复现

新本机目录：

```text
/home/yukikaze/Documents/workspace/robot_rl/reports/current/kaiser_round3_a_20260914/
  evaluation-recovery-20260914T070110Z/
    diagnosis.json
    old-failed/                    # 原四项log、summary与旧failed结果
    recovered/
      results.json
      provenance.json
      ground-download.json
      default_environment.usd
      h029…h032/telemetry.csv, summary.json
      h029…h032.log
      h029…h032.receipt.json
```

远端新目录：`/home/kaiser/robot-rl-sim60/experiments/a-evaluation-recovery-20260914T070110Z/`。
完整模型保持在原A run中只读使用，没有再次拉取全部训练数据。恢复结果SHA：`ea5f6aae5e3aa8423ba8c03a9c0fa9cb23ae89e896b12596fd54687212b9a1a7`；每高度CSV SHA见精简JSON。

有效入口示例（已有目录不要重用，运行工作目录为新A副本代码根）：

```bash
source /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh
P=/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/train
G=/home/kaiser/robot-rl-sim60/experiments/a-evaluation-recovery-20260914T070110Z/default_environment.usd
timeout --signal=TERM --kill-after=20s 620s \
  python -u scripts/play_v40_onnx.py \
  --research --headless --num-envs 1 --seed 44 \
  --ground-usd "$G" --onnx "$P/policy.onnx" --contract "$P/contract.json" \
  --command 0 0 0.30 --max-steps 6000 --max-wall-seconds 580 \
  --report-dir /new/unique/recheck-output
```

恢复控制器及精确argv在远端`recover_a_evaluation.py`和每项receipt内；本次没有再跑额外确认场景。原自动评估尚未自动选择该缓存参数，未来部署应明确提供本地固定资源，而不是继续依赖评估收尾时外网可用。

CPU验证：`tests/test_onnx_replay.py`与`scripts/round3/test_post_evaluation.py`共 **62 passed**，覆盖缓存错误字节拒绝、构造期间只替换URL/保留物理参数、异常后恢复spawner、构造前再次哈希检查及原评估回归。

本次仅改评估脚本及必要测试/文档；本地B1功能改动完整保留，没有修改B1核心功能、原A冻结目录或旧receipt，没有commit/push。
