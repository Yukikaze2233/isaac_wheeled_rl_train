# Round2 最终 ONNX：Kaiser 五高度零速站立实测

实测时间：北京时间 **2026-09-12 23:55:45 至 2026-09-13 00:14:43**。

## 结论：能维持五档附近的高度与姿态，但尚不能认为五档都已原地站稳

**五个高度都已实际完整跑完，不是依据训练曲线推测。** 0.28/0.29/0.30/0.31/0.32 m 各6000步、累计60仿真秒，固定命令 `(vx=0,wz=0,height)`、seed42。

- 全部数值有限，均为0次任务失败终止、3次timeout reset；瞬时倾角越35°、实际持续失稳、低高度、膝硬界异常诊断均为0。
- **所有高度都出现零速漂移**，稳态平均平移速率为0.033–0.201 m/s，均超过此前建议的0.02 m/s静止目标。不能把“未倒下”表述为“原地站住”。
- **0.29 m 的高度控制最好**：稳态MAE0.894 mm、P95 1.541 mm，稳态无非轮净接触诊断；但每个约20秒回合仍漂移0.58–0.77 m。
- **0.28 m 有持续非轮净力异常**：稳态5397/5397帧超过1 N，平均约450 N、峰值890 N。现有字段没有碰撞对手身份，不能认定为地面支撑，也不能将其忽略后声称站立验收通过。
- 0.30/0.31 m 无非轮接触诊断，但漂移明显，每回合约3.9/2.7 m；0.32 m 每回合约0.99 m。后续训练应针对零速稳定性，而不是仅延长现有训练等待自然改善。

**这里的60秒是累计时间，绝不是连续无reset站立60秒。** 合同20秒timeout保持原样；实测在总step1999、3998、5997触发timeout，前三回合各1999步，最后还有一个3步的部分回合。

## 1. 运行身份与隔离

| 项目 | 实际设置 |
|---|---|
| 主机/运行栈 | Kaiser WSL，已部署Sim6，CUDA PhysX，单环境 |
| Runtime | `/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh` |
| 源码 | `/home/kaiser/robot-rl-sim60/v40-live-snapshot03` |
| 模型 | Round2服务器最终ONNX，CPUExecutionProvider，确定性raw mean |
| 原训练栈 | Sim5.1 / Lab2.3 / RSL3；此次为Sim6跨版本评估 |
| 本次实际版本 | Sim6.0.0.1、Lab6.1.14、RSL5.5.1、Torch2.11.0+cu128、torchvision0.26.0+cu128、ORT1.30.0 |
| Lab源码 | `ffff603eafc6b74264a5261cc0183d6a65390d78`，`v3.0.0-beta2.patch1` |
| 环境参数 | `--headless --num-envs 1 --seed 42 --device cuda:0`；physics_dt=0.005、policy_dt=0.01 |
| Evaluation reset | 原有固定命令行为；观测噪声关闭、根reset速度0；初始nominal高度0.32 m |
| 任务隔离 | tmux `v40-r2-height-155346`，五高度顺序执行；每项外层620s timeout，脚本wall预算580s，总任务3300s上界 |
| 退出 | 五项进程均exit0、`stop_reason=step_budget`；批任务完成，专用tmux已自然退出 |
| USD | 各run独立USD导入缓存，未复用旧run、未设置外部USD seed |

开始前核对远端`env.py`、`play_v40_onnx.py`、`train_v40.py`、资产构造代码与本地当前版本SHA完全一致，已包含修正后的真实`diagnostic_flags`。结束后再次核对四个源文件，未改变。68个资产文件与最终训练`source_hashes.json`逐项一致，五次preflight和ONNX sidecar/contract/asset检查全部通过。

关键身份：

```text
policy.onnx:
7d4a93f0d0692f914dc769fb94c88550de62e44a0b3553d95b20d80d406b700e
contract.json file SHA256:
5b97bf2935b5bb04357737e2a54046ab891c15bc15d6e4c2ad9d4d6150e3e8cc
contract semantic digest:
667865de8352e67cff6720af9a142d00ff9b980132b27e4993a25c8e09d3d22e
asset manifest:
df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886
env.py:
e94314e55133884106349481929a80fe937dafdfd7a4ca313aa927c964fb07ad
play_v40_onnx.py:
91064e8a1b120d355d79ce5a49b2489e2a83c4bb854f32683ccb68ec2a50bb72
```

完整版本、其余源码/文件哈希和每高度CSV/summary哈希保存在下述精简JSON。

## 2. 统计口径

每项从真实`telemetry.csv`的6000个**pre-reset snapshot**统计，并核对summary逐项诊断计数、termination、timeout、action/reward固定高度与全部数值有限性。

- **初始化段**：每次reset之后`episode_time_s <= 2.0`，共603帧/6.03秒。包含前三回合各200帧，以及最后部分回合3帧；不是仅剔除整条轨迹的最初2秒。
- **稳态统计段**：`episode_time_s > 2.0`，共5397帧/53.97秒。“稳态”只是统一时间窗口命名，不表示已证明动力学收敛。
- 高度误差：`base-link world z - commanded height`；平地env origin z为0。MAE、P95均为绝对误差；P95采用排序位置`0.95*(n-1)`线性插值。
- 平移速率：`hypot(vx,vy)`，使用snapshot提供的root COM速度在body系的两个平面分量；不是带正负抵消的平均vx。
- 位移：**每回合内**root-link世界XY首个已记录姿态到最后一个姿态的净位移，不跨reset相减；首样本为episode约0.01s，没有虚构t=0的CSV记录。
- 非轮净力：非轮body的history最大净力，再取非轮body最大值。阈值1 N；既不是独立碰撞次数，也不是ground-pair力。

## 3. 前2秒初始化统计

各高度均为603帧，包含所有reset后的初始化窗口。

| 命令高度 m | 高度MAE mm | 高度P95 mm | 平移速率均值 m/s | 平移速率P95 m/s | 非轮净力>1N帧 | 净力峰值 N |
|---|---:|---:|---:|---:|---:|---:|
| 0.28 | 3.347 | 19.104 | 0.05784 | 0.12670 | 508 | 657.69 |
| 0.29 | 3.749 | 17.645 | 0.06006 | 0.15809 | 287 | 88.24 |
| 0.30 | 7.670 | 14.416 | 0.18067 | 0.21675 | 0 | 0 |
| 0.31 | 8.991 | 11.907 | 0.13302 | 0.20740 | 0 | 0 |
| 0.32 | 6.163 | 7.540 | 0.04583 | 0.05076 | 0 | 0 |

所有高度都从同一nominal reset姿态开始，并非各自按目标高度IK重置，因此初始化误差包含降/升高过程。
0.29 m 的287帧非轮净力全部在这些初始化窗口内；它确实在后续统计段消失，但不能把全程表述为零接触。

## 4. 2秒以后的稳态统计

各高度均为5397帧。以下五项的高度偏差均为正，高度MAE也等于正偏差均值。

| 命令高度 m | 高度MAE mm | 高度P95 mm | 实际高度范围 m | 平移速率均值 m/s | 平移速率P95 m/s | 非轮净力>1N比例 |
|---|---:|---:|---|---:|---:|---:|
| 0.28 | **4.185** | **11.207** | 0.280291–0.293977 | **0.15001** | 0.28504 | **100%** |
| 0.29 | **0.894** | **1.541** | 0.290478–0.292439 | **0.03299** | 0.05969 | 0% |
| 0.30 | **5.977** | **6.731** | 0.305102–0.306867 | **0.20060** | 0.21931 | 0% |
| 0.31 | **8.217** | **8.988** | 0.316332–0.319617 | **0.13742** | 0.17306 | 0% |
| 0.32 | **6.253** | **6.259** | 0.326234–0.326272 | **0.05016** | 0.05027 | 0% |

0.28 m 稳态非轮净力：均值449.69 N、P95 754.86 N、峰值889.99 N。其三个完整回合的稳态高度MAE为4.116/4.297/4.141 mm，接触异常持续出现；无需为了看到同一现象再扩成长测试。

## 5. 每回合位移：timeout不会消除已经发生的漂移

前三回合各1999步；表中净位移使用episode第1步至第1999步，稳态位移使用第201步至第1999步。单位均为m。

| 高度 m | 三个完整回合净位移 | 三个回合仅稳态段净位移 | 每回合采样XY轨迹路程范围 |
|---|---|---|---|
| 0.28 | 0.835 / 0.778 / 0.823 | 0.839 / 0.784 / 0.837 | 2.874–2.907 |
| 0.29 | 0.579 / 0.720 / 0.766 | 0.477 / 0.616 / 0.661 | 0.612–0.799 |
| 0.30 | 3.925 / 3.924 / 3.927 | 3.570 / 3.576 / 3.572 | 4.000–4.002 |
| 0.31 | 2.720 / 2.747 / 2.714 | 2.440 / 2.508 / 2.442 | 2.733–2.767 |
| 0.32 | 0.991 / 0.992 / 0.991 | 0.902 / 0.902 / 0.902 | 0.995–0.996 |

净位移不是走过的路程：0.28 m 路程远大于净位移，不能只看首尾相距约0.8 m就忽略过程中的移动。最后3帧的部分回合位移仅保留在JSON中，不与完整回合混排。

## 6. 真实诊断与终止

| 高度 m | 全程非轮净接触帧 / 6000 | 稳态非轮净接触帧 / 5397 | failure终止 | timeout reset |
|---|---:|---:|---:|---:|
| 0.28 | 5905 | 5397 | 0 | 3 |
| 0.29 | 287 | 0 | 0 | 3 |
| 0.30 | 0 | 0 | 0 | 3 |
| 0.31 | 0 | 0 | 0 | 3 |
| 0.32 | 0 | 0 | 0 | 3 |

五项全程下列diagnostic计数全部为0：`nonfinite`、`knee_limit`、`instantaneous_tilt`、`low_height`、`base_visual_bounds_ground`、`failure_gravity`、`sustained_failure`。全部`termination_flags`也为0，二者由不同字段核对，没有相互替代。

`knee_limit`来自真实关节位置与原合同有限膝硬界（含0.001rad tolerance）的比较，未检测到越界。**CSV没有逐关节q/内角列，因此没有声称获得完整实测内角范围或最小限位裕量。** 其他continuous关节不能被错误地当作有限膝界。

0.28 m 的净力是否来自自碰撞、未排除的内部接触或腿对地接触，现有CSV不能回答。此次未改物理模型、未增加API、未通过放宽碰撞/终止定义“修好”结果。需要碰撞对手身份或等价独立证据后，才能判断它是否依赖非轮地面支撑。

## 7. 哪些通过、未通过、还需评估

这是**研究指标**，不是硬件放行门槛。采用此前独立审计提出的稳态高度MAE≤5mm且P95≤10mm作为高度精度子项，另以平均平移速率≤0.02m/s作为静止子项；二者均不等同位置保持。

| 高度 | 未倒下/姿态保持的实测证据 | 高度精度子项 | 静止速率子项 | 当前结论 |
|---|---|---|---|---|
| 0.28 | 有，三个约20s回合 | 未过，P95超10mm | 未过 | 持续净力异常，站立支撑性质需追加诊断 |
| 0.29 | 有，三个约20s回合 | **通过** | 未过 | 已有良好高度控制；仍有慢漂，不能称原地站稳 |
| 0.30 | 有，三个约20s回合 | 未过，MAE超5mm | 未过 | 维持高度附近，但明显后退漂移 |
| 0.31 | 有，三个约20s回合 | 未过，MAE超5mm | 未过 | 维持高度附近，但明显后退漂移 |
| 0.32 | 有，三个约20s回合 | 未过，MAE超5mm | 未过 | 高度偏高约6.25mm，仍持续慢漂 |

不能据这些结果宣称连续60秒稳定、所有随机seed稳定、在训练原Sim5.1栈数值相同或实物已可用；此次也没有推扰/地形/动态高度切换。五高度的基本能力证据已经补齐，下一步优先改进零速漂移并定位0.28 m净力来源，而不是再做大规模重复测试。

## 8. 证据与实际运行入口

可随代码仓库保存的小文件：

- [精简结果JSON](evidence/round2-height-kaiser-20260912T155346Z.json)：约57KB，包含分阶段/逐回合指标、版本、哈希。
- [精简结果CSV](evidence/round2-height-kaiser-20260912T155346Z.csv)：15行，五高度×全程/初始化/稳态。

原始证据，不作为源码提交内容：

- 远端：`/home/kaiser/robot-rl-sim60/round2-height-20260912T155346Z/`。
- 本工作树：`reports/round2-height-kaiser-20260912T155346Z/`。
- `h028`…`h032`分别保存完整`telemetry.csv`、`summary.json`；各高度`*.receipt.json`保存完整argv、cwd、开始/结束时间和exit code。
- `provenance.json`保存起始身份核验；`batch_completion.json`保存完整五项完成收据。
- `run_evaluation.py`是本次实际运行的顺序、有界入口；`analyze_evaluation.py`为可复算分析脚本。它们只在独立reports目录，不是训练API。
- 完整Kit日志、run-local USD及策略副本留在远端独立目录，未复制到`docs/evidence`。未复制checkpoint。

实际tmux任务使用：

```bash
tmux new-session -d -s v40-r2-height-155346 \
  -c /home/kaiser/robot-rl-sim60/v40-live-snapshot03 \
  "timeout --signal=TERM --kill-after=30s 3300s python3 -u /home/kaiser/robot-rl-sim60/round2-height-20260912T155346Z/run_evaluation.py > /home/kaiser/robot-rl-sim60/round2-height-20260912T155346Z/batch.log 2>&1"
```

上述目录已有结果，**不要原样重跑覆盖**。如需单高度复核，在新的独立tmux里使用同一runtime与入口，且指定新report路径：

```bash
source /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh
S=/home/kaiser/robot-rl-sim60/v40-live-snapshot03
P=/home/kaiser/robot-rl-sim60/round2-height-20260912T155346Z/policy
timeout --signal=TERM --kill-after=20s 620s \
  python -u "$S/scripts/play_v40_onnx.py" \
  --research --headless --num-envs 1 --seed 42 --device cuda:0 \
  --onnx "$P/policy.onnx" --contract "$P/contract.json" \
  --command 0 0 0.30 --max-steps 6000 --max-wall-seconds 580 \
  --report-dir "/home/kaiser/robot-rl-sim60/r2-height-recheck-h030-$(date -u +%Y%m%dT%H%M%SZ)"
```

本次没有Round3训练，没有更改训练源/合同/奖励/终止或新增API，没有commit/push。
