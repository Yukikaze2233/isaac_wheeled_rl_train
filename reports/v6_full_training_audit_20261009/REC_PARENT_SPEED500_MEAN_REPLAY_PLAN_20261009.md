# REC 实际父与 source029 speed500：固定均值动作回放准备

状态：只读准备；未启动 Isaac、GPU、PPO 或评测。speed 首个完整 500 actor seal 尚未认证；小套跨合同回放 adapter 尚未实现。该方案不改变课程、门禁、账本或 qualification，也不要求停止当前 speed。

## 1. 输入先固定，不能拿旧交付包代替真实父

- 实际父：`train/recovery/attempt_010/train/model_final.pt`，SHA-256 `e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967`；消费 41393、actor 消费 41223、REC actor 14188、学习谱系 27646。
- 父合同：`contracts/v6_task_campaign_recovery_worker_v21.json`，SHA `eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2`。
- 子：source029、`train/speed/attempt_003/train` 的 **task actor=500 完整不可变检查点**；具体文件名、PT/合同/seal SHA 和所有时钟等首个真实 seal 后读回，不猜 live `model_N.pt`。
- 子合同：`contracts/v6_task_campaign_speed_worker_v26.json`，SHA `941fb9e1f7a8ced3aa83fca76e82b7cf3064b594602d1e2900e56e0df0cc6b64`。
- 两者各自保留 authentic `.contract.json`、`.checkpoint.json`、原生 `agent_config.json`，PT 内身份与 sidecar 相符。先认证原合同，再验证共同 ABI；不修改 ONNX metadata 让不同合同看起来相同。
- 实际父的五文件本机副本已取回至 `reports/v6_full_training_audit_20261009/source029_parent_rec41393/`。`source029_parent_download001.json` 核对每文件 SHA；`source029_parent_verification001.json` 使用冻结 e5 原生 CPU 工厂严格恢复9个actor、8个critic张量和17个Adam状态，身份/控制/资产/RNG/时钟一致，0GPU/0PPO/0远端写。这是原始PT副本，尚未导出新的ONNX。
- 本机现有 `models/v6_recovery_priority_handoff_20261009_r2/model/policy.onnx` 仍来自 REC14169/消费41374/SHA `ed24fe4e5e62fd8c8e01f81427c0741ae46799a237dee2b0fbd4b288df4e1085`。可以另列历史基线，不能替代上述精确父。

共同控制 ABI 已按冻结 JSON 只读比较一致：单帧35D actor/81D critic、6动作、history=1、manual35、相同动作顺序/`v5_control`/机械资产/控制源，策略50Hz（20ms）、物理反馈PD200Hz（5ms）、160/2.5。使用真实源 `e5bd9c1367b875d7f229b9a1930647013d023372`，不换成未部署的 host-sync/audit/v27 候选。

## 2. 可以复用的入口与严格边界

| 入口 | 已具备 | 不能直接假定 |
|---|---|---|
| `scripts/export_chassis_candidate.py` | CPU-only PT/seal/合同/原生MLP认证，导出确定性均值35→6，ORT数值校验 | 导出不证明物理能力；两个模型必须分别导出、保留各自合同 |
| `scripts/evaluate_chassis.py` | 两次 `--checkpoint`，同一 FixedCaseEnv 依次 reset/reseed，均值动作、首回合、现有 EpisodeMetrics/RequestProtocolMetrics、无 PPO 更新 | 不能任意用 `--cases` 减成7项：REC full设计行63项、speed full193项、flat full87项；quick/confirmation投影仍须保留全部必需项及原行范围/4副本 |
| `reports/.../model_compare/compare_mean.py` | 两模型同场景、明确 requested/applied、真实定时松杆、事件归属/首次pureRL/放行与终止记录、原生metrics/TB event | 当前硬要求两份ONNX源合同逐字相同；只投影 flat+fallen_left；不能直接放入REC21和speed26 |
| `scripts/play_chassis_suite.py/.sh` | 原生 FixedCaseEnv/FSM、四倒地场景、headless有界单模型ORT均值 | 只支持固定 `--initial-vx/yaw` 或人工WASD，没有定时release CLI；单flat场景默认305mm；人工松杆不是可复算固定协议 |

**因此本次不启动63/87/193项全桶，也不通过删除 required names 绕过 full/quick 约束。** 目标是复用 suite 的独立 `interactive_unscored` 设计投影；小套 adapter pending，而不是一个已可运行的7项正式评测命令。

脚本 SHA（e5）：export `5623f95a3da34c973f5f9898814da23faa33bc1698cc59de7829f373fd48e473`；evaluate `d6937572f1c2d8e3a4cd291bf67ba289884afd3fdc1a01ee97b6c5393a15be64`；play `71b4dbafa9ef46fd75ff9efee899907f1f03d89b8e97ba3ebf7fc4a733f92d59`。既有 compare_mean SHA `1a96310816e5509d83a639ce7e131a303921b0f0d1c1cc4f901c42a7e03ca439`，其 `ROOT` 目前固定由脚本位置推导至主仓库，未来 adapter 必须显式绑定同一真实冻结 runtime，不能默默用候选主树。

## 3. 可执行的既有命令（仅准备，未运行）

取得两个真实认证副本后分别 CPU 导出，路径变量必须填入已读回 seal 的实际值；输出目录须不存在：

```bash
RUNTIME=/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009
PY=/home/yukikaze/isaacsim60-venv/bin/python
REC_PT=/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/source029_parent_rec41393/model_final.pt
SPEED_PT=/absolute/path/to/authenticated_speed_actor500/model.pt
REC_EXPORT=/absolute/path/to/new_REC14188_export
SPEED_EXPORT=/absolute/path/to/new_speed500_export
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$RUNTIME/scripts/export_chassis_candidate.py" --checkpoint "$REC_PT" --output "$REC_EXPORT"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$RUNTIME/scripts/export_chassis_candidate.py" --checkpoint "$SPEED_PT" --output "$SPEED_EXPORT"
```

既有 compare_mean 的准确命令接口如下。**只对同合同的历史REC模型可直接双输入预检，跨REC/speed目前会正确拒绝；不要把下一行当作本次已可运行的比较。** `--preflight-only` 不造场景；去掉该参数才启动原生仿真，当前不做：

```bash
"$PY" /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/model_compare/compare_mean.py \
  --onnx /absolute/path/to/REC_same_contract_A/policy.onnx \
  --onnx /absolute/path/to/REC_same_contract_B/policy.onnx \
  --output /absolute/path/to/new_preflight_evidence --steps 600 --device cpu --wide-floor8 --preflight-only
```

固定套的现成接口为 `evaluate_chassis.py --contract C --checkpoint REC_PT --checkpoint SPEED_PT --role full --seed 290929 --cases <C绑定设计full行的完整有序名单> --output OUT --device cpu`。不能把名单替换为7个focus case；确认第二seed应使用 `--role confirmation --seed 290930` 及其独立设计名单。本次只记录接口，不安排全桶运行。

## 4. 固定协议、零请求与统计口径

| 组 | 冻结协议 | 必须分开报告 |
|---|---|---|
| `stand_230mm/305mm/430mm` | 原用例cmd `[0,0,h]`；10s、4固定副本、seed290929，确认290930；原warmup1s保持 | 世界平面速度、10s位移峰值、height MAE、pitch/roll/角速度、失败/边界、有效时间；累计位移不能代替零速段速度 |
| 真实制动 | 使用已有 `flat_request(t)` 与 `advance_playback_command`；目标瞬时归零，实际命令仍沿1.5m/s²与4rad/s² slew | 请求时刻、实际零指令时刻、减速段与零指令段；记录requested/applied及35D实际命令，不能把“减速中”混为停车 |
| 四类原始倒地链 | `recovery_left_side_90_stand`、`recovery_right_side_90_stand`、`recovery_prone_belly_legs_above_stand`、`recovery_prone_back_legs_above_stand`；各20s、跟随 `[0,0,.305]`、exact birth，禁L0/L1替换 | 每类结束回合分母，脚本完成/首次pureRL/连续接稳/放行/后续完成；pureRL拥有样本、脚本失败与RL失败分列，不能用混课L2总率替代四类 |

已有 compare_mean 的请求表：0–1s静站；1–5s `[2,0,.305]`；5–7s零请求；7–10s `[0,2π,.305]`；10–12s零请求。原制动统计窗5–7s、实际零速指令窗6.34–7s、旋转后实际零窗11.58–12s。这里“零”指**实际命令为零**，不是机器人实际速度为零。这是真release，却是短观察挑战：2m/s与2πrad/s超过原REC训练域，零指令保持不到1s，不能当10s驻停验收。复用时应明确保留此挑战标签；要比较10s持续停车，最小还需固定更长quiet窗口/相应有界steps，不能冒称现有600tick已覆盖。

四链使用既有物理判据，不加 gate：脚本≤8s、一次blend/一次handover、pureRL、连续物理接稳≥1s、有效放行、连续后续≥1s、无失败/截断。原生物理接稳含双轮接触/无非法接触、倾角<12°、roll/pitch角速度范数<.75rad/s、高度误差≤5cm、持稳期平面速度<.25m/s；stand后续平面速度<.10m/s，同时跟踪误差满足原线。报告原始指标和事件分母，不把诸项全过作为继续PPO的新阻断门。

## 5. 固定环境与当前恢复脚本不能混称

固定套 `frozen_module_enablement` 开启 dynamics/contact/command_transport，关闭 signal_perturbations 与 signal_delay；虽然raw `observation_noise_enabled=true`，SCUT35实际额外观测噪声路径是 `self.perturbations.observation`，关闭后没有该注入。无高斯动作采样。基础case没有专属dyn/contact/link profile：已有固定case机制使其名义值/无丢包配置；变体case才加载显式profile。固定suite不同于训练中的ms延迟、探索和加难DR，不能将训练窗趋势直接当均值遗忘结论。

同一scene一次初始化，依次候选使用 `reset_suite()` 重设reset/RNG/USB（若开启）、固定副本与原生出生，只计首terminal及之前帧；terminal后auto-reset的frame不进入统计，未到结尾的是右删失。物理/命令/重置/RNG/controller设置均在两模型间相同；同seed不代表跨GPU逐位轨迹等价。

**隐藏投影差别已确认**：REC21正式训练配置side speed=1.5rad/s、`complete_started_blend=true`；`fixed_suite_contract` 经v20→v19→v18递归恢复旧exam配置，最终side speed为默认6rad/s、旧blend语义，使用v2几何和wheel_balance。speed26固定套走native speed18旧exam，其恢复配置还来自v1。故不能分别用各模型自己的投影后宣称“同场景同控制”。最小共同协议必须选定一份 **REC原固定exam投影** 同时供两actor，清楚标注历史examFSM；若要测当前实际1.5rad/s接管，则需另一份显式unscored投影绑定REC21训练controller与exact四出生，不能隐式修改正式exam。

## 6. 最小 pending 工作与产物

只需对既有 compare_mean/suite support 做隔离、未评分 adapter：逐模型原合同认证＋共同ABI比对；显式绑定e5 runtime和同一共同场景/controller；从独立冻结设计行投影三静站＋四原始倒地，另复用现有定时请求制动函数。固定设计投影须独立核对允许改动，不删正式required或伪装成full/quick。长quiet窗口/正反向与高度扩展是固定协议参数，不再造第二FSM。当前该adapter与CPU事件归属验证均 **pending**，没有生成可误部署合同或新增框架。

预期产物复用 `report.json`、`candidate_00/01.json`、原生metrics/首次事件时间、ownership与command曝光、终止与right-censor、TensorBoard event和必要trace；保留输入PT/ONNX/合同/seal/source SHA及startup控制读回。动作均值差异可解释固定条件下能力保持，仍不授实车资格；统一采样动作对照可另做，不能与均值结果混分母。本机不启TensorBoard服务。
