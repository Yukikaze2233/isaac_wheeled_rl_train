# 自起与高速优先的模型选择盘点

日期：2026-10-09。仅读取本机模型、导出回执、冻结边界与已有评测；没有 SSH、GPU、PPO、模型导出或模型修改。目标是为一个模型同时支持倒地脚本后接管与高速运动寻找有证据的候选，不把独立分支成绩合并。

## 结论

现有证据不能选出“倒地自起与高速都最强”的一个模型。最新 REC41374 是优先评估的恢复候选，已有可直接回放的 REC38000 是恢复候选，speed19837 是高速专项候选；三个候选必须在同一组固定回放条件中分别评估。flat14020 保留为低速基础对照，不作为本次第三/第四个交付候选。

较晚 checkpoint、多训练次数、ONNX 数值验证和 GUI 具有某个场景按钮，都不等于相应技能已通过。脚本自起与 RL 接管是两部分；交付只有 actor ONNX 不能代替恢复脚本、混合控制与正确交接状态。

## 已有真实文件与模型身份

| 身份 | 本机 ONNX | 已核验 SHA256 | 真实学习覆盖和限制 |
|---|---|---|---|
| flat14020 | `models/v6_flat_candidate_14020_20261003/policy.onnx` | `4006bf79182e14074f38c3e8f573fe1870fdfeba3fcc0760bb24cc5752161e8a` | flat 终点 actor13000，低速基础同条件证据较强；没有后续高速/恢复的成果。 |
| speed19837 | `models/v6_speed_candidate_19837_20261005/policy.onnx` | `294d80c8618fdc4baced6587c3c36edd8d97cfba3d99afd05b3639c681e8b738` | speed←flat 独立谱系，本域 actor3563。请求范围到±5 m/s、固定旋转最高±18.85 rad/s；范围不是成功率。没有脚本倒地自起后的 RL 接管专项覆盖。 |
| REC38000 | `models/v6_recovery_candidate_38000_20261008/policy.onnx` | `cd9c9be93ed65e56d9d4835ae9057490d271b4ad2e41cdf429778f1cf605b2ad` | recovery←flat 分支、REC10795；54恢复组+41普通组，普通覆盖仅|vx|≤1 m/s、|yaw|≤1 rad/s。不能赋予独立 speed 的高速成果。 |
| REC41374 | 本机 `models/` 未发现对应 ONNX | PT封存 `ed24fe4e5e62fd8c8e01f81427c0741ae46799a237dee2b0fbd4b288df4e1085` | REC14169、学习谱系27627；旧 REC 完整状态合法封存，尚无此权重同条件能力评测或本机导出证据。 |

以上前三个 ONNX 均读取实际文件计算 SHA，文件各 204185 bytes。speed 与 REC 包均有原始 PT、contract、agent_config、provenance、deployment/interface、ABI vectors 与 manifest。其 `qualified=null`、`hardware_deployment_ready=false` 保留；这里没有额外加入新门禁。

REC41374 的冻结远端入口由本机 `reports/v6_scut_posture_completion_20261006/source_repair_028/deployment_result_pending001.json` 与 `boundary_live007.json` 绑定：

`/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/recovery/attempt_009/train/model_final.pt`

源合同 SHA `eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2`；封存文件字段 `learning_updates=41374` 需与合法 `learning_lineage_updates=27627`、`task_actor_update=14169` 区分。它不包含尚未执行的 V26 联合续训成果。

## 现有评测可以证明什么

### 同条件的确定性固定对照

已有 `reports/v6_scut_posture_completion_20261006/mean_comparison_019/comparison_guard_candidate_004.summary.md` 比较 flat actor13000 与较旧 REC actor2458，两者同 seed290929、quick63、4回合/例、均值策略、0 PPO。结果：

| 指标 | flat13000 | 旧REC2458 |
|---|---:|---:|
| 305mm静止平面均速 | 0.484 cm/s | 0.547 cm/s |
| 前进0.5 m/s MAE | 1.735 cm/s | 26.007 cm/s |
| 后退0.5 m/s MAE | 1.226 cm/s | 25.745 cm/s |
| 正转 yaw 角速度 MAE | 0.010 rad/s | 0.309 rad/s |
| 真实驻停均速 | 3.177 cm/s | 1.516 cm/s |
| 严格逐回合整链完成 | 110/216 | 57/216 |

这证明旧 REC 在停车上更好、低速前后与正转精度退化。不是 REC38000/41374 的测量，也不是高速实绩；不能把旧结论按更新数量外推给较新恢复模型。另一个 20261003 包中的 flat vs REC16000 对照同样只对应那两个 PT SHA。

### REC38000 同模型的实际高速失败观察

`reports/v6_recovery_candidate_38000_playback_20261008/manual_session001.summary.json` 将日志绑定到上述 REC38000 ONNX SHA。一次手动高速反转请求回放里无 jump request，实发 yaw 约 −7.3…−7.9 rad/s 时出现双轮接触力零、FLIGHT、俯仰约40–42°，最终非轮接触；最大闭链间隙约 .674 mm，小于3 mm门。此证据说明这一恢复候选在该超出真实采样域的高速条件下失败，不能把它当可用高速模型；单次手动记录也不是对 V5.3 或 speed19837 的公平比较。

### 数值、工程预检与能力的区别

REC38000 的独立 CPU 数值对齐通过，25组 ABI 向量最大误差约3.8e−6；speed19837 也有 ONNX/ABI 对齐。这些证明导出正确，不证明稳态、恢复或高速能力。`verification/native_speed_018_final.identity.json` 虽名带 speed，实际 PT SHA 是 flat14020，scope 是工程探针；不能将它算作 speed19837 的学习成绩。GUI 18场景 CPU preflight 和 flat 启动也不代表18种技能完成。

## 最多三个候选的共同筛选方案

候选仅 **REC41374、REC38000、speed19837**，分别从相同冻结模型/合同快照评估。flat14020 原有证据作为低速锚点；若需要重跑该基线，应清楚标为对照而不是第四个新模型候选。

1. 固定清洁均值策略：四种倒地姿态→同一原生脚本→同一实际 blend→纯RL接稳→同一前后/正反转后续命令。分别列脚本完成率、纯RL连续接稳率、实际运动放行率、完整链完成率及交接 pitch/roll/omega/腿轮速度。不能把脚本完成算成策略自起完成。
2. 固定平地：230/305/430mm静止；0.5/1/2/3m/s双向、实际命令斜率一致的加减速/松杆停车；yaw±1/±π/±2π，再有条件扩展。记录速度/yaw角速度MAE、停车后均速、pitch/roll RMS/p95、轮支撑占比、非请求离地、身体接触、闭链/气簧残差，区分加速姿态与恒速姿态。
3. 相同毫秒延迟、摩擦/负载随机化复验最佳候选，只改变随机化条件，模型冻结。先给基本恢复与高速的实测表，再据用户优先级选一份交付；不存在自动把两个 specialist 权重或成绩拼成一个策略的合法捷径。

这些旁路比较不改变 PPO reward，不要求45项齐过才继续训练，亦不自动授予实车能力。若暂时只交现成文件，需按真实范围称为“恢复候选REC38000”或“高速候选speed19837”，不能称最强综合模型。

## 可复用导出、打包和交接入口

- 原生 CPU 导出：`scripts/export_chassis_candidate.py`。严格按所选 PT 的 contract/agent/资产/控制来源构造 RSL-RL MLPModel，复制原始 checkpoint，导出确定性均值；不得以当前 V26 合同改写旧权重身份。已有 REC38000 的 `export_receipt.json`、`numerical_parity.json`、`bundle_verification_receipt.json` 可复用工作流。
- 高速现成交接：`docs/V6_REAL_DEPLOYMENT_HANDOFF_20261005.md`、`models/v6_speed_candidate_19837_20261005/handoff_document.md`；归档 `models/v6_speed_candidate_19837_20261005_handoff_r2.tar.gz` 附 SHA。它明确写出未覆盖倒地脚本接管。
- 低速基础交接：`docs/V6_FLAT_DEPLOYMENT_HANDOFF_20261003.md`；完整原始 PT 位于 `models/v6_model_delivery_20261003T2235/recommended_flat/`，不能因 flat 单独小包没有 PT 而认为完整封存不存在。
- 恢复本机回放：`docs/CHASSIS_GUI_SUITE.md` 与 `scripts/play_chassis_suite.sh`，默认固定 REC38000 同一 ONNX。fallen_left/right/belly/back 使用原生自起FSM、真实blend与RL；界面的高速/跳跃挑战不是已学能力声明。
- 所选新模型交接必须带自身 SHA、原始PT/合同来源、35D→6D观测动作序、50Hz策略/200Hz仿真反馈PD、腿160/2.5与动作缩放、轮侧响应、命令组合投影/斜率、完整恢复脚本与交接语义、实测能力表。只替换 ONNX 不能假设 RMCS 的恢复FSM/上下文已经一致。腿侧bag未qualified状态保持。

本审计没有重新运行模型、归档或硬件。所有后续选择以新生成的同模型证据回执更新，不将本报告中的比较计划当已完成测试。

## 本轮后续补证

root 随后已取回并导出 REC41374，ONNX 实际 SHA `63574953c97aad374fa1554c53486cd540a3019857af72e0c06cc010214299bf`，目录 `models/v6_recovery_candidate_41374_20261009/`。同源 REC41374／38000 的独立旁路两 lane 原生均值回放也已实际完成；未启动旧 speed 的跨 FSM 公平比较。两 REC 都在本次左侧倒地链 7.46 s 成功，REC38000 的共同制动窗口明显更稳；最新更快的加速伴随较大倾角，且在高 yaw 足够曝光前发生侧界截断。不能称最新为综合最强，也不能授予旧模型高速实车资格。详见 `model_compare/native_comparison_addendum.md`、同名 JSON 和 `model_compare/MODEL_HANDOFF_ADDENDUM_20261009.md`；其余三倒地、普通上台阶、跳阶、rough、主动跳跃、反向高速分别未测。以上首次静态盘点与后续实测按时序分开，不把最初比较方案冒充全量执行。
