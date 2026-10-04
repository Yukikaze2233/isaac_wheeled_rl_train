# 四倒地初态的实际物理前缀诊断

## native_prefix_001：记录器误判，未形成行为结果

2026-10-04 23:06 UTC，旧flat终点PT SHA `7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f`、显式V8 recovery full-role、固定4个首批出生，实际启动了Isaac GPU。运行在第1个5ms物理子步被记录器的pure-RL断言中止。

源码中初始被动释放期间，FSM保持RL枚举9/blend=1的占位状态，但`release_finished=false`，原生motor_enabled=false。首帧真实post-mix与applied motor torque均为六轴零力矩；它不是已经完成BLEND的RL所有权。记录器仅据phase=9就检查40子步BLEND属于诊断实现错误，不能当作原生FSM绕过BLEND、策略摔倒或自起失败。

保留 `native_prefix_001/report.json`、首帧与事件快照，不覆写。实际曝光0.005s、未完成任何20ms策略间隔、4个首批回合均为右删失；不能把这个0/4视为20秒自起成功率。PPO与优化器更新为0，Kaiser训练和合同未改动。后续只修记录器以区分raw phase和实际释放/电机所有权，再按同一合同和固定N=4重测。

## native_prefix_002：真实四姿态运行完成，两个脚本失败、两个接稳完成

2026-10-04 23:17:43 UTC 开始，记录器修正后实际运行同一旧 flat 终点 actor（PT SHA `7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f`，flat task actor 13000、学习更新计数 14020），使用 V8 recovery full-role、`b4ce809` 冻结物理核心。**不是正在训练的 v14 speed actor**。每姿态只计第一次出生、固定 seed 290929、N=4；仍保留 `qualification:false`，PPO/optimizer 更新均为 0。

| 实际 lane / 姿态 | 终局 | 首次真实纯 RL | 原生终止时间 | 最后控制归属 |
| --- | --- | --- | --- | --- |
| 0 / 左侧 90° | 脚本失败 | 未发生 | 8.00s | PREPARE(7)，script |
| 1 / 右侧 90° | 接稳后续完成 | 5.615s | 7.66s | RL(9)，rl_unlocked |
| 2 / 腹卧、腿在上方 | 接稳后续完成 | 5.190s | 7.20s | RL(9)，rl_unlocked |
| 3 / 仰卧、腿在上方 | 脚本失败 | 未发生 | 8.00s | PREPARE(7)，script |

首批整链完成 **2/4**，脚本失败 **2/4**；timeout、boundary、未结束右删失均为 **0/4**。全部首批回合在 8 秒内关闭，故实际曝光为 400 个 20ms policy interval、1600 个 5ms 物理步，墙钟约 40.26 秒；请求的 20 秒只是最大 horizon。记录 6172 个 lane×物理帧、1543 个 lane×策略帧、68 个事件，不能将聚合帧数当单个机器人的连续时长。

四个 lane 最初各有 201 个 `passive_release` 子步，真实 upstream/applied 六轴力矩均为零，motor disabled、release unfinished、native pure-RL false。期间 raw phase=9 是初始化占位，未被计为接管。两个成功 lane 才真实经过 **40×5ms BLEND**：腹卧在 4.990–5.185s 应用混合力矩、5.190s 首个纯 RL；右侧在 5.415–5.610s 应用混合力矩、5.615s 首个纯 RL。随后分别完成 truth dwell、sensor dwell、motion unlock 和 1 秒 followup，两个 truth dwell 均先于解锁，`premature_release:false`。

两个失败 lane 始终只有 `passive_release:201` 与 `script:1399` 个实际子步；**没有 BLEND、没有纯 RL、没有运动解锁**。两者在 PREPARE 失稳后各出现一次重新 FOLD/ORBIT/THRUST/CAPTURE/PREPARE，最终仍未达到交接条件。原生 terminal 的唯一 failure reason 是 `recovery_script:true`，`recovery_actor_failure`、`recovery_premature_release`、fall、boundary 均为 false。

按已认证的冻结 `RecoveryTraining.failed/diagnostics` 重建，两者为 **failure code 6：脚本预算耗尽前尚未接管**。外层 elapsed 在最后一次 torque 前约 7.9951s，调用后越过 8s；此时原生 FSM 仍为 PREPARE(7)，而非 FAILED(10)，失败来自外层脚本预算。**数字 code 没有直接导出在本次审计的四个原始 JSON/JSONL 中**，回执将其明确标为源码重建；8 秒外层预算也包括最初约 1 秒被动释放，不能称作完成了 8 秒有力矩脚本。

失败的 PREPARE 阶段，左侧 427 个、仰卧 404 个实际子步中，所有 sensor gate 同时为 true 的帧均为 0，`ready_ticks` 最大值均为 0，因而没有形成原生 0.1 秒 ready dwell。最后一帧未过门分别为：

- 左侧：`body_clear / tilt / wheel_speed`；height、support 等最后一帧通过，不代表此前连续稳定。
- 仰卧：`support / body_clear / tilt / gyro / leg_speed`；height、wheel_speed 最后一帧通过。

因此本轮暴露的是**两个指定物理初态到交接入口的脚本能力缺口**；不能归因于 RL 接稳奖励、正在训练的 v14 遗忘，也不能把 handed-over 后 2/2 完成冒充首批整链全部完成。N=4、单种子、四个 stand followup、full-role 延迟关闭的诊断，尚未证明全姿态、运动后续、粗糙地形、台阶或 delay-role 的自起资格。没有新增成功门槛，也没有据此修改训练。

原始结果保留于 `native_prefix_002/{report.json,cohort_summary.json,events.jsonl,physical_frames.jsonl}`；四份文件 SHA、实际分区/阶段/门与 failure-code 重建边界见 [NATIVE_PREFIX_002_RECEIPT.json](NATIVE_PREFIX_002_RECEIPT.json)。

主任务已将原报告派生的 step 0 诊断 event 核对 SHA 后上传至 Kaiser 原有 TensorBoard；[TensorBoard](http://127.0.0.1:6006/#scalars) 中别名为 `v6-recovery-prefix-flat-parent-20261005`，上传回执见 [tensorboard_upload_receipt.json](native_prefix_002/tensorboard_upload_receipt.json)。这是固定四样本的单次结果面板，不能读作 PPO 更新曲线；没有启动本机 TensorBoard 或重启训练。

### 逐时刻物理诊断曲线

原始200Hz物理帧每四帧取一帧，派生1543个lane-frame的50Hz诊断曲线，已核对event SHA并上传同一Kaiser别名；工具为 `write_prefix_trace_tensorboard.py`，回执为 `native_prefix_002/trace_tensorboard/{trace_writer_receipt.json,trace_upload_receipt.json}`。`RecoveryPrefixTrace/...` 的step是原始5ms物理tick，`sim_time_s`给实际秒数，绝不是PPO iteration。记录姿态、角速度、原生阶段/八门/ready dwell/BLEND和实发六轴力矩。姿态来自物理推进后的root四元数，阶段与门来自发力前时刻，不能混作同一采样点。

失败两lane的实际RL ownership始终为0；成功两lane的40子步BLEND仍由原200Hz帧认证，50Hz曲线只用于查看时序。此前failure code 6的源码重建说明只涉及那四份指定原始文件；另存的 `policy_frames.jsonl` 与完整tensor快照直接有该数字码，后续只读门审计分别绑定来源，没有改写历史原件。

真实接管瞬间包含非零COM世界系速度与joint dq，不能用旧的零速度姿态reset冒充动态交接。只读核查见 [state_seed_audit/README.md](state_seed_audit/README.md)；当前只提取候选数据，未接入正在运行的PPO或授予新姿态资格。
