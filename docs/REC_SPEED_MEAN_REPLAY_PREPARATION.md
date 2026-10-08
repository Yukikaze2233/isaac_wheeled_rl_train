# REC 父模型与 原生speed597 均值策略对照：CPU 准备候选

这是独立、未评分的同条件回放准备器；`run_scope=interactive_unscored`、`qualification=null`。它只认证输入、验证独立测试设计和控制协议、准备事件统计，不运行 Isaac、GPU、PPO，不产生生产准入或正式 baseline 成绩。下面记录原 11-case CPU 准备接口；新独立 [7场景原生驱动器](REC_SPEED_MEAN_NATIVE_DRIVER.md) 已实现可执行代码路径并完成 CPU 事件/构造边界验证，实际 Isaac 物理运行仍未执行。

## 输入与来源

运行库必须是未经修改的真实冻结 `e5bd9c1367b875d7f229b9a1930647013d023372`；通过 `--runtime-root` 指向该版本的独立 checkout，不导入主树可能被修改的 GUI/preflight。工具验证该 checkout 的 HEAD 与生产目录 tracked diff。原 REC21 合同 SHA 为 `eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2`，speed26 为 `941fb9e1f7a8ced3aa83fca76e82b7cf3064b594602d1e2900e56e0df0cc6b64`。

实际父 PT 是主项目 `reports/v6_full_training_audit_20261009/source029_parent_rec41393/model_final.pt`，SHA `e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967`，消费 41393、REC actor 14188、学习谱系 27646；已有下载回执 `source029_parent_download001.json` 与原生 CPU 全状态认证 `source029_parent_verification001.json`。父已由原官方CPU导出器导出到主项目 `models/v6_recovery_candidate_41393_20261009/policy.onnx`，25原ABI向量最大数值误差4.917383193969727e-7，qualified=null；交付manifest保留原PT/合同/接口来源。不猜测尚未提供的原生speed597文件/SHA。

每个输入提供 PT、其不可变 `.checkpoint.json`、原 `.contract.json`、保存的 `agent_config.json`，以及 ONNX 和原 `.onnx.json/.onnx.contract.json`。复用原 `export_chassis_candidate.authenticate`、原 `validated_session`，核验完整模型/合同/封存/资产/控制身份；另以原生 RSL 均值 actor 和相同 ABI 向量重算 ONNX 最大误差（≤1e−5）。每份原生产合同都按其完整 `full` 角色认证，预算、响应证据和来源规则照常执行。共享 ABI 显式比较 35D/81D/6、单帧、50 Hz 策略/200 Hz 反馈 PD、动作顺序、控制、资产、参考命令、命令斜坡和架构；不要求两个生产合同整体字节相同。

默认仅允许准确 REC41393 父与真实原生 periodic sealed speed task actor597；同时核验包含10次critic的消费42000、actor消费41820、stage actor28153、谱系28253，以及源v26 integrated lineage的origin=e04c…3967、joint/domain actor597。原冻结 train_chassis 按累计消费500倍数保存；evaluation_interval=0不在actor500额外封存，消费41903不可用。首个≥500 actor原生周期点为42000/597，端点选择及原代码/合同SHA写入回执；旧500候选回执保留，不改PPO保存规则、不要求临时保存。`--interface-smoke` 只允许两份现有 REC21 同合同导出，标为接口验证；若两者来自同一个旧 PT，就不能作为父子能力/遗忘对照。

## 固定协议与独立投影

共 11 个 case，每项 4 个固定出生副本：230/305/430 mm 静站（原 10 s），左右侧倒与正反俯卧四个原始几何倒地链（原 20 s），正反 0.5 m/s 和正反 1 rad/s 的四个制动 case。倒地组保留原 case/pose，不用混合 L0/L1/L2 命名推断真正倒地分母。模型顺序不同也必须采用相同原生 reset、seed、35D 观测与随机化配置；本次尚无场景读回或行为结果。

REC 控制器取已认证 **REC21 train** 行：侧摆 1.5 rad/s、`complete_started_blend=true`、200 ms 混合、持续接稳 1 s、脚本上限 8 s，原 wheel-balance 和 v2 profiles。exact-case 模式关闭 reset 课程抽样，保留四个真实原始倒地出生。`verify_common_controller` 必须在未来驱动器中对实际 FSM 实例调用：复用原 `native_runtime_binding`，以真实 REC21 train 配置认证原类定义、源码 SHA、速度和陀螺门，并核验 blend/stable/script 设置；当前 CPU 测试只构建真实原生 FSM，不推进其状态或仿真。

旧生产 `full` 角色明确要求侧摆 6 rad/s，与真实训练协议不同；不修改它。未评分投影单独移除 `task_campaign.revision_migration`，明确记录原因与完整 before/after diff，不伪造其他版本标记。原 full 完整认证仍保留。

投影调用已有 `validate_design_contract(role='baseline')` 验证独立 SHA 期望表，并调用原响应证据、恢复配置、台阶配置校验。`baseline` 只是该验证器已有的内部角色名；不是正式 baseline 分数。**没有对投影调用生产 `design_preflight`/预算或 train_chassis 准入**；原 full 预算回执不能被解释成新投影可部署的预算。本候选明确记录调用覆盖/未覆盖，未来独立物理驱动器的 bootstrap 与 runtime readback 仍待确认，不能将产物直接交给生产训练入口。

制动请求：0–1 s 真零请求，1–5 s 固定运动目标；从 5 s 起请求立刻变为 `[0,0,0.305]`，继续使用已有 `advance_playback_command` 的原合同斜坡，因此实际命令会稍后才归零。17 s 的总窗口覆盖实际命令归零后连续 10 s 安静期；**该安静期按请求与实际命令定义，不按机器人已经停止定义**。后续速度/漂移均值必须用这个固定窗口的现有 `EpisodeMetrics`，不能筛选只停住的帧。

`ReplayEvents` 接受 pre-reset 的 CPU diagnostics、实际完整策略区间的 `extras.policy_learning_mask` 和步末瞬时 ownership。瞬时 first-pure 与整段 policy-owned 曝光分开；terminal 帧计一次，后续保持 done 帧排除，未终局副本标 right-censored。复用 `RequestProtocolMetrics` 的原始终局恢复账本，每组终局分母只输出一次，复用原 `recovery_episode_diagnostics`；不把复习抽样率、零力矩或单次好副本当完整恢复成功。

## 可执行 CPU 命令

以下是两份真实旧 REC41374 ONNX 的**接口验证**，不是实际 REC41393→原生speed597 比较。输出目录须不存在。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/yukikaze/isaacsim60-venv/bin/python -B scripts/prepare_rec_speed_mean_replay.py \
  --runtime-root /home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009 \
  --candidate /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/rec41374_source/model_final.pt /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_41374_20261009/policy.onnx \
  --candidate /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/rec41374_source/model_final.pt /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_priority_handoff_20261009_r2/model/policy.onnx \
  --interface-smoke --output /tmp/rec-speed-mean-interface-new
```

得到 `preparation.json`、`interactive_unscored_contract.json`、独立期望表和 `projection_changes.json`；回执包含脚本/合同/SHA、原 full 认证、共享 ABI、确切控制/随机化配置、覆盖与 pending。实际两个新模型具备后用相同两个 `--candidate PT ONNX` 参数去掉 `--interface-smoke`；任何缺失输入、错误父/actor 时钟、原始合同/封存或数值不符都会拒绝。此命令仍只做 CPU 准备。

验证只运行 `tests/test_rec_speed_mean_replay.py`：literal-zero 与实际斜坡分开、final-mask 与瞬时接管分开、终局去重/真实组分母、安静连续 10 s 与右删失、ABI 负例、真实 e5 独立投影与真实原生 CPU FSM 身份/错误速度负例。实际 ONNX 双输入 CLI 接口检查另存机器可读回执。物理均值表现与是否遗忘仍未验证。
