# 四 case 原生物理前缀诊断入口

状态：`implemented_cpu_verified / physical_not_run`。这是独立诊断入口，零 PPO 更新，不提供自起资格，也没有更改正在运行的 v14 训练。原 `PHYSICAL_PREFIX_PLAN.md` 与 `physical_prefix_design.json` 保留为当时的未实施设计记录；本文件记录实现边界。

入口为 `scripts/probe_recovery_physical_prefix.py`，CPU 支持模块为 `scripts/recovery_prefix_support.py`。默认并固定四个 lane，每个已有 case 只计算首次出生，20 秒上限：

- `recovery_prone_belly_legs_above_stand`
- `recovery_prone_back_legs_above_stand`
- `recovery_left_side_90_stand`
- `recovery_right_side_90_stand`

```bash
/home/yukikaze/isaacsim60-venv/bin/python scripts/probe_recovery_physical_prefix.py \
  --contract contracts/v6_task_campaign_recovery_worker_v8.json \
  --checkpoint models/v6_model_delivery_20261003T2235/recommended_flat/model_final.pt \
  --output /tmp/v6-recovery-prefix-preflight --preflight-only
```

`--preflight-only` 不导入 `AppLauncher` 或 `FixedCaseEnv`，也不创建物理场景。省略该选项才会启动一次 headless 本机 GPU 诊断；本次实现任务没有执行这一步。输出目录必须不存在，避免覆盖已有证据。

## 输入身份和原生入口

明确使用 V8 recovery worker（SHA `f6aa77c6808c9746dc8dee431e8b3331b6b0f883db701335e57069e7df738ee8`）和 V8 campaign（SHA `c77926685265ebc7700820abdb2230bfb8fb4d9ec8c6921c6e77187e29d97abb`）。使用原生 campaign 的 required/monitor 选择，将 322 项总套件选择为 63 项 recovery full-role，再运行独立设计预检认证原 252 lane invocation；随后分别投影 runtime 合同和独立 design 为四项、每项一次、N=4。不能把 322 项原 worker 或旧 1 kHz 默认计划直接当这个角色。

复用 `probe_v6_recovery_runtime.project_evaluation/project_scenes`、`fixed_suite_contract`、`train_chassis.preflight` 与独立 `design_preflight`。物理入口复用 `FixedCaseEnv`、原生 `RecoveryTraining/RecoveryFSM`、`probe_recovery_takeover.load_actor/IntervalAudit` 和原生 `env.step` 的 outcome/autoreset。没有自写第二套闭环、相位机或成功门。

full-role 保持 200 Hz 新反馈/PD、50 Hz 策略、35D/81D/6D ABI、`signal_delay:false`、`command_transport:true`；不随机打开训练时的整帧延迟条件。独立 delay-role 仍是后续任务，不属于本入口结论。

资产、控制与物理核心模块绑定冻结提交 `b4ce80964d47c0127e0b3d9929ad35a620bbf204`。现有用户修改的 `preflight.py` 没有被覆盖，其实际 SHA 单独写回执；全部 chassis 模块和两个复用探针的实际 SHA 也在执行前后比较。任何运行期间文件变更会令诊断失败。

全 PT 固定为平地终点 SHA `7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f`，对应 source contract SHA `666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7`。CPU 认证完整 actor、critic、Adam、infos、逐文件 seal、资产、控制和观察/动作 ABI。认证的学习消费为 **14020**，平地 task actor 更新为 **13000**，两者分开保留。原生严格 actor 工厂只用于确定性推理，不调用 learn/update/compute_returns；执行后再次比对 actor 张量和源 PT SHA。

## 出生、物理前缀和终止捕获

`FixedCaseEnv.reset_suite()` 是唯一被计数的出生事务，允许父类与子类各自完成原生初始化。事务结束核对全部 18 个实际关节位置、root pose、固定 seeded root-z jitter 与认证几何库。这里明确保留 `geometry_valid:true / dynamic_recovery_validated:false`，出生摆位不算已完成物理自起。

随后拦截原生使用的四个 state-write API：root link pose、root COM velocity、joint position、joint velocity。仍存活的首批 lane 在出生后调用这些 API 会在实际写入前抛错；不得通过设置姿态、速度、关节或相位来制造接管入口。原生终止调用 `reset` 前先捕获该 lane 的物理状态和控制历史，再关闭首批 cohort 的记录，随后允许原生 autoreset。后续 episode 不再进入分母或成功计数。

`sim.step` 后紧随的 `robot.update` 才推进物理曝光时钟；出生时的仅数据刷新不计物理步。前缀使用真实控制与物理执行，记录每个实际 FSM 转换，以及释放结束、PREPARE→BLEND、第一混合力矩落地后的状态、最后混合子步、首个纯 RL 子步、sensor dwell、truth dwell、motion unlock 和 followup。BLEND 实际应用必须有 40 个 5ms 子步，控制混合比例为 0 至 0.975，下一子步才允许原生 pure RL 比例 1；没有进入相位时不伪造快照。

快照是事件当时全 batch 的 clone；事件索引中的 lane 指出对应 cohort。首个 pure RL 是上游脚本/actor 混合的控制归属，不能把仍留在 transport/response 中的历史称为已清空。物理后快照在接触读取、延迟反馈追加之前，terminal 快照则在原生 outcome 后、autoreset 前；这两个采样位置在文件中明确区分。

## 实际覆盖和未覆盖

原生可读字段按实际属性记录，缺失字段写入 `snapshot_field_missing`：

- 实际 18DOF 位置/速度；root xyzw pose、线速度、角速度；已施加力矩、闭链 gap 和两轮位置。
- 实际 actor 消费的观测和请求动作、动作历史、命令、当前 PD 输入与 actor PD 力矩；原生混合后力矩和 transport/response 各环节输入/输出，以及最终 `env.torque`。
- FSM 相位、route、timer、目标、各接稳门、orbit/thrust/side 内部参考；连续编码器 winding 与上次原始角、IMU 上次速度、wheel support probe 历史。
- 执行器 response state；command transport 队列、held、延迟/丢包状态、CAN phase、RNG 等可读字段；启用模块的实际存在性。full-role signal delay 不存在，明确记录 absent，而非伪造延迟 buffer。

不能保存或复原 PhysX 内部接触、solver warm-start 等状态，因此 `event_snapshots.pt` **不是精确恢复检查点**。原生 `torque()` 内未导出的独立 script request/capped 局部量没有被猜算；保留真实输出、PD 输入、力矩上限和混合比例。拦截范围仅是冻结原生代码实际使用的四种 reset API，不声称能阻止任意外部 PhysX view 写入。初始化场景构造在 wrapper 安装前完成；单独计数的 `reset_suite` 事务和之后的执行有写保护。

内部合法未初始化 history 的 infinity 保留在 PT 快照中，JSON 预览显式转换为 null；必需物理状态中 NaN/Inf 直接失败，不能靠 JSON 转换掩盖。

## 固定出生分母和输出

`cohort_summary.json` 固定 N=4，关键 milestone 未发生时仍输出 **0/4**。最终 attribution 使用原生 `diagnostics.success/terminated` 和 `extras.time_outs`；机械 failure 优先，其次 native success，其次 boundary truncation，最后 timeout。保留所有原始重叠 reason flags，五分区合计为 N。原生 20 秒 horizon 的 done 为 timeout；探针提前异常停止而尚未 done 的个体为 right-censored，记录实际 `censor_time_s`。异常/finally 同样刷新实际 policy、physics 步数和曝光秒数，不能把请求的 20 秒或上一次每 50 tick 进度当实际曝光。

输出包括 `effective_contract.json`、独立 `engineering_expectations.json`、`report.json`、`events.jsonl`、200Hz `physical_frames.jsonl`、50Hz `policy_frames.jsonl`、`event_snapshots.pt` 与 `cohort_summary.json`。主指标始终为 reached/N 和 success/N；不以 success/ended 或只计算 handed-over 样本冒充整链完成率。快照文件可有诊断 I/O 开销，该入口只有四个 lane，未加入任何常驻训练路径。

## 已执行验证

CPU preflight 实际通过；新增测试 17 项与原生 FSM、wrapper、terminal ownership、rollout mask 相关集合合计 **111 项通过**。覆盖严格 worker SHA/独立设计、全 PT ABI/消费钟、N=4 首批分母、写入前拒绝、autoreset 前 18DOF 捕获、真实原生 FSM BLEND 40 子步、缺失事件 0/N、timeout 与右删失、非有限物理状态失败、preflight 禁止导入 AppLauncher。CPU fixture 的物理 buffer 只验证捕获顺序，不证明物理自起成功；真实四 case 场景尚未运行。
