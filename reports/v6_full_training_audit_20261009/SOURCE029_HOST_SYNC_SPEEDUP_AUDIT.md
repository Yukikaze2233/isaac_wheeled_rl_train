# source029 热路径与日志计时审查

2026-10-09；只读基线 `e5bd9c1367b875d7f229b9a1930647013d023372`，源码来自 `.worktrees/serial-dispatch-source028-20261009`。本文的行号均指这个冻结版本，不能用尚有改动的主工作区行号代替。未运行 SSH、仿真、性能 benchmark 或 PPO。

## 1. 实测耗时与归因边界

现存 `source029_tb001.json` 是 speed/attempt_003 的只读 TB 快照。选 TB step 41410–41429，20 个更新：

| 每更新宿主墙钟 | 均值 / 秒 | 含义 |
| --- | ---: | --- |
| iteration | 23.6026 | 更新间的总墙钟 |
| env_step | 18.2999 | 24 个策略步，包括物理推进、奖励、终止、命令、reset 及其等待 |
| diagnostics | 0.7277 | audit observe/capture/drain 等已包裹部分 |
| logger | 0.4641 | RSL 原生 logger.log |
| optimizer | 0.1263 | 已包裹的 PPO update |
| returns | 0.0100 | 已包裹的 returns 计算 |
| progress_io | 0.0152 | 进度落盘，含 fsync |
| checkpoint | 0 | 本 20 更新不含保存，不能外推为长期零开销 |
| unattributed | 3.9594 | 总计时减已归属计时；尚未逐项定位 |

约 77.5% 在 env_step，16.8% 未归属。`TrainingTiming` 是 **exclusive host wall、无 CUDA synchronize**，所以这些不是 GPU kernel 时长。等待先前异步 GPU 工作的时间可能落在随后第一次 CPU 取值处；不能把一条 scalar 转换的全部墙钟当作它自身的计算成本。

快照暴露 **31,811 个 scalar tag**，这是 run 内可见 tag 总数，不证明每次更新都写 31,811 项。源码确实在每次更新逐个写 `LearningAudit` 的已转换 Python 值；该循环落在 `diagnostics` 和 `logger` 两个 scope 之外。序列化、event queue/backpressure 是可核查候选，但**不能把 3.9594 秒全部认定为它**：actor inference、rollout/storage、其他未包裹写入也在这个余额中。

## 2. 逐策略步的明确同步与逐物理步的工作

每更新 24 策略步，每策略步 4 物理子步，共 96 次物理推进。当前已是 5 ms 物理与反馈 PD、20 ms 策略；SCUT 已核验的三个保存配置也都是 dt=.005、decimation=4、4096 env、24 steps/update。再次改成“200 Hz”不会产生新的收益。

| 冻结代码位置 | 执行节拍 | 明确行为 / 可优化性 |
| --- | --- | --- |
| `env.py:1462–1559` | 每物理子步 | 新反馈 PD、传输/响应、气簧、PhysX、传感与接触更新；该段直接代码没有 `bool(CUDA tensor)`、`.item()`、`.cpu()`、`.tolist()`。不能据此认定 native API 内部绝无同步，但不能编造“每子步六次标量同步”。 |
| `env.py:1549–1557` | 每物理子步 | 接触矩阵 reduce、轮接触峰值；有跳跃任务时每子步 COM 高度/竖速 reduce。均为设备张量计算，不是 CPU 取值；COM 与接触时序参与物理事件，不宜降到策略频率。 |
| `env.py:1573` | 每策略步 | 六次 `bool(torch.isfinite(t).all())`，正常有限路径六次标量取回，可合并为一次、保留原检查点。 |
| `env.py:1576` | 每策略步 | `float(gap.max())` 取回闭链峰值；后接 3 mm fail-fast。与有限性检查有先后关系，本次首选方案不移动或降频。 |
| `env.py:1886` | 每策略步 | spring reserve 计数一次整数取回。仅诊断，但移动会改变异常前计数记录，旧候选刻意保留。 |
| `env.py:1930–1933` | 每策略步 | 每个 reason 的 `int(mask.sum())`，另 success 与 timeout 两次。可一次 int64 stack→CPU，保留 reason 独立计数及插入顺序。 |
| `env.py:1911,2073` | 每策略步 / 恢复启用 | prelude/reset 的 `nonzero()` 生成动态长度 CUDA 索引；可能同步。改变 reset/select 顺序可能改变随机数消费与事件语义，不列为第一批。 |
| `env.py:2082–2102` | 每策略步 | `if change.any()`、`if change_height.any()`、`if pushed.any()` 各触发标量条件；事件发生还会 nonzero / sum。可以研究合并布尔取回，但应保留三条分支原次序、随机调用和异常点。尚无等价证明。 |
| `env.py:963,1591` | 普通爬阶启用，每策略步 | 每类 body support contact view 的容量检查 `bool((counts>32).any())`；不是每物理步。容量保护必须保留，本轮 speed 不应默认按 terrain 后的 view 数估算。 |
| `learning_audit.py:169,178,188` | 每更新 capture/drain | 固定高度子组 `ids[bool_mask]` 是动态形状选择；随后每组 advantage/reward/stop/window 归约，产生许多小 GPU 算子。 |
| `learning_audit.py:212` | 每更新 drain | **已经**一次 stack→CPU→list，不是每 scalar 一次 GPU 回传；不要再建议给这里做同样“批量化”。 |
| `train_chassis.py:969–972` | 每更新 | drain 已计 diagnostics；随后的 Python `add_scalar` 循环没计时，待用独立 `audit_logger` 补足归因。 |

两个容易误报的点：`SubstepControlAudit.end_policy()` 仅处理 Python 计数器，不会把 GPU 状态逐步搬到 CPU。`env.py:1246` 的旧逐环境 `ids.tolist()`/随机 sign 循环，在当前 `skills` 路径由 `1211–1220` 提前 return 绕过，不能拿旧路径说明当前每次 reset 的瓶颈。启动读取 `.cpu().tolist()` 也不能摊进持续每步。

## 3. 首选最小改动：复用已存两处 host-sync 候选

现存候选位于 `../v6_scut_posture_completion_20261006/source_repair_027/performance_followup019/host_sync_candidate001/`，状态明确为 `accepted_local_candidate_not_deployed`。e5bd9 的这两处仍为旧实现：

1. 六个有限性 tensor reduce 后 `torch.stack(...).all()`，一次 `bool`；抛出位置与文本保持。
2. reasons、success、timeout 的 int64 reduce stack 后一次 `.cpu().tolist()`，按旧顺序加到 Python 整数；独立的重叠 reason 不合并。

若当前 reason 数为 R，正常每策略步的这两块 CPU 取回次数由 `6+(R+2)` 变为 `1+1`，少 `R+6`；每更新少 `24×(R+6)`。这只是静态取回次数，**不是测得的提速倍数或秒数**。旧候选有 63 项 CPU 通过及独立审查，覆盖有限/NaN/Inf、空/stride、整数大计数、顺序、PD 输出/RNG保持和其余 step AST。仍没有 native GPU profile 或 PhysX 轨迹等价结果。

最小验证应先把仅两块移植到 **实际 e5bd9** 并确认其余 AST、合同、RNG/PD/终止/奖励与报告频率保持，然后在将来的受控边界用同输入的原生运行比较输出和墙钟。本审查没有执行此测试。坏 runtime 下有已声明差异：有限性原来短路，候选会求值全部六项；后续 reduction/transfer 本身异常时，逐项版可能已更新前几个计数，而批量版尚未更新。

## 4. 已单独实现的计时补丁与下一步边界

独立 worktree `.worktrees/audit-logger-timing-20261009` 基于同 e5bd9，仅 `scripts/train_chassis.py` 6 行 diff：把原 loop 原封不动放入局部 `log_audit()`，用已有 `timed_call(timing, "audit_logger", log_audit)` 包裹。原 tag、值、step、顺序和频率不变；动态注册 `Timing/audit_logger_seconds` 与 `_calls`。没有新 logger、CUDA synchronize、训练超参或 runtime 修改。这是归因补丁，**本身不是提速**；event writer 后台的异步写盘并不全被这个 scope 捕获。

已有 `tests/test_training_timing.py`：正确设置 `PYTHONPATH=src` 后 3 passed，0.04 s，diff check 通过。首次新 worktree 收集因未设置 PYTHONPATH 报 `ModuleNotFoundError`，未执行测试，保留在回执。未部署、未 benchmark、未改主工作区源码。

建议次序：先采用有局部等价依据的两块同步合并，并在同次受控续训带上计时补丁；据新 `audit_logger` 占比决定是否用原 TensorBoard writer 支持的批量 Summary 写入，保持 tag/value/step/frequency。需要以后证明完整 event 内容等价、异常传播、flush 和后台 queue 行为；本次没有编写或部署批量 logger。减少记录频率会损失现有逐更新审计分辨率，不能冒称日志完全等价。

不建议与这批混在一起：减 solver64/32、放宽3 mm闭链、200→100 Hz、子步重复旧 torque/旧 feedback、降低接触/恢复/延迟/响应采样频率。这些改变物理或控制/事件语义，须单独机械与行为验证。现有 native COM 静态缓存已在 e5bd9 的 `root_kinematics.py`/`env.py:410` 接入，不能再当作尚未实施的收益。也没有证据把当前瓶颈归因为多个 PPO 进程或线程争抢，不推荐泛化地加线程。

证据：`source029_tb001.json`；`scut_saved_config_extract.json`；旧候选 `candidate_report001.json` / `static_review001.json` / `root_candidate_review001.json`；本次补丁与验证回执 `source029_audit_logger_timing.patch` / `source029_audit_logger_timing_receipt.json`。当前运行维持冻结生产代码；本文未授予任何机械或实车资格。

## 5. 后续已完成的局部验证

本段追加证据，保留上文原审查时的限制。Kaiser实际4090／torch2.11+cu128上，冻结e5基线与e16候选的真实AST有限检查／计数块57例通过，包含空/非连续张量、NaN/Inf、4096/20000个mask和大Python整数。32轮交错局部计时中位旧5.223ms／新1.759ms，peak tensor allocation665600B；无远端文件写入、无原生仿真或PPO。只支持这些报告块的局部等价和耗时，不支持PhysX轨迹或整更新三倍提速。见source029_gpu_blocks001.json。

原状态发布每0.25s重选约190个组，并逐组取回origins/commands/episode_step。候选b752a395保持完整JSON、所有组/原顺序、dtype与int64、频率和原子replace，将四字段各批量取回一次，新增独占state_publish计时。18CPU原检查及组合148CPU已过；这仍没有完整更新提速测量。schema/字段完全等价不意味着异步GPU等待必然减少固定秒数。

首100实际speed actor窗口41403–41502的wall23.791s、env18.516s、unattributed3.939s与原早期读数相符。实际 source029还未采用上述候选。完整正式合同、正常边界控制和原生启动仍需完成；不热改当前worker，也不拿来源摘要或新计时项当训练质量通过。
