# v14 日志性能只读评审

冻结源码 `b4ce80964d47` 与 Kaiser 实际项目文件 SHA 一致，已安装 RSL-RL 为 5.5.1。日志存在可优化路径；现有证据只能定位算子和计时范围，不能把全部秒级开销归于日志。

原生 event 的 4 MiB 有界尾部经 TFRecord CRC 校验，排除截断首尾更新及 `/time` 的独立秒时钟后，完整更新 **24023 有 14,197 个唯一 scalar tags**，其中 LearningAudit 13,764、六个 release_stop 组 792（含相位姿态 144）、三成本的分组密度 432，另有三个全局 reward tags。新增字段加重既有大日志；release 组与成本密度存在交集，不能简单相加当作新增总量。这是标签计数，不是性能实验。

| 路径与源行 | 实际操作 | 当前计时归属 |
|---|---|---|
| `learning_audit.py:211–212` | 全部结果一次 `stack().cpu().tolist()` | `diagnostics`；没有逐 tag 设备取值 |
| `train_chassis.py:729–733` | drain 后逐 tag 写 CPU 浮点 | 写入循环在 `Timing/logger` 之外，落入 unattributed |
| 已安装 `rsl_rl/utils/logger.py:185–203` | 每 key 拼接最多 24 次；GPU mean；writer GPU scalar；GPU scalar 字符串格式化 | `Timing/logger`，wrapper 位于 `train_chassis.py:887` |
| 已安装 `logger.py:129–147`，runner `on_policy_runner.py:98` | 每策略步追加 extras；`done.nonzero()`；reward/length 两次 `.cpu().numpy().tolist()` | 项目现有 Timing scopes 之外，落入 unattributed |
| `episode_metrics.py:162–218`，`train_chassis.py:787–798` | 每 10 更新逐组 GPU reduction、host 取值、Behavior 写入及 JSON | `diagnostics`/`logger` 之外，周期性落入 unattributed |
| `env.py:1707–1710,1799–1851,1862–1864` | 逐项 host 计数、reset 前诊断快照、auto-reset | `env_step` |
| `skill_commands.py:570–572` | 每策略步为 135 组重建静态 sequence length | `env_step`；每更新 24×135=3240 次索引赋值 |

逐 tag 同步的源码链明确：PyTorch TensorBoard `summary.py:382` 调用 `make_np`，`_convert_np.py:35` 执行 `detach().cpu().numpy()`；Tensor 字符串格式化在 `_tensor.py:1150` 调用 `detach().item()`。LearningAudit 已先转为 CPU 浮点，因此它的写入循环主要是 Python/protobuf/event 工作，不能套用这条 GPU 同步归因。

现阶段参考中位数约 wall 14.66／env 11.43／optimizer 0.11／logger 0.22／unattributed 2.43 秒。`TrainingTiming`（`training_timing.py:1–5,61–68`）是无 CUDA synchronize 的 exclusive host wall 计时；异步 GPU 等待可能在后续 host 取值处兑现。unattributed 还含 policy act、算法 step 处理、NaN 检查等。标签计数及上述同步调用不能解释或承诺消除全部 2.43 秒；各分项中位数也不能直接相加分摊。原 logger 0.22 秒单独约占整轮 1.5%。

建议作为未来独立性能版本：先补 RSL `process_env_step`、项目 scalar 写入、Behavior report 的 host 计时；再将 extras 改为每 key 一次拼接、所有 GPU 均值一次 CPU 拷贝，同一 CPU 值同时用于 TensorBoard 和 console。保留缺 key／向量／标量的原始元素加权分母、tag 名、step 和每更新完整输出。静态 sequence length 可初始化缓存并在诊断返回时 clone，动态数据继续在 reset 前快照。批量 Summary event 写入需另验 CRC reader 兼容性；`add_scalars` 会改变 writer/tag 结构，不适合作为等价替换。

未来验证需比较完整 tag/value/step、episode 顺序及 deque 截断，再用相同合同／资产／状态／人口的原生整轮 wall 分布确认收益。微基准不能代表整轮倍速。本次仅整理已有只读证据，未修改运行源码／合同，未做 GPU 实验或重启；`qualification=false`。

机器回执：`learning_audit/logging_source/logging_source_audit.json`（算子与建议）、`remote_source_receipt.json`（实际版本及 SHA）、`additional_source_receipt.json`（PyTorch 来源 SHA 与有界原生计数）。vendor 源码副本不纳入提交。
