# Round4 最终策略自动评估

执行器：`scripts/round4/evaluate_policy.py`。它直接运行最终 ONNX 策略，不执行 PPO 训练，也不启动独立 GPU smoke。

```bash
python scripts/round4/evaluate_policy.py \
  --request /path/to/evaluation/request.json \
  --output /path/to/evaluation/reports
```

必须从**训练所用冻结仓库**执行，evaluator 本身必须包含在该仓库的 `snapshot.json` 中。输出目录必须不存在，父目录必须已存在。dispatcher owner 应在 freeze/commit 中包含此脚本，并将它作为正式训练成功后的 evaluator entry。

## 输入与启动门禁

- 验证 request、training plan、冻结 source snapshot 文件哈希及 source commit。
- 要求完成全部 30000 updates，`completion.json` 状态 `completed`，final/export verified，逐项核验全部 completion artifacts 和 ONNX sidecar。
- 验证 scratch provenance、实际合同及原旧结构 research 资产身份；运行时确认 7 body，禁止以尚未制作的新机构替代物理环境。
- 复用 `train_v40.preflight/make_env` 与 `play_v40_onnx.load_policy` 的契约、runtime、ONNX SHA 和签名检查；不依赖键盘或 GUI。
- 默认物理设备继承训练 command 的 `--device`，ONNX 使用 CPUExecutionProvider。仅显式 `--device cpu` 等覆盖会切换物理设备，并记录训练设备、评估设备和 override 标志，不能将其隐藏为同 backend 结果。
- 单环境评估的 B1 startup bucket/nominal mask 与训练 1024 环境不保证相同。完整保存本次 `check_round3_materials()` 报告，并由实际 wheel PhysX 系数与已验证 ground 系数计算 average-combine effective 摩擦；不能默认称作固定 μ=0.5。

## Request 恢复协议（dispatcher 需要同步）

Request `schema_version=2`：固定包含 8 个 standing/push case 加 4 个 tracking case，共 12 个。旧 v1/8-case request 被明确拒绝，不能静默省略 tracking。结果和哈希清单的 `schema_version` 仍为 1。

```json
{
  "deadline_s": 2,
  "steady_band_hold_s": 1,
  "height_error_band_m": 0.01,
  "planar_speed_band_m_s": 0.05,
  "tilt_band_deg": 10,
  "hold_must_finish_within_deadline": false
}
```

三项条件必须同时满足并连续保持 1 秒；**保持段的起点**距实际扰动不超过 2 秒，确认保持完成可以晚于 2 秒。CSV 按 100 Hz 离散样本判断；缺失、非有限、终止状态不能补成恢复。旧 `.005m/.02m_s/.5s` request 会被明确拒绝，不静默换口径。

## 运行矩阵与事件边界

- 高度：0.29、0.30、0.31、0.32 m。
- 每高度 standing：3 次独立 reset 的 20 秒 episode，共请求累计 60 秒，包含正常 timeout；**不是连续站立 60 秒**。报告保留实际观察时长，例如环境在 19.99 秒 timeout 时三次累计 59.97 秒。
- 每高度 single_push：20 个独立 episode，每个最长 20 秒。第 5 秒、下一次策略推理之前，调用一次 `env.apply_velocity_impulse(delta_xy, env_ids)`。
- Tracking：`tracking_vx_pos3`、`tracking_vx_neg3`、`tracking_wz_pos6`、`tracking_wz_neg6`，command 分别为 `[3,0,.32]`、`[-3,0,.32]`、`[0,6,.32]`、`[0,-6,.32]`；每 case 两次独立 20 秒 episode，不施加任何 push。
- 每个 episode 先 `set_evaluation_command`，然后 reset，关闭训练随机命令和随机 push。不改变训练合同的奖励、终止或 history 定义。
- Push 前后使用 fresh PhysX root backend 读取速度和 pose；记录 requested/realized delta、before/after COM 六维速度，验证 z/omega/pose 不变。随后调用同 tick `_get_observations()`，确认 actor history 不变、critic 线速度已刷新，再推理下一步动作。
- 终止/timeout 使用拥有独立存储的 **pre-reset** snapshot，不把 DirectRLEnv auto-reset 后的健康姿态当成恢复。

## 统计解释

Tracking 同时给出每个 episode 及 case 合并的 `full`、`steady_after_2s` 统计；后者严格按**每个 episode 自己的 `time_s > 2`** 取样，`t==2` 不计。vx 单位 m/s，wz 单位 rad/s：输出 command、实际速度均值/最小/最大、绝对误差 MAE、绝对误差 P95（排序后线性插值）。合并统计按样本汇总，P95 不是各 episode P95 的平均。失败/censored episode 的已观察样本保留；空稳态窗口返回 samples=0、误差=null，不能冒充零误差。原始 CSV 含实际 vx/wz 与 command vx/wz，JSON 还汇总 termination/接触 diagnostic 的样本计数与净力峰值。没有新设“达标”阈值。

逐 episode 记录实际施加、pre-push failure、恢复、failure、censoring 与原始 push 读回证据。施加失败/未验证不会冒充一次完成扰动。

- `disturbed_episodes`：确实通过 backend readback 验证的扰动次数。
- `pre_push_failed_episodes`：扰动前已 termination/nonfinite，不进入实际扰动分母。
- `recovered_within_2s_episodes`：满足上述保持条件，且整段 episode 未随后失败、未被截断；早期进入保持段后又失败会保留检测时间，但不计成功恢复。
- `failed_disturbed_episodes`：扰动后真实 termination/nonfinite，或完整 episode 内未达恢复条件。
- `censored_disturbed_episodes`：被中止/预算/非正常早期 timeout 截断、尚无确定失败结论的已扰动 episode，绝不计成功。
- 提供全部实际扰动分母下的失败率上下界（将 censored 分别视作未知成功/失败），以及 resolved trials 的 Wilson 95% 区间。

即使 20 次 0 失败，独立 Bernoulli 假设下一侧 95% 上界仍约 **13.9%**，不能宣称失败率低于 1%。固定 reset、固定扰动、同一材质的重复试验可能高度相关，置信区间仅是带假设的描述，不是认证。净接触力只作诊断，不能由阈值推断 ground-pair 接触比例。

## 输出与中止

- `result.json`：执行状态、每个 case 的状态/metrics/episode 明细、request/ONNX/contract/source commit 身份和本次物理材料报告。
- `evaluation_report.md`：可读统计与口径说明。
- 每个 case 的 `telemetry.csv`、`summary.json`；summary 包含逐次 push before/after/realized 证据。CSV 浮点列使用 9 位有效数字，可 round-trip float32 物理状态；派生 double 指标同样按此精度记录。
- `preflight.json`：实际 runtime 启动检查。
- **最后发布** `report_manifest.json`，包含 request SHA 及所有报告文件相对路径、size、SHA256。哈希清单不自包含自身。

`status=completed` / `evaluation_success=true` 仅表示矩阵和报告实际完成，失败 trial 仍原样列出；`policy_quality_verified` 始终为 false，不以 reward 门槛或零失败自动认证。

默认内部预算 8700 秒，包括验证/正式启动，留出 dispatcher 外层 9000 秒 timeout 的收尾余量。SIGINT/SIGTERM、内部超时或异常保留已完成和部分 CSV，未执行 case 明确标为 failed/unstarted；失败报告先于可能较慢的 Kit cleanup 发布。SIGKILL/主机断电无法可靠收尾：没有有效最终 manifest 就不能视为评估成功。不会修改训练的 completion 或任何训练产物。若 final 导出失败，dispatcher/launch owner 应记录 blocked，不能启动此 evaluator 后假装成功。

新增 tracking 共 160 sim 秒，完整请求从 1840 增为 **2000 sim 秒**。按每秒 60 个 policy step 估算，200000 个 100Hz policy tick 对应约 3333 秒纯 rollout，但不保证实际 wall time；仍使用内部 8700 秒/外层 9000 秒预算，预算不足保留部分统计并将未完成 case 标为 failed/censored/unstarted，不丢 case。回收端的固定 case-ID/count 校验也必须同步为这 12 个 case。
