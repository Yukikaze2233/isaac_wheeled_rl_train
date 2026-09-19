# Round4 最终模型导出恢复

## 原因与算术边界

Round4 已完成累计 30000 次 PPO 更新，最终 checkpoint 已保存；原自动流程在
CPU FP32 Torch/ORT 合成数值一致性检查失败，因此原 `completion.json` 为
`export_failed`。PPO 完成不等于导出成功，更不等于独立评估通过。

默认导出仍采用 FP32 和原来的 9 个输入、`atol=1e-6, rtol=1e-5`。
显式 `--precision float64-internal` 将原 FP32 参数精确提升为 FP64 做内部计算，
输入输出仍为 FP32。ORT CPU 无 double ELU kernel，导出使用标准
`Where/Exp/ClampMax` 实现 ELU；独立参考保留 PyTorch 原生 FP64 ELU。
校验扩展为 4105 个确定性输入，容差不变。每个隐藏层均处理，包括共享 ELU 实例。

**这不是与原 FP32 运算逐位或原容差完全等价的证明。** Sidecar 分别记录
FP64 参考验收与原 FP32 对照失败数量、前 9 个输入中的失败索引、最大绝对差。
原权重文件不改写。控制效果必须通过闭环评估另行确认。

## 独立恢复流程

在以原训练提交为基线的干净 Git checkout 中执行；恢复目录必须不存在：

```bash
python scripts/round4/export_recovery.py prepare \
  --plan /absolute/original/audit/plan.json \
  --output /absolute/new-recovery-directory
python scripts/round4/export_recovery.py evaluate \
  --request /absolute/new-recovery-directory/request.json \
  --output /absolute/new-recovery-directory/reports
```

准备导出在独立 CPU 进程中完成，仿真启动前和结束后重新检查：

- 原训练 snapshot、全部 completion 工件哈希、计划与 resume 来源、累计更新数；
- 原失败必须是全部请求更新完成后的 export-only failure；
- 修复源码必须是干净提交，物理环境、资产和训练入口与冻结源码逐文件相同；
- 恢复 ONNX、sidecar、原 manifest 副本及 `recovery.json` 的 SHA；
- 合同、7-body 资产、ground 与原计划一致；
- 原 12-case、20 秒 episode、100 Hz 及推扰恢复判据沿用原评估器。

原训练收据保持 `export_failed`，不生成冒充训练成功的 `completion.json`。
新 `recovery.json` 明确记录训练提交、修复提交、原 checkpoint/收据 SHA、算术模式。
普通训练完成门禁不接受这种恢复路径；只有显式恢复入口能够调用它。
执行完成状态与 `policy_quality_verified` 分开，后者不会自动置为 true。

## 本地验证

- 导出专项：126 passed。
- `tests/v40` 加 Round4 评估和恢复专项：953 passed、1 failed。
- 唯一失败为未修改的 `test_mujoco_static_parity_without_stepping`：本机
  MuJoCo 3.8.0 的网格距离为零；测试注明 CI 使用 3.12.0。没有放宽该物理门槛。

此处记录实现验证；正式 final 模型导出与物理评估结果另行归档。
