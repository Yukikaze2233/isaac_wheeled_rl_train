# V6 两侧整腿辨识回执

所有结果 `training_release=false`，不构成电机真实力矩标定。最终容量试验只使用未辨识的工程 prior。

## 当前可重复的比较

`final-heldout-recheck-v1/` 使用 final_r3 资产（manifest `9542c788…`）、1ms 物理步、固定既有六参数候选，对37/39/41/43/79/89六段分别跑名义模型与候选模型。左右共四个报告、四份原始命令和脚本均保留，当前脚本 SHA `bd7e66f6…`。这是同一录包内的完整留出段，不是新采集，也未重新搜索参数。

原始NPZ和TensorBoard event留在Kaiser `/home/kaiser/robot-rl-sim60/experiments/v6-pair-fit-20260929/final-heldout-recheck-v1/`；`summary.json`只汇总分段RMSE算术平均，不把大段按样本数重新加权。`feedback-ablation-v1/`另保存仅按实录sequence重复进行反馈保持的成对比较。

## 历史来源边界

- 最早 `left/right-baseline`、`left/right-fit` 的报告记录脚本 SHA `1b91cc55…`，但当时未在修改输入校验之前单独封存该脚本，当前存下来的对应前期版本 SHA 是 `4331bb67…`。这些早期优化记录原样保留，不宣称已追回完全一致的原始源码。现有候选作为固定数值输入，已在上述最终版本中独立复算全部留出段。
- `scripts/identify_pair_dynamics.py` 为 SHA `4331bb67…`，与 `fit-1ms`、`fit-quarter-ms`、`fit-torque` 报告完全匹配。
- `gain-ablation-source/identify_pair_dynamics.py` 为 SHA `fbd4dad6…`，与 `gain-ablation`、`gain-ablation-torque`、`final-asset-1ms` 报告完全匹配。gain候选因力矩输入回放恶化被拒绝，不能视为测出的力矩尺度。
- `diagnostic-fix-v1/` 保存修复audit几何getter后的脚本、24项测试结果及双侧81个静载段复算：仅内角读数约0.0005°变化，预测持力矩逐点不变。未覆盖旧曲线和旧静载文件。
- `assessment.json` 是较早的历史汇总；当前六段闭环数值以 `final-heldout-recheck-v1/summary.json` 为准。两者的资产封装版本与积分步长有明确记录，不能将文件更新日期当物理版本。

完整判断与未决机械事实见 `docs/V6_IMPLEMENTATION_PROGRESS_20260929.md` 和 `reports/v6_zero_pose_audit_20260929/README.md`。
