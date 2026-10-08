# 2026-10-09 模型交接补充：恢复与高速制动

本页追加到已有交付文档，旧交付包和 ONNX 不改写。本次同源原生短回放显示，**如果优先倒地恢复与高速制动，REC38000 更适合作为目前两候选中的保守回放选择**。最新 REC41374 在加速段更接近速度目标，但制动与姿态明显较差；不能将“最新”表述为“综合最强”。

| 交接用途 | 原模型位置与身份 |
|---|---|
| 优先回放候选 REC38000 | [policy.onnx](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_38000_20261008/policy.onnx)；SHA `cd9c9be93ed65e56d9d4835ae9057490d271b4ad2e41cdf429778f1cf605b2ad` |
| 最新研究候选 REC41374 | [policy.onnx](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/models/v6_recovery_candidate_41374_20261009/policy.onnx)；SHA `63574953c97aad374fa1554c53486cd540a3019857af72e0c06cc010214299bf` |
| 原部署接口／控制映射 | 使用各模型目录中的 `deployment/interface.json`、`deployment/manifest.json`、`policy.onnx.contract.json` 和原 README；本页没有新的轴映射、观测布局或增益修改 |

两候选同源合同 SHA 为 `eaaf0a2a8fdb502c31c0a3aca5c4b60d1796bd6861c17736e0b35e41daaccda2`，35D 单帧输入、6D 均值输出，50 Hz 策略／200 Hz 仿真反馈 PD；恢复由同一 REC v2 native 脚本→混合→RL 完成，不能把脚本翻身全部归功于 ONNX。

此次证据范围：各一个左侧 90° 倒地出生和一个 flat 命令序列，最多 12 s；无 PPO。两模型左侧链均在 7.46 s 完成，经过 1.02 s 连续稳定峰和约 1.00 s 后续保持。共同实加命令归零窗口 6.34–7 s，最新真实世界平面均速 1.965 m/s，旧 0.083 m/s；最新还在 7.06 s 到达侧界。旧的 2π rad/s 旋转也有 1.509 rad/s MAE 和 28.98° 最大倾角，因此 **两者都没有实车高速合格证明**。源 REC 的普通训练命令约在 ±1 m/s／±1 rad/s，本次 +2 m/s／+2π 是域外挑战。

普通滚动上台阶与腾空跳台阶是两项不同任务，**本次分别都未测**。右侧、腹部、背部自起、rough、跳跃、反向高速也未测。所有 qualification 继续为空；腿侧 bag 不能因此标为已 qualified 或已校准进入训练。旧模型同样没有因为本次结果自动获得实车部署资格。

详细数值、分母与原文件身份见 [原生短回放附加证据](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/model_compare/native_comparison_addendum.md) 和 [机器可读提取](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/model_compare/native_comparison_addendum.json)。累计移动后从出生原点计算的 `stand_drift_max` 不能当独立静止漂移；原 passive 阶段 `first_motion_release_s=0.02 s` 也不能当有效接管放行，正确 pure-RL 后放行时刻为 6.48 s。
