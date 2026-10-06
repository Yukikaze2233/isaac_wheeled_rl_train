first50 support/body_clear 口径澄清（不修改原始报告）

此前 first50_summary.md 中“接触／离地条件缺口”的描述不够准确。日志 `/recovery/hold_gate/support_pass_fraction` 与 `body_clear_pass_fraction` 来自恢复状态机的可部署 FK／IMU observer 决策，**不是仿真的真实接触、接触力或真实离地状态**。约55%的通过率只表明这些传感器推断门经常未通过，不能据此证明轮子实际失去接触或接触力不足。

冻结 source021 `src/wheeled_tasks/chassis/recovery.py:41` 明确区分这两个信号与仿真接触。`recovery_training.py:250` 同样说明条件FK高度不是接触测量。真正接触另在 `RecoveryTraining.outcome`（343起）使用 `contact.all(-1)` 与 undesired、姿态、高度、平台等真值连续评分，随后再考原目标跟随；它不反向输入可部署 FSM。

本澄清适用于后续100 actor及两半50窗口摘要。先前first50 raw报告、reader和summary保持原字节，原始零成功与门通过率数字仍然有效；修改的是对信号来源的解释。不能用这些边际门率推出真实接触率、连续1s联合概率或L0单独释放次数。
