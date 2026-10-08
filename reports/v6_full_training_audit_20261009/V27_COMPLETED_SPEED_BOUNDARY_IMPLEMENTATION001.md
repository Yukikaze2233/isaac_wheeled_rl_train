# v27 正常 speed 边界接续：隔离候选回执

冻结 `b26a3bd461c0fad76e0681916d727639b840a5b5`，12 文件，父 `bc99f12d2fd802117115fe2ca082e8c513f6c256`（已含 A 静态作者的等价 cherry）。本增量交给 root 组合至 `0d6294b`；未部署、未发远端信号、未运行 Isaac/PPO，也没有未来真实父检查点。

正常 source029 speed 完成后，保留旧 REC/旧 speed 身份与全部付费和学习时钟，用命名 predecessor 证书进入 v27 terrain；完整 actor/critic/Adam/std/RNG 恢复，terrain actor 从0开始，十次 critic 计账。普通大阶8行绑定真实 terrain 引入时钟与余量，后续 joint 阶段保留。未来控制器仅在真实终点前 opt-in 暂停 supervisor，worker 继续正常结束；成功路径原生退出且零 PPO 补闭合旧 speed record，然后才可准入新 terrain。

CPU final002：24 passed / 947.47s；final003：16 passed / 180.42s。11个控制器用例重复，实际29个新增独立用例。旧 compatibility：68 passed / 84.66s。final001 的1 passed/3 setup errors/主动中断和 legacy004 的不存在测试文件/0执行均完整保留，不计通过。最终29项覆盖真实5093+14907付费形状（10critic+14897actor）、preterminal旧record→原native零PPO闭合、跨prefix准入、正常双open/partialcritic ownresume、错误父/未完成/退款/旧terrain/同钟fork拒绝、拒绝后合法重试、三写切断恢复，以及pidfd/STOP cleanup/取消/超时/互斥/原子证据发布。

947.47s为重复 canonical graph/archive/CPU学习状态fixture构造的测试墙钟，不是PPO更新耗时。新增认证位于准入路径，没有进入physics/PPO热循环。未测真实native scene startup或未来首terrain actor，qualification仍null。原始日志、SHA、精确文件blob和限制见 `V27_COMPLETED_SPEED_BOUNDARY_RESULT001.json` 与 `completed_speed_boundary_cpu001/`。

组合时保留root在production inventory中的state_snapshot/step_climb/learning_audit及future recovery_phase_audit字段；本增量不覆盖A四个recipe作者函数。
