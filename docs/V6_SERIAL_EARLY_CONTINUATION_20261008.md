# 同一模型串行课程：授权提前封存恢复课

2026-10-08，用户要求“能不能马上冻结改为串行的”。该请求替代此前等待 REC 正常 15000 更新边界的调度安排；它不是恢复能力达标声明。

课程依次为：当前 REC 完整状态封存 → joint speed → joint terrain → joint jump → joint stepjump。只有一个正式 PPO actor；4096 并行环境用于采样，后续课程保留旧能力复习，不另训互不继承的交付模型。

## 冻结范围

命名 source028 / v26，基于已冻结 source027/v25 的 59c201678b37cb33a03a878e734d1a263d8ab97a。沿用 v25 的四域 recipe、heading、稠密奖励、物理参数、域随机化、评测与复习配额，仅新增授权 REC 提前停止入口和实际时钟继承。未提交的本机性能候选不进入该源。

保持 4096 环境、35D actor / 81D critic、6 维动作、50 Hz 策略、200 Hz 物理与反馈 PD 160/2.5、64/32 求解器、3 mm 闭链边界、1 s 接稳要求、轮侧响应、毫秒延迟随机化、固定学习率 1e-4。未 qualified 的腿侧 bag 不标记为接入。每域 20k 和总消费 121000 上限保持；历史消费不退款。

## 一次接续事务

1. 提交验证过的代码、合同和本文，从同一 commit 生成归档并核验远端逐文件 inventory。
2. 新入口 CPU 静态准入通过后，仅退休旧 source027 的等待 observer。认证 PID、启动时钟、旧合同、job 和全部 once-action 产物；不向 PPO 发停机信号。
3. source028 observer 使用共享所有权锁，认证唯一活动 REC、真实进度下界及冻结授权，仅向旧 supervisor 提交一次 SIGTERM。旧 worker 完成当前可封存更新，保存 actor、critic、Adam、RNG 和全部真实时钟。
4. 等旧 supervisor、worker、wrapper 退出且归档完成。将实际 stopped campaign_state/completion 原字节保存为不可变 origin；选择同一 REC attempt 的最新 model_final 完整封存，不回退到旧 speed 分支。
5. 原生 4096 环境执行一策略步、四物理步准入；然后一次启动新 wrapper/PPO。10 次 critic 适应计入消费，随后 actor 从实际 REC 源状态继续更新。

提前停止记录为 `authorized_REC_early_stop_unassessed`，qualified=null。实际 REC actor 更新数写入 origin，joint speed 从自己的零时钟开始；不得把不足 15000 的实际训练写成完成 15000。不可变 origin 在后续 own resume 和下一域选择时继续认证。

## 推进与验证

后续自动推进沿用已冻结非阻断调度：到合同端点后保存完整状态，再由同一学习链进入下一域。固定评测和曲线用于描述能力与退化，不能把总 reward 或课程切换当作技能通过。未知 once-signal / native / launch 结果不得自动重发或并行启动。

本文件是执行前冻结计划；实际是否停止、封存、接续及首次 PPO 学习，以 source_repair_028 部署回执为准。TensorBoard 仍运行在 Kaiser，使用本机 6006 SSH 转发；6007 与其它训练项目保持独立。
