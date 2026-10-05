# 实跑配置核对：不能从模板推断已启用奖励

只读核对SCUT本地Git `b8ff79f3df855faf9dc92f4a282bd80c42649466` 的rough_dash/2026-07-19_19-15-07保存参数与当前源码；选定参数/函数文件无工作树差异。保存参数与当前函数可核对语义，不冒充七月当时运行源码的逐字快照。我们的基准是Kaiser冻结b4、speed v14；未修改当前训练合同。

| 项目 | SCUT该保存run | 我们实际v14 |
|---|---|---|
| 静止L1 | stand_still_deadzone_enabled=false，实际reward keys不含stand_still_lin_vel | 前向指令绝对值<0.1时−2(|vx_body|+|vy_body|)，普通纯旋转仍支付 |
| 轮目标动作尺度 | 10rad/s×action | 相同 |
| 腿动作尺度 | .5rad×action | .25rad×action；机构/PD不同，不能直接评价优劣 |
| 初始探索/entropy | 非状态相关scalar Gaussian、std1、entropy .005 | 相同；SCUT最终std未核验 |
| 动作正则 | 一阶−.01、腿二阶−.05、轮二阶−.01，对raw动作 | 同系数，对clip后动作 |
| effort/正功率成本 | −1e−5 | −1e−4，实际倍率1，十倍差异 |
| 普通姿态成本 | s=2exp(−vx_cmd²/3)+2，−2(s·gravity_axis)² | 复用并加floor；零指令两轴均32，3m/s至少pitch18/roll32，SCUT约8.82 |

`flat_orientation_*_v`是随指令速度调整的姿态成本，不是姿态角速度成本；真正roll/pitch角速度成本两边都是−.05(ωx²+ωy²)。此前将这些项说成“SCUT实跑已开−2静止L1”或“−2姿态角速度”不准确，不能继续作为修复依据。

证据：SCUT保存env.yaml L1707动作/L1719开关/L2075奖励表；wheelbipe25_v3/env.py L3553角速度/L3554动作正则/L3571静止mask/L3584姿态/L3748静止L1/L4005配置key汇总；agent.yaml L18探索/PPO。均位于`scut-wheeled-legged-rl`上述冻结revision。我们的`contracts/v6_task_campaign_speed_worker_v14.json` L15392努力倍率/L15445静止与floor；冻结`b4:src/wheeled_tasks/chassis/rewards.py` L138–148、189–220、415–430、463–473。

v14静止成本没有整体饱和；在支持、零指令、直立、纯前向漂移.13m/s的反事实里，broad+fine+L1仍支付约−1.97/s。存在付费不保证PPO采样优势能消除漂移。clean与pipeline同PT配对已证明本模型探索损害静止瞬时精度，不能把全DR的13–14cm/s全归因于探索；也不能因两边轮尺度相同就否认我们实际8.8rad/s target std的控制后果。十倍努力成本、raw/clip正则差异、最终std是待独立验证项，不是已定根因。
