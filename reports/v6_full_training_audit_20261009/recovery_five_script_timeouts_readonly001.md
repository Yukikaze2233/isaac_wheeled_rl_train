# 五个侧倒脚本超时的只读定位

REC fresh004 world_00 为详细样本；004/005 其余五个 fresh world 仅核对失败 lane、终止 phase、8 秒时钟及交接事件。六个 world 均完整结束四倒地 16 回合，11 个完成、5 个脚本超时；失败集合和终止阶段逐值相同。未执行 native/PPO/远端动作，未改输入、驱动、运行时或训练合同。qualification=null。

| lane / 副本 | 最终阶段 | 步后倾角 / 角速度模 | 最后参考条件未通过 | 首次 BLEND / RL |
|---|---|---|---|---|
| 12 / 左12 | CAPTURE | 36.904° / 1.723849 rad/s | body_clear, tilt, gyro, wheel_speed, leg_speed | 无 / 无 |
| 13 / 左13 | PREPARE | 11.853° / 0.895896 rad/s | tilt, gyro | 无 / 无 |
| 16 / 右16 | CAPTURE | 35.959° / 4.023281 rad/s | tilt, gyro, leg_speed | 无 / 无 |
| 17 / 右17 | CAPTURE | 31.558° / 0.869509 rad/s | support, body_clear, tilt, gyro, leg_speed | 无 / 无 |
| 19 / 右19 | PREPARE | 6.976° / 0.120777 rad/s | wheel_speed | 无 / 无 |

三个 CAPTURE 副本最后仍未达到接稳到 PREPARE 的状态；lane 13 曾在 7.90 秒采样见 PREPARE，但结束时倾角和陀螺条件仍未过。lane 19 从 7.52 秒起采样见 PREPARE，其五个脚本 PREPARE 样本轮速条件都未过（7.82 秒还缺 body_clear）；结束时仅 wheel_speed=false。它们的 ever_blended=0、handover_count=0、纯 RL 时间=0、policy 学习 mask 全无。末态 phase 是 CAPTURE/PREPARE，外层解除驱动 ownership 为 failed_disarmed，不能把初始被动释放期间显示的 RL phase 当作实际 actor 接管。

原生 CAPTURE→PREPARE 要求倾角<20°、支撑成立、FK 高度>0.22 m、陀螺模<0.75 rad/s；PREPARE→BLEND 要求八个传感条件同时成立（倾角<8°、高度在0.27–0.36 m、轮缘速度<0.25 m/s、腿速<2 rad/s等），另要求参考剩余角<0.02 rad、成对反馈误差<0.12 rad，并连续20个5 ms参考 tick通过。随后仍保留200 ms力矩 BLEND、纯 RL 连续1秒稳定及原1秒后续任务。条件计数或稀疏采样不能替代连续 ready dwell。

8.0 秒是从初态 reset 起的外层原预算，源码明确包含侧倒1秒零主动力矩的被动释放，约余7秒脚本参考时间。外层达到8秒时停止，内层 FSM 的脚本 age 尚未达到自身8秒，所以保留 CAPTURE/PREPARE、诊断补 code=6；账本 script_seconds=7.9949998856，原生解释为 script_timeout。该现象有原生时钟与注释支持，不能当作新发现的“8秒错配”，也不建议由此延长超时或缩短稳定要求。

采样时序必须分开：trace 的 omega/gravity/root height 是四个物理子步结束后的状态，八个 FSM 条件来自最后一次参考更新、发生在最后5 ms积分之前。lane 17 的7.90秒行因此同时记录 gyro_pass=true 与步后角速度模1.017 rad/s；这不证明守卫错误。真实 root height 也不是 FSM 使用的最小轮接地 FK 高度。

腿速/轮速仅有布尔条件，可给最后参考反馈的门限区间，不能还原具体 dq。每组有原生 torque RMS/peak、closure 最大值，但没有逐失败 lane 的完整电机角度/速度、目标误差、ready计数、轮平衡目标、请求/限幅/滤波/实驱力矩、FK高度、支撑探针和实际接触/闭链几何，不能据此归因饱和、卡几何或唯一训练原因。准备姿态是静态载荷/Kp修正，最大偏置0.027667 rad小于0.12 rad误差门限，单凭这一静态偏置不能证明接管条件互相冲突；nominal/support实际相同使extension差项为零，这在冻结配置中已存在，未证明它造成失败。源表也明确侧倒初态geometry_valid=true而dynamic_recovery_validated=false。

可证实的缺口发生于 native 脚本控制、尚未 BLEND/RL 的阶段，继续 PPO 对这五段没有直接力矩控制。下一次获准验证应复用真实脚本→RL整链，并在现200 Hz参考时刻保留上列反馈、时钟和驱动/几何数据，定位哪一状态量阻止接管；本审查不授权运行或修改。不能让网络替代还在脚本控制的阶段、放宽 gate 或把失败记为成功，也不授完整功能、qualification或实车资格。

详细有限 schema、逐 lane 最后脚本/终止样本、真实 ended ledger、源码行指向及 37 个原证据 SHA 见 [recovery_five_script_timeouts_readonly001.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/recovery_five_script_timeouts_readonly001.json)。JSON SHA-256：`7fb3aaa3e6893401047d54cb3edbbc2f126230da254d19423550e68767182b7e`。

重点原生源码：[recovery.py](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/recovery.py), [recovery_training.py](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/recovery_training.py), [recovery_observer.py](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/recovery_observer.py), [evaluation.py](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/evaluation.py), [env.py](/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009/src/wheeled_tasks/chassis/env.py).
