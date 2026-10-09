# 实际脚本诊断008独立审查001

008真实原生诊断完整结束：父记录PID189537/start619094与constructor一致，exit0，48.6814秒。独立纯CPU重新认证双模型/pair ABI并执行冻结strict single validator通过；全部原intent源pin、native/diag/NPZ/completion SHA一致，无failed_export回执。审查native/PPO/远端均0，原报告和训练未改。

NPZ有51字段×1600行×5lane，六method各1600次，4,760,000B未压缩；shape/dtype完整。唯一非有限为probe_age_s的19315个+inf（未建立probe证据），无NaN/−inf。pre0..7.995与post.005..8.0是名义原200Hz栅格；实际float32 elapsed末7.995135/8.000135秒。next pre q/dq与previous post q/dq逐值相等，无额外物理步。

原腿PD160/2.5、轮P.6+probe精确重建L1请求，L2精确限幅L1，transport/response链逐值对齐，L3 issued与L4 actual逐值相等。脚本释放后两轮请求峰22.74–26.17Nm、L2/L3/L4均限4.5Nm，超原当前轮界比例约8.51%–11.87%；四腿没有请求超当前界。这里证明记录到限幅和反馈，未证明限幅造成超时或哪种修复有效。

独立重算analysis001的562标量全精确一致。五lane ready最大0且从未BLEND：12/16/17末CAPTURE，13/19末PREPARE，原8秒script timeout及0actor-owned事实不变。最后200子步 all-sensor-pass均0；tilt失败分别200/200/200/200/64，gyro138/121/128/147/84，轮速146/101/148/150/187。原ready还要求remaining<.02、paired error<.12与连续20ticks；remaining未记录，不能把sensor mask缺口当唯一物理原因。

phase_after计数中的RL1.005秒为初始被动释放：四层驱动力矩全0、learning_maskfalse，released以后blend=0，并非RL接管。PREPARE的phase_before为23/104行，phase_after24/105行，两个分母须区分。条件FK高度/支撑为原传感估计，不能授true contact等价。FSM与encoder累计角在释放后有≤1.192e−7rad的少量浮点差异，晚窗PD/paired误差仍同分析值，不宣称逐位角相同。

reset、完整policy trace、native events、episode metrics和quiet metrics都与004未记录旁路的REC world00逐值相同；仍不授隐藏solver状态等价、唯一actor/训练归因、qualification或实车资格。原006/007与负审查保持。有限JSON含全证据SHA、每lane分母/范围和精确源行：[recovery_script_diagnostic008_independent_review001.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/recovery_script_diagnostic008_independent_review001.json)，SHA `04783e9d6daed1fe033e0b9a975541a46564e260a8783eb987e7fdf62a176f36`。
