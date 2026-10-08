# 被动 REC 保持联合轨迹

这是诊断附件，不是训练入口或新的控制器。只使用原函数已消费的张量与原状态；没有 App、PhysX、反馈／控制生产、随机数、进程、重试或 checkpoint 选择。生产源、合同、队列与当前 GUI 未修改。

`HoldingTrace(env, mechanical_trace.Hooks(), lanes)` 必须由既有 source023 same-seal 诊断的 observer 创建，显式指定 1–8 条 lane，并使用独立、空的 Hooks 实例。把本附件 SHA 加入既有诊断的 observer artifact binding；源、模型、arm 与 admission 仍由原驱动核验，本附件不另选来源。

既有 observer 的一次策略端点回调中，在获得原 `extras` 后调用 `holding.after_policy(tick, extras)`；原 `mechanical.after_policy` 和父 observer 的调用保留。`_before_close` 的 `finally` 清理中调用 `holding.close()`，再执行既有父清理；不得因诊断异常漏掉原 App 清理。结束时原 publisher 可把 `holding.summary()` 写入其结果。不需要包装 `env.step`、PD、力矩下发或奖励函数；本目录未修改原驱动，也未启动任何原生运行。

记录作用域：

- **5 ms**：原 `FSM.update` 的编码器角度／角速度、重力方向、gyro、估计高度、估计支撑／离地输入，以及原八门和前后 stable ticks／峰值／break／phase／motion release。第 k 次 reference 的名义观测物理端点是 **k−1**；USB 反馈延迟及采样时间戳未观测。没有 5 ms 真实接触。
- **20 ms begin**：原 begin-policy ownership mask 与 motion hold；区间内 actor mask 是 reference 回调处的已有值，位于该子步 torque bookkeeping 之前。
- **50 Hz outcome**：原 outcome 的真实双轮 contact、undesired、重力方向、omega、高度、速度、platform evidence，及原真实稳定／后续成功状态。物理端点为 **4n**，不能拼成 reference 4n 的实时真实接触。
- **20 ms endpoint**：原 extras 的完整区间 actor mask，以及已有 hold-cost producer diagnostics（存在时）。没有重新计算奖励或支付 mask。缺字段即未观测。
- 出生编号仅为 **attach 后观察到的 RecoveryTraining.reset 序号**。释放零力矩时的 FSM.reset 另标记，不能伪装成新回合。出生 reset 后，旧 extras 仍绑定 begin 时的出生编号。

缓存有界；5 ms 只克隆选中 lane 的 device 张量，CPU 序列化发生在策略端点。记录被覆盖、丢事件、缺端点／原调用数不足均显式报告。没有 GPU overhead 测量，也没有当前 REC 的联合归因或功能资格。

CPU 测试从冻结 `6f07ec298e01d2686216787cfba34b394151f1aa` 读取原 RecoveryFSM 与 RecoveryTraining 的 begin／outcome／ownership AST，检查原调用次数、返回对象、输入与状态字节、RNG、八门断裂、reference k−1／outcome 4n、200-tick release、出生／内部 reset、4096 人口下只同步选中 lane、边界容量、清理和异常。episode-reset 侧使用轻量生命周期 fixture；它不等于 Isaac 原生 episode reset 验证。

实际 CPU 输出与限制在 `cpu*.receipt.json`；首轮 fixture 错误回执原样保留。CPU 不授予原生执行、PPO、部署或 qualification 信用。
