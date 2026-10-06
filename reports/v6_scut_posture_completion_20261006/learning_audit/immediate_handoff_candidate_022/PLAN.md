用户要求“执行完动作马上 RL 接管”。本候选只在 source022/v20 的原生 PREPARE 参考生成完成边界改变 actor 的力矩所有权，不等机械稳定或成功。

端点使用原函数实际计算的 `preparing & (remaining < .02)`。`remaining` 是当前参考轨迹尚余角度，沿用原终点容差；实测关节偏差 `measured_error` 与八个稳定谓词只记录，不作为开始 actor 接管的门。原函数两处控制表达式分别从“八门、参考末端、实测误差就绪”改为“参考末端”，从“100ms 就绪后进入 200ms BLEND”改为“直接进入 RL”。原生函数体每个 5ms 参考 tick 仅执行一次，既有 PD、反馈、力矩帽、3mm 机械终止、8s 脚本预算、最多一次 reroute 均保持。

这是“生成参考完成边界接管”：该 tick 最后生成的脚本目标尚未以脚本 PD 施加完整物理子步。接下来的实际 torque provider 返回 100% actor 力矩；不声称实测机械 q 已到末端。端点日志包含原生 remaining、实测误差、前一和本次脚本目标、八门与 phase。根执行器应从原生机械记录提取接管前后各轴实际力矩、姿态与闭链峰，评估突变。

两层语义必须区分：actor 力矩立即拥有控制；原 post-RL 连续1s稳定及真实接触结果判定继续保留，后续 vx/yaw 请求仍按原 motion_released 语义放行。当前不额外改动 command-release。当前 policy 区间若从 SCRIPT/PREPARE 开始，其学习 mask 仍无效；下一原生 policy begin 才成为完整 actor 区间。此诊断不执行 PPO。

`ImmediateHandoffHook` 在原 native observer 完成 source022 身份认证之后、任何 stepping 之前安装。它修改 private observer 的 `original_update` 动态绑定；所有已有 observer 回调保持。`arm=baseline` 只添加读取已有局部值的 metadata 回调，控制 AST 与原函数完全一致；`arm=immediate` 的控制 AST 仅允许上述两表达式改变。使用独立 `HookSet`，必须先恢复此 hook，再调用继承 observer 的 finish/cleanup。`finish()` 校验实际2400次调用及原生参考 cadence，返回恢复错误列表；失败及不完整数据保留，并标注 `qualification=null`。

唯一原生执行由 root 负责，同 source022/v20、同 sealed REC3076/mean actor、同4096初生、RNG、DR、采样协议、600 policy/2400 physics，复用完整 source022 baseline。历史 REC3076 用于同模型 counterfactual 对照，不当成正在训练的最新 checkpoint。干预前比较截止 all4096 最早实际干预的前一物理帧；之后 shared reset/RNG 可分岔，不能仅按选中 lane 的首次干预作全程严格单变量证据。保留 guard/失败/未完成，不据 CPU 工程验证声称 native 功能达标。

CPU 命令（无 App、物理仿真、GPU/PPO）：

    env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/yukikaze/isaacsim60-venv/bin/python -m pytest -q -p no:cacheprovider reports/v6_scut_posture_completion_20261006/learning_audit/immediate_handoff_candidate_022/test_immediate_handoff_hook.py

CPU 检查覆盖真实55 observer baseline261 tick全状态逐位一致，参考尚未到端点不切换，实测滞后/八门失败时仍切 RL，真实 frozen RecoveryTraining.torque 六轴等于 actor 输入，当前/下一 policy 学习归属，原1s连续门、nonfinite/8s/>45deg takeover故障、AST/SHA及恢复拒绝。root 原生数据优先放明确允许的 `/tmp` 新目录；运行前磁盘剩余至少512MiB，禁止覆写旧 baseline/失败证据。
