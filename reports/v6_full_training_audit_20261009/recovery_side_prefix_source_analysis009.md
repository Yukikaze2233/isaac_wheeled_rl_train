# 侧倒脚本上游与驻轮机制只读分析009

原脚本已把侧倒转为矢状面恢复。五条超时轨迹约5.305–5.320秒进入CAPTURE，侧向姿态代理仅约0.21–0.30°，其余约64°tilt主要是pitch。动作经过FOLD→SIDE→再次FOLD→ORBIT→THRUST；SIDE退出只要求侧向重力减小，并不要求站直。不能把后续CAPTURE/PREPARE失败说成脚本不能恢复roll90。

CAPTURE/PREPARE按pitch修正四腿参考并以4rad/s追踪目标；冻结源把nominal/support都设为同一个静载补偿prepare命令，所以extension×(support−nominal)恒0。这是可证实的no-op，尚未证明它是错误设计或失败根因。13/19的23/104个PREPARE前采样里，源代数重构remaining<.02与paired_error<.12全通过；重构目标与真实目标误差≤4.77e−7rad，局部remaining未直接记录。sensor conjunction全0、ready从未开始，不能以延时替代原完整门。原1秒稳定计时在RL接管之后；8秒里仅要求达到BLEND，已开始blend可完成。

轮平衡启用时，原控制式可化为`request_i=axis_i*(8*pitch+1.5*gyro_y)−.2*dq_i+probe_i`；关闭时为`−.6*dq_i+probe_i`。正负轮轴对应一致，没有证实反号。末lane19 pitch−7.004°、gyro_y+.0660、probe0，轮速−4.388/+4.307rad/s、目标−4.390/+4.336，请求仅−.00133/+.01750Nm；未饱和却rim.263/.258m/s超.25门限，其他七sensor通过。不能把过渡期4.5Nm限幅当它最终不接管的唯一原因。

在gyro/probe0且请求0的条件代数下，原轮平衡稳速与轮门共同要求|pitch|<5.968°，比tilt门8°更窄。它说明控制驻轮与姿态接受区有相互作用，不是含接触/摩擦/重力的完整平衡证明。既有三个侧倒完整成功事件在6.60、6.98、7.64秒进入BLEND；其5Hz稀疏trace只作机制对照，剔除额外事件样本，不能重建200Hz力矩、连续门或证明时间充分适用于全部lane。

最小后续候选仅建议独占未评分旁路把wheel_balance阻尼.2→.4：其余kp8/kd1.5/轮P.6/腿PD、4.5Nm界、全部门/20ticks、1秒释放/8秒脚本/.2blend/1秒RL稳定不变。相同lane19输入下该代数预期出现约+.876/−.843Nm刹轮请求；只有真实固定初态全脚本→RL结束与原成功任务保护结果才能验证。此候选未创建、未执行；父任务先做更新adapter的确定性模型回放，不改生产或PPO。

公开机制表述：侧倒参考先用左右不对称腿摆动把侧向姿态转为矢状面恢复，再由腿姿态与轮平衡捕获俯仰。末段须同时收敛姿态、角速度、腿参考和轮速；当前证据提示轮平衡稳速区与停轮判据需要单变量验证。尚未证明控制修复有效。

只读native/PPO/远端均0，不改源码/合同/时钟/训练。有限报告包含源SHA与行、分lane轨迹、条件重构及建议：[recovery_side_prefix_source_analysis009.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/recovery_side_prefix_source_analysis009.json)，SHA `e52444d68667256d330c5b6e0fa885b58459b9ef18353e618c96ba9441235dc4`。
