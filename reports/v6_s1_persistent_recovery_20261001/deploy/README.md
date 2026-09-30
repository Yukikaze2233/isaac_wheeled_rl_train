# S1 常驻PPO部署回执

2026-10-01 01:43（UTC+8）部署。本文记录真实启动身份；PPO更新以runtime_verification.json为准。

- 冻结源码：`f7e041b4041516eee190fecc8d7889bbb78a71bd`，无overlay。
- 归档SHA：`50e6189b031cbe8b7df52a40eff7c5ff2177281303012d0bf60c4bae6c95be47`。
- 远端根：`/home/kaiser/robot-rl-sim60/experiments/v6-s1-continue-20260930T174327Z-45dd84`。
- tmux：`v6-s1-continue-20260930T174327Z-45dd84`。
- 合同：`contracts/v6_s1_persistent_recovery_v1.json`。
- 从A完整学习状态恢复，消费778→最多1000；最多222次actor+critic，同一常驻场景。
- 20k、35D/81D、50Hz策略／1kHz物理PD、fixed LR1e-4；四腿响应资格未改变。
- 停车真实0–8s等待后执行完整16sprofile；普通行为落地奖励保持修复。
- 首779及每10消费更新封存；860/940旁路单seed45项观察器已在同tmux独立窗口启动，资源不足明确延后，正式出口与观察器共用评测锁。
- 预算出口45项双seed及322全量双seed，满足原目标与保护才接受S1；不把未过或中途单seed当晋级。
- 当前对话每15分钟跟进automation `kaiser`已启用；正常检查点不再等人工发令。

TensorBoard真实Kaiser入口：[6007 scalars](http://127.0.0.1:6007/#scalars)，实验前缀`v6-s1-persistent-f7e041b`。6006当前是旧本机TensorBoard，不能用它判断本轮曲线。服务在Kaiser，本地仅SSH转发。

证据：[launch](launch.json)、[旁路观察器启动](observer_launch.json)、[TensorBoard环境](tensorboard.json)、[验证记录](../validation.json)。方案见[常驻接续计划](../../../docs/V6_S1_PERSISTENT_RECOVERY_20261001.md)。

## 01:51（UTC+8）真实PPO核验

已新增2次actor+critic更新，累计消费780，actor学习谱系370；不是只建场景或只跑测试。第779封存checkpoint的9个actor张量（8个MLP参数加std）均相对A改变，seal哈希一致。A actor/critic/Adam逐项exact恢复，实际LR与779优化器均为1e-4。

20k原生场景启动断言通过：实际停车时序enabled=true／0–8s／policy_dt=.02；被动落地成本窗口.6s生效。GPU快照90%、13088MiB显存，训练仍在进行。TensorBoard服务已列出`v6-s1-persistent-f7e041b/A_persistent_recovery/train`。详见[runtime_verification.json](runtime_verification.json)。

该证据证明真实学习及配置启用；本轮尚无新的固定评测成绩，不能据此声称停车／yaw已改善。

## 02:25（UTC+8）常驻场景越过40更新

累计消费828、本轮50次、actor谱系418；同一worker PID2023393和同一event文件持续运行，未在40次重建。820封存哈希验证一致；GPU90%／13288MiB。静站组已有20s超时回合结束记录，最近5批制动与驻停窗口均有真实样本。LR仍1e-4，32个普通行为落地奖励tag全0，无非有限scalar。

本项证明常驻生命周期与相位采样实际工作；尚未到860旁路固定评测，不能推断静站／停车／yaw验收已改善。观察器正常等待，无需修改或重启。证据：[continuity_verification_0225.json](continuity_verification_0225.json)。

## 02:54（UTC+8）首次旁路并行核验

860封存点的45项单seed观察于02:47自动开始，checkpoint哈希与seal一致；02:54主线已继续到866消费／本轮88次／actor谱系456，同一PPO worker未中断。仅一个评测进程并行，GPU98%／16502MiB，剩余显存7637MiB、主机可用内存约3.24GiB，仍高于运行中1GiB退出线。PPO单次更新变为约75.6s，资源共享带来减速；此快照不保证后续峰值容量。

TensorBoard实际LR1e-4、scalar均有限、普通落地奖励全0，未见新增成批失败。评测尚未产出结果，不记通过或晋级；合同、预算与进程保持不变。证据：[observer_concurrency_0254.json](observer_concurrency_0254.json)。

## 03:17（UTC+8）860旁路评测完成

45项单seed通过20，S1目标6/16；停车最坏速度由A约24降至13.46cm/s，350/380mm相对778恢复通过，但正向0.5m/s实际仅约0.390m/s，运动域多项回归。两次低姿态前进提前结束是场地边界截断，180回合failure终止为0。未晋级，A保留；主线03:24已到894，按冻结合同继续到940复测。详情与原始哈希回执见[860结果](../observations/update_00000860/README.md)。
