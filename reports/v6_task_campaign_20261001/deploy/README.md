# 分任务连续训练部署回执

- 初始源码：`d073fe4d03c1aeeae52a9569b15c1ec6c226cf66`；首attempt完成20k场景和A完整状态恢复，但首策略步审计输出缺项失败，新增PPO／消费均为0。证据：`first_ppo_verification.json`。
- 修复源码：`b111344932757a2424a46a73dbc1442fa520f280`；只修正已计算reward分项的返回条件及同目录源码修订恢复入口，训练合同和六域预算未变化。
- 当前冻结source：`/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_b11134493275`。
- 当前tmux：`v6-task-campaign-20261001T110045Z-282358-r1`。
- 原run-root及共享预算保留，旧source、job、日志、attempt_000和恢复前顶层回执均留存；新worker为`flat/attempt_001/train`。
- `source_revision_001.json`认证旧job SHA、相同合同、新源码归档及逐文件内容。远端静态预检30角色通过。19:28（UTC+8）已核实真实PPO新增5更新，总消费1005，任务时钟5、actor谱系373；首1001不可变检查点9个actor参数张量变化，恢复与源码哈希一致。证据：`recovery_first_ppo_verification.json`。
- 历史消费1000／actor消费980。flat课程从任务时钟0开始；actor／总学习谱系从A的368／378承接，三种计数不能混用。
- TensorBoard：<http://127.0.0.1:6007/#scalars>。别名`v6-task-domains-d073fe4`表示同一campaign首次部署，attempt_001实际源码以当前launch为准；无需因修订重启TensorBoard。

SCUT的12份event已实际读取，最新flat约1.955s/更新，地形约4.6–5.2s；我方旧无同卡旁路评测时中位42.342s。环境数／物理步／求解配置与硬件不同，不能把iteration比值全部归因代码。参见计划§6.1及`../performance/scut_timing_comparison.json`。

新线首／第二轮实际墙时44.30／43.18秒；第二轮env.step合计42.09秒、纯optimizer计时0.235秒、诊断0.149秒、日志0.055秒。两轮不足以代表稳态；已落实的是调度减少额外耗时，尚未证明物理循环本身提速。20026个TB标量均有限，域固定评测尚未到期。

恢复直接来源是失败attempt_000封存的0更新初始态；已逐元素核对其actor／critic／Adam与A相同。startup中的`pending_first_step`是预rollout快照，未伪改成通过；原生每策略步的严格控制／物理刷新计数断言已在成功rollout路径执行。
