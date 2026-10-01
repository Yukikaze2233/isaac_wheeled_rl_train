# 分任务连续训练部署回执

最新为2026-10-02 05:42核验的v6：源码`34138176d33dcdbbce9a19b2f3c5f7d3f9733639`，合同`v6_task_campaign_v6.json`，同root tmux `v6-task-campaign-20261001T110045Z-282358-r6`、worker `flat/attempt_006/train`。以`launch.json`和`v6_first_ppo_verification.json`为准；actor／critic／Adam完整恢复，10次critic适应计消费，首2443九actor张量改变。4096／35D81D／50Hz策略200Hz仿真反馈PD、fixed LR1e-4、轮响应、每500保存和每域20k上限继承；曲线加行为review可提前结束域，无45项阻断gate，未qualified腿bag仍未接入。

物理时钟延迟和完整10s静止窗口已部署，原生4096工程探针通过。正式flat actor1500→2500渐入，05:42时钟1435所以年龄仍0；不是漏启用。05:45最近20次actor-PPO wall中位11.500s（env9.825／optimizer0.133），v5末50为6.187s，新功能开销待定位，不能继续引用旧5.39s为当前速度。新驻留和停车窗口按有效samples加权，无窗口的0不是成绩。原始当前曲线在`../learning_audit/v6_metrics_snapshot.json`，详细表在`../../../docs/V6_STATIONARY_AND_DELAY_AUDIT_20261002.md`。

以下为v2历史记录，保留当时身份和性能，不代表当前进程：2026-10-01 19:47（UTC+8）部署、19:55核验真实PPO，源码`6fe8f9acb4522852acc02046b6dccde277e3ae50`、合同`v6_task_campaign_v2.json`、tmux r2、worker `train/flat/attempt_002/train`。

## v2历史执行语义

六域flat/recovery/speed/terrain/jump/stepjump，每域20k消费上限、独立模型，顺序常驻PPO。每500消费更新与每attempt首个更新封存；`evaluation_interval=0`，训练中不等待评测、不按分数早停、不回滚。域跑满后封存unassessed候选，qualified为null，再进入下一域。固定评测仍保留，只由独立offline入口执行；训练完成不能冒称通过或六域集成。

跳高概率按任务actor时钟0/2000/6000推进，原地与行进跳均覆盖3–30cm；固定测试和预设台阶高差不重采样。保持20k环境、35D/81D、50Hz策略与1kHz物理/PC PD160/2.5，dense宽域+精核、0.6s落地修复、轮侧响应不变；腿bag未qualified。

## 本次迁移证据

旧v1正常SIGTERM封存于flat第30次、全局消费1030；actor消费1010、actor学习谱系398、完整学习谱系408。v2从该终点恢复，剩余19970次；不是重新从A学习，也不退还历史1000或这30次更新。

`plan_revision_002.json`绑定旧job、新源码提交/归档、旧新plan哈希；`plan_revision_002_evidence/`保留停止、账本迁移、旧completion和新invocation。旧source、job、log与全部attempt保留。六域×五入口预检30项、CPU303测试通过。

`v2_first_ppo_verification.json`记录：actor/critic/Adam逐值恢复，任务时钟从30接到31，首1031不可变模型9个actor参数张量改变，19:57已到本域35次、累计消费1035，新worker连续5次PPO，20786个TB标量有限，GPU约93%/13344MiB。运行时20k和控制配置通过；startup是预rollout快照，未伪改pending字段，成功env.step路径每次严格断言1次解码/20次PD及物理刷新。尚无新技能通过结论。

## 耗时与曲线

GPU控制子函数逐值及符号位等价，微基准0.663→0.292ms（2.27倍），报告`../performance/v5_control_cuda_equivalence_benchmark.json`。不能据此宣称整个PPO快2.27倍。旧30轮剔首29轮，iteration wall中位45.089s、env_step43.905s；新轮次对照持续从TB抽取至`../performance/ungated_timing_comparison.json`。19:56新4轮剔首后的3轮wall中位43.032s，比旧基线低4.56%；样本少且场景重建，不能作稳定速度承诺。更大采样量、1kHz步进与闭链求解成本保留。

TensorBoard只运行在Kaiser，直接读`/home/kaiser/robot-rl-sim60/tensorboard-v6-active`；本机仅SSH转发，入口<http://127.0.0.1:6007/#scalars>。本机旧TensorBoard约9.9GiB RSS及event-sync已停止，v57/v58/v59相关user services全部disabled；见`tensorboard_remote_only_20261001.json`。不为看曲线在本机重开服务或镜像全部events。

## 历史身份

初始d073fe4首attempt因奖励分项导出缺键在首策略步失败，新增消费0；b111344修复后于attempt_001完成30次PPO。`launch.initial.json`、`launch.v1_before_plan_revision.json`、`first_ppo_verification.json`及`recovery_first_ppo_verification.json`是历史证据，不代表当前进程状态。TensorBoard别名中的d073fe4表示该campaign首次部署，不表示当前worker源码。
