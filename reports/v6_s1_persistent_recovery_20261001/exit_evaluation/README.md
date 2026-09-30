# S1 消费1000出口评测（45项双seed完成，全量审计进行中）

2026-10-01 05:03（UTC+8）首seed290929的45项评测完成，通过20/45、S1目标4/16：stand_305mm、stand_continuous、backward_05、rotate_1。05:09实读确认seed290930正在运行；322项双seed全量审计尚未完成。本记录不是最终晋级结论。

首seed中305mm静站最大位移2.77cm，高度MAE4.65mm；连续高度静站也恢复通过。但前进命令0.5m/s实际0.325m/s，速度MAE17.53cm/s；驻停最坏10.69cm/s仍高于3cm/s。正转yaw0.1393rad/s通过，反转0.4444rad/s未过。相对940有部分静站恢复，同时运动域仍退步，不能把通过数17→20视为全面收敛。

原始报告SHA-256：`e09151148f9fca1657a061fed85ff84340d27207a113806612786074b4f08282`。评测使用model_final SHA `7dd3ed3b3c959ca31b6bb7f02488d54a7a908a9591d44b059f7c0997b6b63160`，与封存一致；settings、startup和source manifest与940相同，unfinished=0。

消费固定1000、actor谱系590、总学习谱系600；没有再启动PPO、退款或晋级，原A与全部检查点保留。等待确认seed及全量结果，按原目标和39项保护集判定。

证据：[首seed原始报告](evaluation_290929/evaluation.json)、[精简摘要](evaluation_290929/summary.json)、[PPO完成回执](../deploy/ppo_completion_verification.json)。

## 05:13 确认seed完成

seed290930于05:13:51（UTC+8）完成；两个seed都是20/45，S1目标均4/16，通过名单逐项一致，累计39项保护各通过18项。确认的是同一个PPO模型的两次评测，不是两次独立训练。未达到晋级条件，仍按冻结计划完成322项双seed全量审计。

| 指标 | seed290929 | seed290930 |
| --- | ---: | ---: |
| 前进0.5m/s实际速度 | 0.32474m/s | 0.32471m/s |
| 驻停最坏速度（阈值0.03m/s） | 0.10689m/s | 0.11443m/s |
| 正向yaw角速度MAE | 0.13929rad/s | 0.13887rad/s |
| 反向yaw角速度MAE | 0.44441rad/s | 0.44468rad/s |
| 230mm静站最大位移 | 0.14105m | 0.14480m |
| 305mm静站最大位移 | 0.02770m | 0.03420m |

每个seed的180回合均无failure终止、场地边界截断或未完成回合；这不等于所有任务成功。相对A，停车有改善，但前进欠速和反向yaw精度退步不能称为全部贴线，也没有证据证明继续同样训练就会自然恢复。

第二seed原始报告SHA-256为`ba56f81980143cd5e4f6f2c96a5cbc11cbae02e173d533e5074e6ed0ab890094`；检查点与首seed同一封存哈希。settings仅seed不同，startup仅预期seed和quick/confirmation角色不同，source manifest完全相同。两个评测TB事件各3151个scalar tag，step=1000，未发现非有限值。

05:28核验主调度与tmux存活，第一轮322项全量评测PID2092594正在运行；第二轮尚未开始，最终completion尚未产生。消费仍1000，没有新增PPO、修改合同、放宽判据或启动下一阶段。

证据：[确认seed原始报告](evaluation_290930/evaluation.json)、[确认摘要](evaluation_290930/summary.json)、[运行核验回执](../deploy/exit_confirmation_verification.json)。

## 05:37 首轮全量审计完成

322项seed290929于05:37:28完成，共67项通过；S1目标仍4/16，39项保护仍18项通过。与同seed短套共同的45项定义逐项相同，通过判定也完全一致，没有发现短套与全量判决错位。全量包含后续尚未训练的高速、地形、跳跃和恢复等技能，67/322不是S1通过率，也不自动产生下一阶段能力信用。

全量1288回合均完成记账，failure终止634、场地边界截断75；共同45项的180回合failure与boundary仍均0。这些任务终止不是评测进程崩溃。全量驻停最坏11.54cm/s、前进0.5实际0.32481m/s、反向yaw角速度MAE0.44371rad/s，未改变当前主要缺口判断。

原始文件本机与远端SHA一致：`09e303957e4ba377288980190e1672e7474e3b26326b25222e17e4878317dd7c`。模型、源码manifest、合同和资产哈希与短套一致。全量由180扩至1288个评测环境，最长用例由28s扩至32s，startup总体统计随用例集合改变；不能声称整个startup与短套逐字一致。新增TB事件有22588个scalar tag，未发现非有限值。

05:39:55实读第二轮全量seed290930已自动运行（PID2099877），主调度与tmux正常，最终completion尚未生成。S1消费仍1000，未新增PPO、重启或更改判据。

证据：[首轮全量原始报告](evaluation_290929_full/evaluation.json)、[摘要](evaluation_290929_full/summary.json)、[远端核验](../deploy/first_full_verification.json)。
