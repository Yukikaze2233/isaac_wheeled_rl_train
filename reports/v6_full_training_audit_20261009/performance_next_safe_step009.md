# 下一步安全性能审计 009

这是只读源码审查，未运行仿真、PPO或GPU探针，未改训练源、运行进程和source031队列。只新增本报告及同名JSON。

实际固定source029为 `e5bd9c1367b875d7f229b9a1930647013d023372`；已排队source031为 `ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a`。检查的工作树文件逐份与对应Git blob一致，路径与SHA-256见JSON。

已保存的source029第1901–2000次actor完整100更新窗，TB step43303–43402，重新从原始数据逐项计算得到：

| 每更新主机墙钟项 | 秒 |
|---|---:|
| 总墙钟 | 26.208867 |
| 环境步进 | 20.907543 |
| 诊断 | 0.776187 |
| 原生logger | 0.499757 |
| optimizer | 0.136710 |
| progress I/O | 0.017828 |
| returns | 0.008995 |
| 未归属 | 3.861847 |

环境约占79.77%，未归属约占14.73%。这是不主动同步CUDA的主机独占区间，不是GPU kernel时间。原数据与已保存summary逐值相同；本报告没有刷新远端状态。

source031已经含有三类相关改动，无需重复移植：

- `train_chassis.py:970–976`为LearningAudit逐tag写event包围 `audit_logger`。source029相应循环位于968–972，未包含在diagnostics或原生logger计时中。
- `train_chassis.py:1055–1097`及 `state_snapshot.py:14–27`缓存组选择、按字段批量读取快照，并记录 `state_publish`；保留原0.25秒墙钟节流。source029的1067–1084逐组读取origin/commands/episode_step且未计时，实际启动带 `--publish-state`。
- `env.py:1581–1584`把六项有限检查合并成一次host判定；1941–1948把终止原因、success和timeout整数计数一次取回。机械闭链gap的立即检查保持。

LearningAudit.drain在source029的211–219和source031的280–290都先执行一次 `torch.stack(...).cpu().tolist()`，再返回Python数值。因此逐tag写循环自身并不是每tag都在做CUDA tensor转换；具体候选是每tag构造Summary/Event、序列化和队列操作。未归属3.861847秒还包括旧状态发布、runner工作和计时边界外的写操作，不能全部归给LearningAudit。计时标量本身在 `finish_iteration`之后写入，也仍落入下一段未归属时间。

现有host报告候选的66项CPU通过、状态发布候选的18项CPU通过，以及组合的148项CPU通过，分别绑定各自旧冻结版本；不能自动转写成source031全套通过。本地GPU旧报告块57例、32轮交错中位5.222607→1.758571ms，只限有限检查/计数报告块；没有远端PPO、PhysX、writer或状态发布收益证明。本次未重跑这些检查。

最小下一步是等现有source031自然边界采纳，再读取至少一个完整100更新窗的 `audit_logger/state_publish` seconds及calls、总墙钟和环境时间。保存的queue readback012仍为 `waiting_for_normal_speed_boundary`，不是本报告的新远端快照。后课地形难度与当前speed不同，跨域墙钟比较不能被称为单变量提速。

若实际audit_logger占比明确，下一独立候选只新增一个本项目日志辅助模块，在drain之后把相同tag/step/值打包成带多个Summary.Value的event；继续保存全部标量和同一event目录，不删指标或降采样。可证实减少的是序列化/入队调用次数，真实节省尚未测得。不能直接用 `SummaryWriter.add_scalars`代替：本地实现会建立不同子writer/目录并改变主tag。验证须覆盖完整有序 `(tag,step,float32 simple_value)`流、原生CRC多值event解析、TensorBoard HTTP读回、flush/close/异常及自定义writer回退，然后才做同输入交错CPU计时和未来冻结版本的实际完整窗评估。

其次可审查source031 `env.py:2098–2099/2116–2118`的 `any()`后再 `nonzero()`，只计算一次IDs可减少重复host同步；需要空/单个/全部/混合mask、升序IDs、命令/clock/推力及RNG逐值一致验证。闭链异常立即检查、reset顺序和随机化语义都不能延后或放宽。LearningAudit分组归约改变浮点求和顺序的风险较高，不作为第一修复。

本地原生logger还有两处可复核：extras逐步重复 `torch.cat`（185–203）和同一 `action_std.mean().item()`用于写标量/打印两次（211、264）。这些仅为本地已安装代码观察，尚未认证远端package字节；应先核对实际runtime来源，再在本项目独立适配器中考虑一次cat与缓存mean，不修改共享安装包。

另有一个测试维护缺口：source031的 `test_env_host_sync_reduction.py:23`仍固定旧741e130；84–92去掉两块优化后断言整step AST相同，独立只读重放为false，唯一余差是已授权 `env.py:1787–1788`新增 `step_jump=`。两块优化本身与旧已验证候选AST相同。此处不证明性能回归，但旧通过数不能当作此source031整step断言已过；未来候选可更新正确父源和明确许可差异，当前队列保持。

原始证据：[source029_tb2000_exact001.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/source029_tb2000_exact001.json)、[source029_tb2000_exact001.summary.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/source029_tb2000_exact001.summary.json)、[source031_queue_readback012.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/source031_queue_readback012.json)、[source029_gpu_blocks001.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/source029_gpu_blocks001.json)。精确源码与本地依赖SHA、候选范围和验证步骤见[同名JSON](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/performance_next_safe_step009.json)。
