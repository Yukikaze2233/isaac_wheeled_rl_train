本方案以冻结 source712、actor2579／update29500完整seal为诊断基线；状态是待同actor原生均值／探索结果补证的提案，没有修改生产。原始full100 SHA `18e4aa8e…934a2d3`，hold100 SHA `e412b1b2…d1ccea`；文件与完整SHA见同名JSON。将来实际接续必须用当时最新own-domain完整seal，不从2579回滚。

已证明的是运动教学曝光不足，而非“没有运动任务”：54恢复组有26静止、26前进0.3m/s与正反yaw0.6rad/s目标，完整actor2469–2568窗口恢复6352800物理观察样本的vx/yaw命令均为零。L0即使释放后仍把原目标清零，窗口L0/L1/L2提案均值88.30%/10.59%/1.12%。基础41组与1449人口仍在，872普通倾角/轻推遗漏已由019修复，不需再修；它们不是新腿库。

100rollout都有hold人口；wheel/leg/gyro单项pass均值19.44%/31.98%/32.92%。这些边际均值不能相乘；all4096 sensorpeak均值14.23ms也不能当恢复组连续1s通过率。14个有效双时钟CPUfixture已排除系统性20ms过早释放，真实物理哪项断hold仍需native逐lesson证据。

SCUT原源码异常出生直接带vx/yaw∈[−1,1]，零主动扭矩0.2s、start-reset1.5s；没有找到active连续1s站稳→motion-release协议。它支持早期非零命令教学，但不能替本repo取消安全门。SCUT静止速度成本按实际vx命令死区触发，纯yaw仍罚身体平移；其当前V14flat未启用tracking奖励扩命令课程。路径、行号、文件SHA见JSON，不以源码中存在的inactive类冒充实际运行。

建议按native结果选择最小候选：

- **先补easy goal-followup。** 最容易的upright/top、零passive-release出生保留每组原followup：26原静止仍零，26forward与正反yaw在真实1s释放后看到原命令。显式改lesson语义/版本；原始历史prefix_stand回执不重写。先保持65%恢复/35%完整基础、原bank与时钟比例，避免同时换reward、探索和难度。若探索根本过不了1s，改goal本身仍不能解锁。
- **hold信号仅按证据补。** 若同2579均值可放行、探索反复断hold，先查实际actor-owned reward付款及wheel/leg/gyro余量。必要时单独冻结一项有界负settling项或替换对应弱项，限定真实actor-owned holding；SCRIPT/BLEND、基础及放行后保持原值。现有姿态/gyro/腿轮成本已经存在，不称缺reward、不重复整套静止税、不奖励历史peak、不恢复旧隐藏peak扣除。
- **困难比例随后推进。** 先证明原目标在actor-valid帧出现。增加中/难样本并不能绕过hold障碍；保持真实reset-only物理前段、单次handover和永久复习份额。离线review不成为在线停训门。

同actor两臂的决策区别：均值放行而探索不放行，支持探索对连续hold/曝光有影响；两臂都卡support/script，应先定位上游；两臂都取得非零command却followup失败，应优先教学/实际motion付款。旧2458均值运动退化是真实结果，但不是2579及reward唯一因果证据。12s仍存活的回合只记右删失。

最终连续1s sensor hold、实际contact hold及followup成功条件都保留；8s脚本、平台、tilt/gyro/height/轮腿速度、vx/yaw跟踪门及原固定评测阈值不变。保持完整actor/critic/Adam/RNG/std，禁用std重置、手工退火或mean-only PPO。新hold项若存在，从首个完全motion-owned策略区间结束；当前释放所在区间仍使用原hold命令，下一50Hz action看到原followup。纯旋转的平移成本继续有效。

| 当前固定里程碑actor | 原origin相对更新 | L0/L1/L2提案 | passive release |
|---:|---:|---|---:|
| 2579 | 285 | 86.5%/12.1%/1.43% | 0.045s |
| 3294 | 1000 | 65%/30%/5% | 0.085s |
| 5294 | 3000 | 35%/40%/25% | 0.200s |
| 8294 | 6000 | 20%/30%/50% | 0.200s |

最早easy目标桥可从下一经冻结候选的首个自然reset生效，无需另等1000更新；不能在真1s门前曝光运动。中/难基线里程碑3294/5294/8294都早于15000终点。首次50更新只作中期观察，100更新才形成完整learning窗口；物理覆盖与行为达标分别验收。

本地launch快照actor2621／恢复已消费2651，独立20k剩余17349；到15000尚需12379 actor更新，假定无新故障/critic消费时恢复总消费约15030，余量4970。这是当前快照预算估算，不是新预留；部署前重读真实账本。所有新PPO计入同一恢复20k及总121k，保留origin和历史消费，不借其他域、不退款、不重复worker。

验收需可观测：source/runtime/95组/41基础/1449一致；两臂初始物理出生与seal哈希一致、savedstd逐值保留；逐lesson真实200Hz八门及50Hzcontact hold分开；首次done永久封口、后续birth分开；每个有自然样本的forward/正反yaw easy目标应在qualified release后有真实actor-valid非零帧，否则明确pending。禁止用组名、平滑人口×边际比例或24帧均counter算精确成功率。第一完整100窗口继续核对九高度静止、vx/yaw、真实conditional停车、完成10s选择样本及分母。机制覆盖通过不给qualified；行为必须过原固定案例/阈值与既有独立seed程序。
