# Entropy 单变量试验的最小实现设计（source021，只读）

修订2（只读复核）：修订1原字节已保存于 `entropy_trial_implementation_design_021_001_preserved/`。原JSON SHA `a7828dd4a96dc7640f6265bf8618df7f417ebf6fa5402ce1df14e44ca3447def`、原MD SHA `dc23ff62a5ad48edbd3e1d40716e6b92023c476726b20d0ba2cb537c496d6a87`。本次只补方案缺口与执行优先级，没有实现或批准440试验。

本设计选择 **4×（10 次 critic-only＋100 次 actor 更新）＝440 次优化消费**。A 的原生 entropy coefficient 为 0.005，B 为 0；两种真实训练 RNG fork 各跑 A/B。沿用原来的 `TaskCampaign`、长期独占 claim 和 `TrainingBlocks.execute` 串行子进程，不增加调度器。当前 source021 正式训练继续保持原合同；本文件没有实现或部署任何支线，也没有有效 PPO A/B 结果。

## 现有机制和必要缺口

| 冻结源位置 | 可复用机制 | 最小必要增量 |
|---|---|---|
| `scripts/run_task_campaign.py:87–123,745–811` | 单 `self.child`、原 claim、全 run 独占 flock、持久报告 | 一个命名 plan 迁移，只增加一次性研究 manifest；在普通域 dispatch 前串行执行四臂 |
| `scripts/run_chassis_blocks.py:262–286` | 原 `execute`、x-mode log、native child、publish/signal/deadline | 直接复用；不造研究 scheduler、并行 worker 或采样线程 |
| `scripts/run_task_campaign.py:140–177,276–314,796–798` | replay/`account`/非退款更新计费 | 统一枚举 manifest 注册的研究 progress；原浅 glob 会漏 nested 支线。报告加 trial/arm 身份，不能把支线行为当主线行为 |
| `scripts/run_task_campaign.py:499–541` | 主线 own seal selector、原 train CLI | 主线 selector 不改；支线显式 common parent 或 own-arm seal，独立 namespace，完成后显式恢复原 main seal |
| `chassis/task_campaign.py:50–110,375–390,709–819` | 命名迁移、v19 精确继承、20k/121000、`max(successful,attempted)` | 另加严格研究 recipe 入口；普通 v19 精确校验、正常 seed290927 和六域预算不放松 |
| `scripts/train_chassis.py:61–72,647–711,753–786,916–950` | 完整 learner/Adam/std restore、原 RNG restore、原 reset、原计费 | 接入已独立 CPU 审过的 optional entropy_entry002；只在研究臂添加 admission＋完整 RNG save/restore/fork，保留原 10 critic 规则 |
| `chassis/preflight.py:95–101,385–461` | 新路径可用现有独立设计表和 fixed seed 规则 | 四臂专用设计行钉住完整 v19、显式 beta 与 fork/admission，不新增 preflight 框架 |
| `chassis/evaluation.py:340–397,435–448` | v19 SHA 绑定的旧独立考试、禁用教学 hold tax | 先审核研究 recipe，再投影其 original v19 exam；case/criteria/catalog 不改 |
| `scripts/chassis_remote_job.py:23–96,182–219` | 一 wrapper/一 supervisor、原 source 迁移、自然归档 | 只登记新 named plan revision；不加 campaign CLI flag，不绕过 old-owner quiescence |
| `scripts/chassis_batch_export.py:51–84,113–120` | 不可变 batch SHA、已有包装 | 显式加入已注册的 nested 支线 checkpoint/completion/eval；原浅 glob 不包含它们 |

上表相对路径均以冻结根 `/tmp/scut-recovery-frozen-021-qv1ukmbr` 为准。JSON 列出每个绝对源路径和 SHA。真正入口叫 `execute`；仓库没有可直接使用的通用四臂 `run_job` API。`queue_task_campaign_continuation.py` 是 v15 的专用自然交接，不能当作此试验的现成队列。

## 一次合法迁移和同父隔离

下一版源在候选目录完成实现、测试与 native 准入后，才进行一次原 owner 合法封存迁移。新 plan 只绑定一个不可变 trial recipe，六条正常 worker 的字节、课程终点、域次序、历史1000/980 和每域20k/总121000不改；实际新 revision 名称仍待作者冻结。

在迁移时只选择一次**当时真实 latest-own REC 完整终点 seal**，将 PT/contract/seal/source/archive/claim snapshot SHA 写入 x-mode admission，claim 的 `trial_state` 绑定 admission SHA。它既是四臂共同祖先，也是 main 的明确恢复源；REC3076 旧0PPO诊断不能自动代替它。静态源 recipe 可绑定选择规则，实际 PT 身份放入原 owner 已认证的 admission，避免在当前 PPO 仍运行时预填一个假的最终父。

研究 namespace 采用：

```
<trainroot>/research/<trial_id>/<seed_arm>/recovery/attempt_NNN/train
```

这样原 `account(directory,record)` 的 `directory.parents[1].name` 仍是 recovery，同时主线 `recovery/attempt_*/train/**/*.checkpoint.json` 不会捞到支线。必须补一个 manifest-owned 的 progress/seal 路径枚举器，在 replay、finally 和 batch export 共用；不能简单扩大 main glob，也不能接受任意 foreign research 文件。

顺序冻结为 seed391021 的 A→B，再 seed391022 的 B→A（两个数仍是提案，尚未准入）。每个新臂都显式载入同一父，无论前臂消费时钟多高；重试只选择本臂有效 seal，保留共同祖先，不得选另一臂。四臂完成后 main 从原 own seal 恢复，保留全部支线消费债；没有默认采用赢家，也没有把这440次支线优化算进 main 的学习谱系。

严格 branch 合同由完整 v19 worker 派生，只容许新增显式 `entropy_coef`、`research_entropy_trial` 元数据与独立 `design_preflight` 绑定；其他字段逐值相同。研究校验先剥离这几条枚举路径并对完整 canonical v19 做相等检查，再执行原控制/网络/时钟/预算检查。正常 v19 的精确 canonical 路径保持原样。可新增一个只在合法 branch 下必需的 worker `--entropy-trial-admission FILE SHA` 参数；campaign 的原 wrapper command 只允许 runner/contract source-prefix替换，因此不应额外加 campaign 参数。

## 440消费、110臂上限和故障恢复

原 `account_updates` 已能将不同 attempt key 累加到同一 recovery domain。claim 仍是唯一权威；无需新预算账户。预检必须验证当前 `charged_REC+440≤20000`、`charged_total+440≤121000`。

440先作为**尚未尝试的容量预留**存在 trial_state；不能预写四个 `attempted_updates=110` 伪装已消费。原 `update_attempt` 会拒绝后续实际计数倒退。每次原生优化前的真实 attempted 记录仍逐次传入 `max(successful,attempted)`；容量单位转成真实收费只做一次。claim先写、campaign_state后写的耐故障顺序不改。`update_attempt` 会把 attempt 重写成 domain/consumed/actor 三字段，因此 manifest/arm/reseed状态应在 claim 的 trial_state，不能假设这些额外字段会自动留住。

每臂110收费上限含10critic和100actor。critic沿用原新SHA10次适应；同SHA恢复仅完成剩余次数，不重新做10次。故障中未封存的已尝试更新照计，可能导致该臂在110收费内无法凑齐100成功actor，应记录 incomplete/inconclusive，不能退款、删目录重跑或私自扩大440。只有本臂有效完整state才能恢复；NaN/坏state不能封成学习源。能力不好只是研究结果；source/owner/机械完整性失败不能被能力指标掩盖。

若 admission 时 main 消费为 C、学习谱系为 L、task actor为T，则无故障四臂的消费终点为 C+110/C+220/C+330/C+440，而每臂自己的学习谱系都是 L+110、task actor都是 T+100。main恢复仍是自己的 L/T，消费为C+440、actor已消费债增加400；四次丢弃支线的优化不会“免费”也不会让 main 假称学习了它们。正常 v19 worker SHA 没变，main恢复不重复 critic warmup。

440×4096×24＝43,253,760次收集 transition，4个200Hz子步＝173,015,040物理interval；每臂52.8s native时程，其中actor学习48s。按目前14.675–21.2355s/update量级，更新本身约108–156分钟，另加四次冷启动/封存/包装与离线诊断；共享负载可改变这一区间。这不是全天训练速度承诺，也不能引用本机 COM 工程加速百分比冒充 Kaiser PPO收益。

## 真正RNG fork：库存和位置

**仅改CLI seed无效。** `train_chassis.py:673–677` 在 native runner fullrestore后，会用父checkpoint覆盖 CPU/CUDA/env随机状态。应在完整模型、Adam、std和这些父随机状态恢复且指纹验证后、真实 task/stage clocks恢复后、`:701–711` 的 reset/resample前执行一次明确fork。

构造期静态DR早于parent load：每臂用相同的已申明 fork seed构造 native ContactDomain/RigidBodyRandomization，保持原 seed派生公式和draw顺序；之后才恢复父保存的随机状态并显式fork live streams。这是两阶段的**同seedA/B配对新episode**，不是完整物理快照fork。

| 实际库存 | 当前REC状态 | 正确处理 |
|---|---|---|
| Torch global CPU、所有本进程使用的CUDA默认流 | active；native actor随机sample与mini-batch randperm使用它们 | 先认证父状态，再按同一seed/device映射fork；本臂resume只restore，不reseed |
| env.generator＝perturbations.generator＝signal_delay.generator | 同一对象，传感器噪声/延迟/命令/姿态/扰动共用 | 稳定logical name＋alias map，按对象identity去重，只seed一次 |
| recovery/ordinary/nonstandard/skill/schedule/JumpAssist随机选择 | 没有独立generator，消费env.random或传入env.generator；JumpAssist当前关闭 | 不制造私有课程随机流，不重复seed，不启用助跳 |
| command_transport.generator | 独立active，原seed XOR0x43414E | 首fork保持原派生；新支线checkpoint须保存它的当前state |
| LegacyUSB两个generator | 当前usb_transport=False，对象不存在 | 要求缺席；不捏造state、不叠加transport |
| StepTaskSensors.generator/其IMU.generator | 仅step_jump.has_tasks才创建；REC未声明该任务，runtime须确认缺席 | 不因顶层配置enabled就猜有live流；意外对象拒绝准入 |
| recovery.imu.generator | 对象存在，fixed190619；**noise=False，当前不消耗随机数** | 注册/保存它，仍保持noise=False；不能说独立IMU噪声正在域随机化 |
| ContactDomain local generator | 构造期局部变量，之后没有livestate | 原forkseed XOR0x46524943；核对实际domain/friction指纹 |
| RigidBodyRandomization local generator | 构造期局部变量 | 原forkseed+8291；核对质量/惯量/COM原生readback与静态COM缓存代次 |

课程alias的实证位置：`env.py:579,586,1002,1007,1046`，`recovery_curriculum.py:203`，`ordinary_posture_curriculum.py:132–148`，`nonstandard_start.py:248–268`，`step_assist.py:55–57`。恢复IMU由 `recovery_training.py:87` 按 `SimulatedImu(...noise=False)`构造，源码没有训练入口改变它；不能为了制造不同seed证据额外启用噪声或多做一次reset。

完整库存须使用稳定logical path，object id只用于alias去重，不能拿来派生seed；也不能用进程随机Python hash。具体channel seed表仍待canonical authoring，当前没有可执行forkhelper。私有活跃state保存是必要最小增量：旧source021仅存CPU/CUDA/env，无法真实恢复parent的command_transport state；首分支应明确是新episode并初始化该流，不能补造丢失的父state。之后所有支线seal保存完整registry/aliases/first_fork_done，本臂resume载入当前state并检查alias，不再次fork。

指纹读取不得消费随机数。记录原producer已经产生的首observation/command/lesson/rawstochasticaction，验证同seedA/B相同、不同seed实际draw不同；同时核对构造DR指纹。同一个seed在策略优化后episode长度和随机调用路径可能分叉，不声称全程common random numbers一致。新源若出现未注册livegenerator必须拒绝，不能静默略过。

## 原生观察和验收边界

每个110更新臂的 dense窗口应读**actor1..100**：消费event为该臂rebased parent+10..+109，前10critic单独排除。现有旧reader默认actor11..110需要120优化，不能生搬到110预算。Behavior可用真实第10critic结束的完整95组快照作actor0 baseline，再用完成110作actor100 endpoint；缺baseline或CRC/tag覆盖不足只能pending，不能填0。

复用原native producer/日志/0PPO观察器，分别读6axisstd、KL、HoldCost付款、普通静止/欠速/yaw角速度/条件停车frames、姿态；L0真实release、原nonzero followup曝光与L2整链成功分开。support/bodyclear是FK/IMU observer proxy，true contact在outcome另判，不能将gate均值当连续1s联合概率。B=0只去除entropy bonus，仍按原stochastic policy训练，不切mean、不手调std。

0PPO工程场景已出现的3mm guard触发与较大closure峰值仍是机械问题，必须保留事件和firstbirth竞争失败/删失分母。哪怕B的释放指标改善，也不能宣布机械合格、整链通过或实车就绪。两个真实训练seed的B−A方向一致且普通质量保留，才形成可讨论的科学结论；std更小、总reward更高或只一seed有效都不足以自动采用。

## 部署前最小验证清单

1. 严格deep-diff：显式A/B beta，缺0字段拒绝；改reward/control/std/Adam/LR/guard/delay/origin/其他worker或seed拒绝。原v19普通入口测试逐值兼容。
2. 原claim/serialqueue：同owner与child-start gap、registered nested replay/archiver、main/own-arm/common-parentselector隔离、x-mode重复launch拒绝。
3. 真计费：440容量不是假收费；优化前/中/完成后断点、postoptimizer coefficient故障、未知坏completion、重复replay都保持nonrefund，预算/remainingcritic正确。
4. CPU RNG：alias只seed一次，未知库存拒绝，savedstate/forkmarker恢复，read指纹不改变state， dormantIMU不新加draw。
5. Actual native：same-parent actor/critic/Adam/std不改；同seed首draw相同、异seed不同；10critic/100actor分账；完整source/design/argv/module/普通1449/原gate/机械guard身份。
6. 独立review＋sourcefreeze/native准入后，只做一次旧owner合法seal迁移。旧wrapper的全量finalize自然退出，不能为了省冷停税绕过封存或强杀。

以上机制仍需实现、独立审核和实际native验证。optional entropy_entry002只提供可选coef接线及postoptimizer故障记账证据；它没有研究recipe、RNGfork、queue或预算准入资格。本设计写入两个新增文件，没有停止或修改当前PPO、没有申请预算退款，也没有部署或认定有效试验。

## 修订2：两处真实隔离缺口必须关闭

**claim 必须拥有 trial_state，容量应由所有真实 attempt 派生。** `run_task_campaign.py:140–160` 目前从claim取attempts后，再把本机campaign_state并入report；`:113–123` 的persist重建claim，没有trial_state字段。因此把研究cursor/容量只放local report会在crash时丢失或落后，随后可能重复执行臂或重复释放容量。最小改动是在同一原claim中持久绑定admission＋trial_state，open_account从claim恢复它，不让localreport覆盖；先replay所有注册progress，按每臂全部真实attempt总消费计算charge，再对尚未terminal的queue entry计算`max(0,110−charged_arm)`容量。容量是派生值，不能对旧计数反复减1。claim先写、local projection后写；重复publish/replay必须完全幂等。closed incomplete臂的未尝试容量可释放，已消费任何单位不得退款。

**研究publish不得调用普通domain review。** `run_task_campaign.py:299–313` 会把active_domain写为REC并调用review_request；`:316–362` 可消费root的domain_review_request、设置main pending_review并给`self.child`发SIGTERM。即使trial snapshot与normalworkerSHA不匹配，让review_available路径跳过，已有合法main请求仍可能误停研究子进程。active_trial时必须跳过原assess_request/review_request，保留main `_active_review`、pending_review、consumed_review_tokens以及root review_available/request/response原字节；支线seal/review只能在自己的manifest-owned namespace。根progress显式标trial/arm，trial行为和startup复制使用独立路径或引用，不假称main当前质量。trial结束恢复原main review上下文；期间新来的main请求可等待正常main路径处理，不能被研究臂消费。

还需明确：这是一个新的命名plan＋source迁移，不能把它塞入已准入source021同合同迁移生成器后仅改archiveSHA。新plan-id应显式加入所有CONTINUOUS/REVIEW/SCHEDULE身份集，固定终点映射逐值继承v19；否则validate_plan对未知版本的分支可能落到旧short endpoint表。原六worker可仍逐字节不变。

这些不是已实现功能。新增CPU验证应包括claim已写／localreport未写的crash、同progress重复replay、跨attempt不确定收费、有效main review请求在trial期间不发signal/不消耗token/不覆盖文件，以及恢复main后原请求正常处理。

## 修订2：下一步优先机械，不立即运行440

根任务决定先完成脚本owned机械capture。已认证same-REC3076 mean/stochastic的lane1708@tick57与2216@tick59逐值相同：SIDE、original_chain、actor mask=false、未BLEND/handover，由closure_gap终止。此时entropy没有控制权；beta A/B不能修这两条真实脚本阶段失败。几何诊断指出非对称台阶支撑与通用对称prepare/fold存在待验证路径风险，但尚未判定第一处动态失配是接触、力矩限幅、参考跟踪还是求解。

mean恢复release612／success281、stochastic0／0仍支持一个独立的应用探索屏障；它不证明entropy bonus是充分根因，也不把机械失败变成reward问题。建议先复现首guard、冻结一个证据支持的脚本／出生／物理改动复验，保留3mm及原诊断／评测覆盖；不能同时改beta遮盖归因。当前正式source021仍不改。新的capture如果只是driver rejection，不算物理结果；本修订未引用任何尚未准入的capture性能结论。

440方案保留为后续条件性因果试验：当物理合规的hold子集仍有同父mean/stochastic分离，而且严格recipe/claim/RNG/原生准入补齐，再评估是否值得执行。不能因为轮std较大就立刻投入440与新增调度改动。原chain机械问题、常规质量和探索hold问题分别报告；改善一项不能授整链或实车资格。
