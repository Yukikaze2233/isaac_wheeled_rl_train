# 项目协作约定

## 训练指标与图表

- 训练曲线和图表数据统一使用 **TensorBoard** 记录与查看。
- 新增reward、loss、性能或行为指标时，接入现有RSL-RL logger的event文件，使用稳定tag和明确的step/单位。
- 默认通过TensorBoard比较实验；需要静态图片时，从event数据导出，避免建立独立的曲线数据源。
- checkpoint、合同、评测判定、归档manifest和SHA-256回执继续承担机器可读的恢复与验收职责。
- TensorBoard和SSH端口转发是独立监控进程，不能成为训练进程的生命周期依赖。
- Kaiser远端训练的TensorBoard服务端运行在Kaiser，直接读取远端event；本机只保留SSH端口转发与浏览器访问。用户于2026-10-01再次明确禁止本机运行TensorBoard（旧服务约9.9GiB RSS）；v57/v58/v59本机TensorBoard及event-sync服务已停止并禁用，不要重新启用。Kaiser曲线入口为本机6007转发至远端6006。

## 当前主线

- 2026-10-02最新用户授权修复并评估续训。候选v4补连续幅值0.05–1及curve组50%独立有符号vx/yaw，固定高度和行为池配额不改；成功事件5→30修复完成前等待的回报反转，失败仍单次200（原生反例否定照抄SCUT的-4），reward_units显式冻结。10次critic-only适应全部计消费但不推进actor课程时钟，同合同故障只恢复余量；其后常驻PPO。365 CPU测试／30入口通过，源v3→v4严格白名单及同root账本迁移，不热改、不回A、不退款、20k/35D81D/50Hz200Hz、每500封存与零评测等待继承。候选尚未部署，实际仍查launch.json；审计 docs/V6_POLICY_REWARD_REFERENCE_AUDIT_20261002.md。

- 2026-10-02 02:45（UTC+8）核验：冻结源码`c491bc4e949853f4562cb65d46b1ff9ac796e514`和v3合同已在原root接续真实PPO，tmux `v6-task-campaign-20261001T110045Z-282358-r3`、worker attempt_003。从v2 flat544／总消费1544完整封存接续，actor/critic/Adam逐值恢复，首1545封存9个actor张量改变；核验时新增20、flat564／总1564、actor谱系932、fixed LR1e-4。20k／35D81D／50Hz策略继承，仿真物理与反馈PD改200Hz（4次新反馈），RMCS和raw bag仍1kHz；显式近似未验证硬件等价。17个去首轮TB更新中位12.83s（旧50次45.14s，约3.52倍），优化器.251s，环境步进11.20s；非严格配对行为测试，不声称技能通过。监督进程5.04s I/O写655kB（旧166MB）而非每秒重抄32.9MB不变startup。326 CPU测试／30入口预检和20k原生探针通过，闭链1.485mm且闭链／气簧行程越界0；零动作探针fall3／非轮接触20336不是策略能力成绩。每域20k、总121000、每500封存、评测等待0、候选unassessed；轮响应／reward／网络／求解器不改，未qualified腿bag仍未接入。实际身份与学习证据见`reports/v6_task_campaign_20261001/deploy/launch.json`和`v3_first_ppo_verification.json`，不要重复启动。现有kaiser自动跟进提示已对齐v3，但工具返回状态仍PAUSED；不声称自动心跳在运行。

- 2026-10-01 19:47（UTC+8）已从冻结源码`6fe8f9acb4522852acc02046b6dccde277e3ae50`在原run-root完成v1→v2封存迁移，tmux `v6-task-campaign-20261001T110045Z-282358-r2`，worker attempt_002。从旧线第30次完整状态续接（总消费1030/actor消费1010/actor谱系398），首新PPO接taskclock31、总1031。19:55核验actor/critic/Adam逐值恢复、首1031检查点9个actor张量改变、4154个TB标量有限；实际20k/50Hz1kHz/LR1e-4、轮响应和dense/落地修复继承。30入口预检、303 CPU测试及GPU控制核逐值等价通过；控制子函数.663→.292ms不能表述整机2.27倍。每500封存，评测等待关闭，无技能通过信用；部署与核验见launch.json和v2_first_ppo_verification.json。训练当前在跑新版，不要把v1停止回执当故障再开一份。

- 2026-10-01最新用户再次明确：“解开gate，修正训练后一次跑完，上限放大，而且要快”。新版`v6_task_campaign_v2.json`延续每域20k／六域121000总消费上限，取代下文v1每1000合作评测和达标早停：evaluation_interval=0、每500封存，按消费上限接下一任务，候选unassessed且qualified=null；固定评测只由显式offline入口执行，不堵住PPO。jump概率按本域actor时钟0/2000/6000推进，原地与行进跳均覆盖3–30cm，台阶几何不改。v1→v2必须在同root从最后封存状态迁移，完整学习状态、任务时钟和全部消费保留，不得回到A退款。运算等价控制优化保留20k/35D81D/50Hz1kHz及物理/reward/轮响应；四腿bag未qualified。具体已部署版本仍查launch回执，不能看到本条就启动第二个worker。方案见`docs/V6_TASK_DOMAIN_CAMPAIGN_20261001.md`。

- 2026-10-01 19:17（UTC+8）同run-root从修复源码 `b111344932757a2424a46a73dbc1442fa520f280` 命名恢复分任务连续训练，冻结合同未变；旧d073fe4首attempt因奖励审计输出键遗漏，在首策略步失败，新增消费0。修复不改reward数值：返回已计算分项；源码修订回执严格保持原命令参数／合同哈希／共享账户，保留旧source、job、log及attempt_000。当前tmux和source以 `reports/v6_task_campaign_20261001/deploy/launch.json` 为准，19:28实读新增PPO5／总消费1005／taskclock5／actor谱系373，首1001快照9个actor张量改变；首PPO证据 `recovery_first_ppo_verification.json`。startup为预rollout快照，其pending_first_step不能误判为没有执行，实际rollout逐策略步严格检查1次解码／20次PD／20次物理刷新。不要把初始失败回执当当前进程状态。SCUT12份event真实计时、平台与闭链／求解配置差异见计划§6.1和performance目录；不得把环境数×子步倍率当墙钟加速承诺。

- 2026-10-01最新用户授权：“像华南虎那样20k上限……别重复启停，得多个单任务”。新计划 `contracts/v6_task_campaign_v1.json` 将 flat/recovery/speed/terrain/jump/stepjump 六域分别设20,000更新上限，历史消费1000（actor980）不退款，总消费上界121000；这取代旧8500的未来预算限制，旧S1封存失败仍有效。首域从A完整actor/critic/Adam恢复，后继域独立模型；每域常驻PPO、1000更新合作双seed评测，低分继续同worker，全过可提前结束，未通过不能qualified；父研究候选可初始化后继，不称集成或硬件通过。20k环境/35D81D/50Hz1kHz/160-2.5/轮响应和落地修复不变，腿bag未qualified。计划 `docs/V6_TASK_DOMAIN_CAMPAIGN_20261001.md`、部署身份 `reports/v6_task_campaign_20261001/deploy/launch.json`；未有真实PPO回执前不可声称已开始。heartbeat应核对新预算、不可变候选、任务时钟与消费分别记账，不再因旧S1余0误停新授权任务；严禁热改、退款或重复并行训练。

- 2026-09-29新授权研究主线为 `contracts/v6_new_asset_training_v1.json`：105°相对膝止挡、四主动轴continuous、20k、1kHz物理/PD与50Hz策略，网络35D/81D不变，从零开始独立8500预算。下列旧750/P0/200Hz记录属于历史运行；不得混用合同。名义工程先验训练不等于G1或硬件发布。

- 业务模块按领域职责使用通用命名，例如`rewards.py`；参考项目来源写在注释和文档中，不作为新模块名前缀。

- V5入口和边界见`docs/V5_ARCHITECTURE.md`。
- 运行、TensorBoard访问、断连与回收说明见`docs/V5_INTEGRATED_TRAINING_20260921.md`。
- 远端运行使用冻结的源码归档；本地改动与提交不等于已部署到运行中的任务。
- 正式35D课程停在750次；当前运行独立诊断`contracts/v6_p0_diagnostic_v1.json`，顺序执行高度扫描和两组各100次更新，协议见`docs/V6_P0_DIAGNOSTIC_20260928.md`。诊断结果不自动晋级或清零正式消费。
- 正式课程版本仍为`contracts/v6_scut35_budget_retry_v1.json`：200Hz物理、50Hz策略，USB下行随机化使用独立1ms时钟及区间力矩脉冲近似。第500次停队分析与预算迁移见`docs/V6_STOP500_BUDGET_RESUME_20260927.md`；精跟踪和各阶段累计保护采样见`docs/V6_PRECISION_RECOVERY_20260927.md`。
- 恢复学习状态与恢复预算必须分开：回滚不返还已经消耗的更新；保护恢复点须包含当前合同下的actor、critic和optimizer。后续阶段保留高速/静止池下限，并根据累计保护行为族分配复习份额。
- 每轮训练迭代必须先提交已验证的训练代码、合同和计划，再从该commit冻结部署；启动回执记录commit与归档SHA。运行结果另行提交文档回执，不以未提交overlay代替训练版本提交。

## 训练设计原则

- 所有技能统一按“完整任务域、稠密过程目标、阶段性难度、分项验收”设计，不仅限于高度。
- 先核对自有模型的机械、执行器和命令组合边界，再定义高度、速度、转向、自旋、地形、跳跃与落地的目标域；参考项目的数值不直接照搬。
- 从相应能力阶段开始就覆盖该任务的完整目标域，低速／中间档作为覆盖样本；用轨迹速度、扰动和组合复杂度分阶段提高难度，不以长期窄带训练代替完整能力。
- 奖励覆盖动作过程与相位，审计实际量纲、核形状和相互冲突；保留固定回归与最终验收，不能把稠密reward上涨当作功能通过。
- 新一轮应先形成统一训练合同、能力矩阵和验证方案，再提交部署，避免不断追加零散小修复。

## 硬件部署约定

- 用户确认：WheelLegInfantryRL硬件为2个DM髋电机、2个同型号DM膝电机和2个M3508轮电机；腿部经链传动至同轴嵌套输出。
- 六轴统一由PC侧位置／速度反馈计算力矩；DM使用模式A的`control_torque`，不使用模式B的电机内位置PD。策略输出的位置／速度目标是PC内部中间量。
- 用户于2026-09-26最后确认：前膝电机、后髋电机，电机输出经1:1外部链传动到同轴嵌套轴系。此条覆盖此前相反的安装记录；功能轴仍按髋输入／膝输入绑定。
- 机械运动关系：大小腿夹角不变时，同侧两个主动输出轴等增量随动；大腿相对车体角度不变时，改变大小腿夹角只需膝驱动轴转动。相对膝角由两根主动轴及闭链几何决定，不等于膝电机轴角。
- 功能轴映射、驱动口绑定和数值标定分别记录；零位、符号、传动比、反馈／力矩接口参考端及实测力矩曲线不得凭安装位置或同轴结构猜测。
- 2026-09-29较早规划（已被下条覆盖）：未来单束前向激光配合预设高度，触发普通STEP与抬高参考，人工请求共用入口。保留此条作为历史，不再作为当前普通上阶实现依据；不代表硬件已安装或RMCS已接线。
- 2026-09-29最新授权：普通上台阶仅接收持续前进速度，由策略自行抬身、允许合法台面承托并收腿；低阶、楼梯、高阶和普通双级均不增加manual/laser STEP触发或定时height_cmd抬升。仅主动跳跃使用请求/参考状态机，台阶跳保留manual/单束激光两种入口。真实地形/接触用于奖励验收，不能进入命令状态机或actor；35D原预留frame[30]仅在台阶跳请求时表达预设落台高差×5，网络35/81和控制50Hz/1kHz不变。
- 新技能候选为`contracts/v6_step_curriculum_training_v1.json`，S1/S2/S3/S4/S5/S5T/S5D/S6共8500更新；原S5的2000拆成1000+600+400。双级为200mm增高/800mm首踏面，再增250mm至450mm顶面；普通双级与双级跳分别验收。相对255通信候选退役10项、保留245项、新增23项共268项；相对真实239项运行源新增39项含16通信项。不得将误生成3×4cm的旧双级成绩转成新能力信用。
- 用户已授权完成修订、验证、冻结提交后部署Kaiser接续。2026-09-29 14:09（UTC+8）已从`c15acfc`冻结部署并启动tmux，20k场景和原生接触工程验证通过；启动时先运行同checkpoint新旧合同双seed迁移评测，不能将启动回执当作新增PPO更新或技能通过。承接338消费/218谱系，不重置预算，实际进度见`reports/v6_step_curriculum_20260929/`运行回执。已验证轮侧响应保留，腿侧响应仍未qualified，不以课程更新声称六轴电机曲线均已通过或解除RMCS硬件能力门禁。详细边界见`docs/V6_STEP_TRIGGER_AND_HEIGHT_ESTIMATION_20260929.md`与`docs/V6_STEP_CURRICULUM_20260929.md`。
- 单束跳跃触发的拟定外参为前0.10m/下0.20m、下倾0.55rad，FK/IMU双侧假设支撑面用于15mm地面回波排除，控制消费投影前向距离。它们是工程候选而非实物标定；任意站高/姿态的可见性未保证，无有效回波不触发，manual入口保留。
- 2026-09-29用户明确完整课程必须考虑非常规倒地与插值脚本后的RL接管。现`c15acfc`只有S3速度扰动，无完整接管状态库；R0不得作为永久排除项而仍称完整能力交付。补强设计见`docs/V6_SELF_RIGHTING_HANDOFF_CURRICULUM_20260929.md`，尚待V6原生RMCS适配、14姿态/台阶复验、状态前史采样和新合同。当前RMCS自起200Hz算力矩/1kHz重发，与目标1kHzPD须明确对齐；不要把历史105°编码器零参考、旧110°机械域的报告移作V6通过信用。

- 2026-09-29最新授权覆盖上条RMCS适配前置：暂不修改RMCS，先实现训练原生在线恢复前段、混合后RL接稳与后续运动；脚本/混合区间须排除PPO并保留物理/通信/响应前史。新候选v6_recovery_curriculum_v1仍保持20k、8500及35/81网络，验证/提交后命名迁移，不热改c15acfc进程、不退款。实际部署身份以新回执为准。

- 2026-09-29 17:02（UTC+8）恢复课程3051c6d已冻结部署，tmux `v6-new-asset-20260929T090222Z-7aef2a`。旧c15acfc正常封存于378消费；从同合同保留检查点218学习谱系恢复，首PPO worker显式LR半减，Adam动量/RNG保留，未退款。17:04实读为同检查点新旧合同双seed固定评测，新增PPO更新0，不能将启动当成已续训更新。最终源码20k探针通过，54×2工程恢复65/108通过；台阶脚本/留台过程教学仍未完成，54项最终验收不减。未来recovery成绩在引入阶段前只观察，不自动成为当前保护项；actor拥有的终止转移保留惩罚，脚本/混合继续排除，提前解锁真值不足立即明确失败。当前合同 `contracts/v6_recovery_curriculum_training_v1.json`，8500/20k/35D/81D不变；部署/探针/未通过域见 `reports/v6_recovery_training_20260929/`。RMCS未修改。

- 2026-09-29 22:30（UTC+8）按用户授权加速评测调度，冻结源码 `7e7e637` 已部署到 Kaiser tmux `v6-new-asset-20260929T142940Z-201411`。同合同续跑，8500/20k/35D/81D/1kHz物理与50Hz策略不变；常规覆盖当前目标∪累计保护∪未完成响应复习，S5保留零离地诊断项；每跨阶段内200消费更新及晋级/预算结束全量审计，确认按候选/新通过/退步/复习需要执行。S1实际45项对照单seed从21.66min降至10.87min，44/45判分一致，320mm漂移边界差异保留失败，不放宽判据。原3051c6d运行在498消费封存，既有stand_350mm退步已触发218谱系保留点回滚及LR半减；继续该恢复事务，不重置尝试、不退款。22:38实读累计消费500、谱系220，actor/critic/optimizer精确恢复、实际LR约1.46319e-4，原合同基线复用。完整方案与回执见 `docs/V6_FAST_TRAINING_SCHEDULE_20260929.md`、`reports/v6_fast_schedule_20260929/`。训练场景仍每40更新重建，未实现常驻worker；不能把评测快一倍表述为整个训练快一倍。TensorBoard别名 `v6-fast-schedule-7e7e637`。

- 2026-09-30补救审计：上述运行已于01:49（UTC+8）在S1消费618/actor谱系338封存，`regression_hold_best_preserved`，三次保护退步且既有一次恢复用尽；S1双seed均4/16通过，静站/停车精度未达标，不等于不会站立行走。运行中实际是dense_tracking，旧两处stationary_width修改对最终reward无效；stationary_tracking未配置，anchor未实例化。恢复LR初值1.46319e-4在首PPO更新结束被adaptive升至4.93827e-4，不可再声称半减持续生效。用户要求补救计划，并提出高速放在平地扰动/非常规姿态接管之后、台阶跳跃之前；候选为原S3拆400+600、高速夹在中间，8500总预算不变。详见 `docs/V6_S1_REMEDIATION_AND_ORDER_20260930.md` 和 `reports/v6_s1_remediation_20260930/`。本轮只审计、写计划、执行42项既有CPU测试；未改训练合同、未部署A/B或重启PPO。未来迁移需显式保留618消费、保护账本和retry历史，旧200Hz reward-resume入口不可强行用于1kHz dense修订。

- 2026-09-30用户授权“部署训练吧”后，S1有界补救源码 `390850f` 已冻结部署于 Kaiser tmux `v6-s1-repair-20260930T035202Z-75babd`，合同 `contracts/v6_s1_remediation_v1.json`，TensorBoard别名 `v6-s1-repair-390850f`。两臂均从618封存检查点完整恢复，A固定LR1e-4，B同LR且仅将生效dense静止权重1.5/.75翻倍，尺度不改；每臂10 critic-only+30 actor/critic，合计80全部计入原S1，618→698，不返还弃选/失败消费。原保护35项、既有恢复用尽记录、1000/8500上限均保留，源stage持久claim防止重复领取。45项×双seed固定评测后封存待判决，不自动正式续跑或晋级。12:00（UTC+8）实读消费621（新增3次critic-only）、actor谱系338；GPU87%/13004MiB，actor/critic/Adam精确恢复，实际LR1e-4、actor KL0；20k场景运行时预检和784个诊断tag写入通过。尚无新增验收成绩。后续S3拆分/高速夹入顺序仍为未部署候选，不因本次S1启动而声称整套重排已实施。回执见 `reports/v6_s1_remediation_20260930/deploy/`，原始计划与执行边界见 `docs/V6_S1_REMEDIATION_AND_ORDER_20260930.md`。

- 2026-09-30 16:00复核：上述S1 A/B已于13:42正常封存为`remediation_review_required`，总消费698/actor消费678，两臂各40（10critic+30actor），LR全程1e-4。45项双seed均通过源618/A/B=30/33/32，S1=4/7/7（共16）；九固定高度双seed漂移均值22.81/12.16/11.75cm，停车最坏速度26.85/23.96/20.47cm/s仍未达3cm/s。A的230mm静站新增通过；B该项未过且yaw/高度扫描更差。两臂共同回归stand_250mm高度与rotate_1__payload_light的yaw，未选优/晋级/自动续训，当前无PPO，不能沿用启动时“正在预热”的描述。S1剩302，A可作下一轮候选对照而非已通过模型。证据`reports/v6_s1_remediation_20260930/result_comparison.json`及`deploy/README.md`。

- 2026-09-30 16:24（UTC+8）按用户“修改并接续”授权，从冻结源码 `8c42db2` 部署 S1 接续 tmux `v6-s1-continue-20260930T082414Z-6699c9`，合同 `contracts/v6_s1_continuation_v1.json`。从 A_fixed 完整恢复，原权重/采样、fixed LR1e-4、20k/35D/81D/1kHz/50Hz不变，不重复critic预热；40次全部actor+critic计原S1消费698→738，actor消费678→718，actor学习谱系368→408。旧35保护与A双seed已过并集39，旧retry历史保留，持久claim防重复预算。40次后45项双seed复核，满足目标/保护才补322项双seed全量审计，封存待判决，不自动花完剩余262或晋级。16:32实读消费700、新增2次actor更新；恢复actor/critic/Adam逐项一致，699检查点8个actor权重/偏置张量均变化，GPU92%/13088MiB。TensorBoard别名`v6-s1-continue-8c42db2`；回执 `reports/v6_s1_continuation_20260930/deploy/`。这是实际PPO学习，尚未有本轮新增验收结果。四腿响应仍为名义先验，PD160/2.5已用但bag拟合候选未接入，不能把轮侧已接入称为全部真实曲线完成。

- 2026-09-30新录包计划见`docs/V6_LEG_BAG_RECORDING_PLAN_20260930.md`。用户要求广域及跳跃覆盖：先左→右→双侧，固定160/2.5与50Hz参考/1kHz PC PD，实际beta分箱和独立几何核验，深屈曲/快速蹬伸/收展/制动/连续相位；架空、带载、真实接触分别记录适用域。新轨迹、腿manifest及四轴入口尚待实现；现有start-pd-loaded仍为旧轨迹，不得宣称方案已部署。记录模板不是可执行profile；该文档提交不改变Kaiser当前训练或RMCS代码。

- 2026-09-30 18:04只读核验：8c42db2接续40次已于16:58完成，17:19双seed评测结束并封存`continuation_review_required`，当前无PPO/评测进程。消费738、S1余262，LR全程1e-4；45项通过A→接续33→29、S1 7→4，均双seed一致。无新增通过，失去stand_230/350/380mm及forward_05_height_230mm；九档最大漂移均值12.16→22.66cm，停车最坏23.96→22.72cm/s仍未达3cm/s。S1存活率均100%，是保持/精度回归，未晋级/未全量出口审计；不应继续把遗留问题定性为仅训练量不足。A源与738检查点均保留，下一步修改尚未部署；见`reports/v6_s1_continuation_20260930/result_review/README.md`。

- 2026-09-30源码与TensorBoard课程审计：698→738没有高度份额切换，stage actor时钟368→408已过200边界，九固定档每个batch实际460–528环境。新发现普通静站RECOVERY持续落地正奖励、成功退出反而少奖励的缺陷；原生相位／奖励探针中11mm误差比9mm多2.605 reward/s，230mm末期实际仍支付落地奖励。已证明激励漏洞，未用PPO消融证明全部回归原因；350/380mm需继续归因。另有S1运动仅305mm与保护联合域不齐、停车相位同步、40更新19.2s短worker及episode总reward不含未结束静站回合的问题。详见`docs/V6_S1_CURRICULUM_CAUSAL_AUDIT_20260930.md`；只提交审计/探针，未改训练合同或部署，余262。不得重复“华南虎无阶段所以不遗忘”“九档退采样”“只剩训练量不足”的结论。

- 2026-09-30用户“训练吧”授权新合同`contracts/v6_s1_landing_repair_v1.json`：从A学习状态完整恢复，最新消费账本738另行认证；40次actor+critic计738→778，actor消费718→758，actor谱系368→408，LR1e-4/20k/35D81D/50Hz1kHz不变。只修被动落地奖励：普通静站无落地bonus，显式落地用独立0.6s窗口零中心成本，旧合同行为保留，物理/观测/命令/RNG不变；联合配额、停车相位和常驻worker尚未变更。257项相关测试及5种入口物化合同静态预检通过，运行时另断言实际实例与窗口时钟。新消费claim建在738预算源；原39保护、retry历史及8500/1000上限保留。40次后45项双seed评测封存待审，不自动放宽gate或花完余量。计划`docs/V6_S1_LANDING_REPAIR_20260930.md`；是否实际PPO启动以新deploy回执为准。

- 2026-09-30 19:18（UTC+8）修复版源码`62c451d`无overlay部署Kaiser tmux `v6-s1-continue-20260930T111823Z-58c545`。19:26实读已新增3次PPO、累计741、actor谱系371；739检查点8个actor MLP张量及std改变，actor/critic/Adam精确恢复、LR1e-4；20k原生runtime断言及新增奖励实例通过。静站含230mm两项落地奖励前三次均为0，证明修复已启用，尚无新技能通过结论。GPU93%/13088MiB；TensorBoard别名`v6-s1-landing-repair-62c451d`。完整部署及运行证据见`reports/v6_s1_landing_repair_20260930/deploy/`。当前正训练该40更新分支，旧738接续仍封存，不得将旧线空闲状态当作当前训练状态。

- 2026-10-01核验：落地修复40次PPO于9月30日19:52完成，20:13双seed评测完成，正常封存`continuation_review_required`；当前无该任务PPO／评测，消费778、余222。32个普通行为落地奖励tag全40次为0，修复生效；45项27/28、S1目标4/5（共16），低于A的33/33及7/7。230mm较未修复接续改善但漂移仍22.75–23.66cm，停车实际驻停23.66–23.96cm/s（阈值3），rotate_1 yaw约0.188（阈值0.15），不能称全部贴线。738与778是同A起点的两个40更新兄弟分支，不是顺序权重接续；两seed是评测重复，不是两次独立PPO。相位同步／短场景生命周期／联合覆盖仍未改，2.5%静站份额不足未被证明。未晋级、未选新部署模型、未重启或修改合同；最新结果和建议见`reports/v6_s1_landing_repair_20260930/result_review/README.md`。

- 2026-10-01用户授权继续训练并持续跟踪：新合同`v6_s1_persistent_recovery_v1.json`从A恢复完整学习状态、最新778账本计费，最多222次常驻PPO至S1=1000；首步779及每10消费更新封存，不再每40销毁场景。保留落地修复，新增停车真实0–8s零速等待后执行完整16s序列，评测／旧合同不变；生命周期与时序同时改，不称单变量研究。20k/35D81D/50Hz1kHz/固定LR1e-4、轮侧响应和39保护不变。860/940旁路45项单seed在资源满足时运行，最终双seed短套及322全量出口，未过不得晋级或超预算。每15分钟当前对话heartbeat `kaiser`持续处理，不再因正常检查点要求用户发令；跨阶段须先冻结继承完整的新合同，不能直接启动旧S2。计划`docs/V6_S1_PERSISTENT_RECOVERY_20261001.md`；实际部署／PPO证据以`reports/v6_s1_persistent_recovery_20261001/deploy/`为准。旋转vx是平移泄漏，yaw是角速度MAE；正反已各约1000环境，左右对称库仑阻力差不能直接证明旋转方向差异原因。

- 2026-10-01 01:43（UTC+8）已从`f7e041b`无overlay部署Kaiser tmux `v6-s1-continue-20260930T174327Z-45dd84`，常驻222正式PPO；01:51实读新增2、消费780、actor谱系370，779封存9个actor张量改变，A actor/critic/Adam exact且LR1e-4。实际0–8s停车时序与.6s落地修复运行断言通过，GPU90%/13088MiB。旁路860/940观察器已在同tmux独立窗口启动。真实Kaiser TensorBoard入口现为本机6007（6006仍是旧本机实验）；别名`v6-s1-persistent-f7e041b`，每15分钟heartbeat `kaiser`已启用。当前训练在进行，无新固定评测成绩；回执`reports/v6_s1_persistent_recovery_20261001/deploy/`。

- 2026-10-01 03:17（UTC+8）860旁路45项单seed290929评测完成20/45、S1目标6/16；A同seed33/45，旧778为27/45。驻停最坏23.96→13.46cm/s仍高于3cm/s，320新增及350/380静站恢复，但正向0.5实际仅0.390m/s，负载/通信变体共同欠速掉分；正向yaw0.1903未过。180回合failure终止0；低姿态前进2/4场地边界截断，不能称摔倒。settings、case顺序和startup核对一致，860等待runtime关闭，未发现已证实实现错配；未单独做同checkpoint GPU重放。主线03:24实读894消费／actor谱系484，仍运行冻结222预算，继续940观察和1000出口，不热改、退款、放宽或晋级。原始评测、SHA与对照见`reports/v6_s1_persistent_recovery_20261001/observations/update_00000860/`。

- 2026-10-01 04:26（UTC+8）940旁路完成17/45、S1目标2/16（backward_05／rotate_1）；正向yaw0.1274通过，反向0.3047退步，停车最坏12.12cm/s未达3cm/s，前进0.5实际0.354m/s，九档静站联合检查全未过（低中档多为高度超5mm）。所有survive满窗，180回合failure与boundary均0；不得用全局rank当跌倒率。940与860完整settings/startup/source manifest相同，未发现新实现问题。观察器两点正常完成退出；04:40主线983仍PPO，随后自动45×双seed并无条件322×双seed出口，即使quick failed也不跳全量。预算仍1000／8500，未改合同或晋级。证据`reports/v6_s1_persistent_recovery_20261001/observations/update_00000940/`。

- 2026-10-01 04:52（UTC+8）常驻PPO正常完成222更新，累计消费1000／actor消费980／actor谱系590／总学习谱系600；S1更新预算用满，不得继续额外PPO或退款。222条TB LR全1e-4、scalar全有限、落地正奖励全0；model_final与update1000各自seal哈希通过。04:52:37自动转入45项seed290929出口评测，后续第二seed及322项双seed仍待完成；当前阶段是fixed_evaluation而非PPO或晋级。主调度tmux仍运行，观察器已经正常完成退出。预算出口证据`reports/v6_s1_persistent_recovery_20261001/deploy/ppo_completion_verification.json`；不要把940成绩当1000最终结果。

- 2026-10-01 05:03（UTC+8）消费1000出口首seed290929通过20/45、S1目标4/16；05:09确认seed290930运行中，322项双seed未完成，主调度尚未封存最终门结果。首seed驻停10.69cm/s、前进0.5实际0.325m/s、正/反yaw0.1393/0.4444rad/s；305mm与连续高度静站恢复，但不具备晋级信用。评测model_final哈希与seal一致，settings/startup/source manifest与940相同。证据`reports/v6_s1_persistent_recovery_20261001/exit_evaluation/`；消费仍1000，无新增PPO。

- 2026-10-01 05:28（UTC+8）核验：1000出口45项双seed均已完成20/45、S1目标4/16（stand_305mm、stand_continuous、backward_05、rotate_1），通过名单一致，39保护各通过18。驻停10.69/11.44cm/s、前进0.5实际约0.325m/s、反向yaw角速度MAE约0.444rad/s；不是全为贴线。两次各180回合failure与boundary均0，不代表任务全成功。settings仅seed、startup仅预期seed/角色不同，source manifest和模型哈希一致，TB全有限。主调度仍运行，322项首seed评测进行中、第二seed待执行，未产生最终completion。消费1000不再增加，不得晋级、退款或另开PPO；结果见`reports/v6_s1_persistent_recovery_20261001/exit_evaluation/`与`deploy/exit_confirmation_verification.json`。

- 2026-10-01 05:39（UTC+8）核验：322项seed290929全量于05:37:28完成67/322，S1目标仍4/16、39保护仍18通过，共同45项定义与短套相同且通过判定一致。全量包含后续未训练技能，不把整体67/322当S1成绩或后续晋级信用；1288回合记账完成，failure634、boundary75，共同45项仍无failure/boundary。第二全量seed290930已由原调度自动启动PID2099877，tmux与主调度存活，最终completion待产生。消费1000不变、无PPO、无合同修改，原始报告和SHA/TB核验见`reports/v6_s1_persistent_recovery_20261001/exit_evaluation/`与`deploy/first_full_verification.json`。

- 2026-10-01 06:09（UTC+8）最终核验：本轮于06:01:19正常封存`budget_exhausted_gate_pending`；45项双seed20/45，322项双seed67/322、66/322，S1均4/16、保护均18/39。四套共同45项通过名单一致，必需集合25失败独立重算与completion相符。所有评测已完成，pending指条件未满足，不是仍在评测；进程和tmux已正常结束，禁止作为故障重启。消费1000／actor消费980、actor谱系590／总谱系600；S1余0，24个不可变检查点与旧A／最终模型哈希通过，无accepted/deployment模型，无晋级。当前没有新证实配置或判分bug；后续先离线归因，新增PPO需要明确S1预算调整授权及冻结新合同，不能自动借用总8500余量。结果与未解决项在`reports/v6_s1_persistent_recovery_20261001/exit_evaluation/README.md`，封存回执在`deploy/completion.json`和`final_verification.json`。状态不变的heartbeat保持安静，不重复重启或通知。
