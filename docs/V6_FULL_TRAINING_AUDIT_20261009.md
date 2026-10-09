# 完整训练审计与接续改进

日期：2026-10-09（UTC+8）。审计对象为实际 source023/v21 恢复训练、误派发后完整封存 REC actor14188，以及已实际进入speed PPO的source029同 v26 单模型串行接续。区分实际运行、冻结候选和建议；本文件不是新的训练合同。

## 结论

现有训练值得保留，但尚不能交付“全部功能已学会”的一个ONNX。主要问题是任务曝光、交接条件与难度安排没有形成完整可核验的教学链；不能笼统归结为MLP太小、训练轮数太多，或没有漂移/姿态惩罚。

当前正确的结构仍是同一个actor顺序进入速度→地形→跳跃→跳阶，每段保留已有能力采样。串行改变的是本段重点，不应把旧能力从后续数据中删除。复习份额只能保证访问，不能保证不遗忘。

## 实际状态与效果

2026-10-09 03:37（UTC+8）SSH核验：旧REC完整封存于14169，消费41374、actor消费41204；本任务无PPO owner。source028启动前检查漏传显式learning-rate-scale，解析为None而认证要求1.0，在创建Simulator前exit1；不是新的物理训练失败。原检查点SHA ed24fe4e5e62fd8c8e01f81427c0741ae46799a237dee2b0fbd4b288df4e1085保留。随后一次命名修复完成真实原生场景：4096环境、1策略/4物理步、0 PPO全部通过。启动适配器误读report.json而原训练写completion.json，导致exit0仍标失败；已取得真实completion SHA并独立验证21项，采用单独冻结的结果认证，不覆盖失败回执或重复建场。首次新生产worker实际仍选中了旧REC attempt010，而不是期望的speed；根因为主调度按六域循环，未跳过已认证提前封存的REC。只读检查发现后，向准确supervisor发送一次SIGTERM，最新REC14188、累计消费41393完整保留，随后仅外层wrapper归档。该19次仍是恢复PPO，不应写成综合速度已开始。选课修复与实际再接续以独立冻结回执为准。

最后已核验完整100更新窗为REC13726–13825，不能称为封存终点的最后100次，也不能当确定性实车表现：

| 指标 | 训练统计 |
|---|---:|
| 230/305/430mm静站平面均速 | 3.42 / 2.68 / 2.94cm/s |
| 对应高度MAE | 2.22 / 2.47 / 6.39mm |
| 前/后约0.5m/s指令的实际均速 | +0.407 / −0.359m/s |
| 前/后约1m/s指令的实际均速 | +0.748 / −0.628m/s |
| 正/反yaw角速度MAE | 0.131 / 0.125rad/s |
| 停车条件平面均速 | 3.267cm/s，35.27%样本超过3cm/s |
| 恢复混课完成率 | 686/7608=9.02%，前窗11.62% |
| 有效PPO样本 | 85.39% |
| 每更新wall / 环境 / optimizer | 19.170 / 16.013 / 0.127s |

9.02%包含容易出生、短零力矩后接管、原恢复链，不能当纯倒地自起率。最新Gaussian动作std约0.0525，早期0.5已不是当前数值。最新静止优于历史强探索阶段，仍欠速、停车长尾明显，恢复完成率未形成稳定提升。最新均值策略同条件完整评测仍缺；本次已另做两个候选的左侧90°恢复与平地加速/停车短回放，见下文，不能用该单初态替代完整评测。

## 串行接续修复的最新部署证据

最终 source029 已完成stage003、CPU preflight/check-only、一次origin更正及一次原wrapper launch，并已真实进入speed PPO；2026-10-09 05:20（UTC+8）首actor只读核验为passed。源码提交 `e5bd9c1367b875d7f229b9a1930647013d023372` 直接以实际 source028 `335542ddc753b84b2778609b92803064682198a9` 为父树，最终归档SHA `7cb0a586a1c83f49325a244d3fe8793c12117ac3bedf69facafd9d06bbc57bb0`，4313个文件。真实树对照仅八个 dispatch/origin/alias 修复文件变化，276份合同与4305个非owned tracked文件的字节和git mode保持；`env.py`、机械资产、reward、物理PD、轮响应与预算不变。

早先 `55ead9e` 候选误以较新本机主树为基础，带额外合同与 `env.py` 等变更；原 helper 在预检的全合同比较处拒绝。其暂存源码、tar、source_revision与stage目录已完整移入远端 `source029_staging_rejected_archive001/`，保留原字节、intent和receipt。拒绝及隔离均为0 PPO、无origin更正、无launch，原账本和最新REC41393完整状态SHA `e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967` 保持，不把拒绝候选写成已部署。

同一最终提交的stage002随后因本机git默认tar权限掩码被origin检查拒绝：4310个共同文件字节相同，但4306项0644→0664、4项0755→0775。该源码、tar、revision及包含preflight/origin日志的stage目录同样完整隔离到 `source029_staging_rejected_archive002/`，仍0 PPO、无origin更正、无launch。只用显式 `tar.umask=0022` 重打同一e5bd9提交后完成stage003；代码、合同和拒绝记录不改，也不放宽权限守卫。

最终修复同时解决三个准入问题：只派发 speed→terrain→jump→stepjump，保留flat/REC历史；以一次性认证事务把误跑19次actor之后的最新REC14188作为来源，全部消费照计；新执行源码目录通过同合同source-revision回执认证，`FrozenPlan`仍读取原不可变manifest根，不重签合同。真实双目录CPU生命周期已通过99.81s：启动前旧job、wrapper保存实际resume024后旧job/receipt配对、更正后open_account幂等、latest父来源、新目录train命令、原生worker admission及speed首派发；错误留存job和合同改动拒绝。它是准入测试，0仿真/0 PPO，不证明新训练效果。

真实launch于05:12:11（UTC+8）提交，唯一owner链为wrapper4141407→supervisor4141412→speed worker4141576，目录 `train/speed/attempt_003/train`。当前worker自己的原生startup与runtime配置检查通过，4096环境、50Hz策略/200Hz物理反馈PD保持；此前REC14169的一策略步场景只作为不变工程先验，不冒充新REC14188恢复验证。

真实父来源为REC41393/REC14188/SHA e04c…3967，初态9个actor张量、8个critic张量和17组Adam状态精确恢复。首critic封存41394时actor/std与actor课程时钟保持；首actor封存41404时九个actor张量发生改变，task/joint actor=1、完整学习谱系27657，10次critic全部计费。中间critic第2–10次没有逐次不可变张量封存，不声称每次单独核读；新回合重置后的RNG也不声称与父环境逐位相同。

首actorreader核验窗口的进度为新增13次更新＝10critic＋3actor、speed task3、stage27559；账本消费41406、actor消费41226、剩79594。进度与账本为分开只读采样，不是原子快照。原41393入场和全部19次误派发消费保留，无退款、无重复REC补跑。后续仍是同actor的speed→terrain→jump→stepjump串行，复习配方不变，不以启动成功授予新高速/跳跃/台阶能力或qualification。详见[部署摘要](../reports/v6_full_training_audit_20261009/SOURCE029_DEPLOYMENT_SUMMARY.md)及[首actor恢复核验](../reports/v6_full_training_audit_20261009/first_actor029_001.json)。

## 当前需要补强的内容

### 当前高速段的早期耗时证据

2026-10-09 05:31:54（UTC+8）只读进程/进度/账本核对，source029唯一owner链保持4141407→4141412→4141576。本段已完成43次更新＝10次critic适配＋33次actor更新，消费41436、actor消费41256、余79564；进度和账本是分开读取，不声明原子快照。这只证明正式高速PPO持续运行，尚不是新高速能力评定。

从Kaiser现有TensorBoard服务读取当前 `speed/attempt_003/train`，共同完整20步41410–41429的host独占均值为：wall23.603s、环境18.300s、诊断0.728s、原生logger0.464s、optimizer0.126s、progress I/O0.015s、returns0.010s、未归属3.959s。该20步没有检查点写入耗时。此为早期短窗，不能与旧REC100窗直接归因为退化；异步CUDA下这些数值也不是各GPU kernel耗时。

当前run已出现31811个scalar tag。源码中LearningAudit逐tag写event的循环位于已计时的diagnostics/logger之外，因此它是未归属开销的具体候选；**tag数本身不能证明全部3.959s都来自序列化**。下一版先为此循环补上独立计时，再据实际占比决定降低日志开销；不以估算承诺提速，不热改当前worker。当前读取直接访问远端服务，没有扫描旧大型event、启动本机TensorBoard或运行额外PPO。

原始证据：[实时owner与账本](../reports/v6_full_training_audit_20261009/source029_live007.json)、[TensorBoard早期记录](../reports/v6_full_training_audit_20261009/source029_tb001.json)、[同一步耗时归约](../reports/v6_full_training_audit_20261009/source029_tb001.summary.json)。

### 恢复结束回合口径的更正

实际源码已提供 `RecoveryOutcomes/{prefix_stand,prefix_release,original_chain}` 的累计ended/successes/failures/collection_timeouts；这些计数按每回合实际采样的lesson，在自动reset之前读取producer结束标志且去重，不是帧均值。此前“纯L2分母缺失”需要细分：**L2原链课程的结束分母已经存在，只是旧窗口报告未提取；真正倒地姿态子集的结束率仍未给出。** L2保留各组原始出生，而部分原始组为直立、释放或平台站立，因此原链课程也不能整体冒称四种倒地自起。

本speed worker截至TB41444的累计producer回合为：L0容易接稳38/180＝21.11%，L1零力矩释放接稳37/298＝12.42%，L2原链35/485＝7.22%；各级成功、失败、采集超时之和与结束分母相符。它包括worker的critic适配期间，并非仅新增actor学习窗口，也不是确定性能力成绩。对应最新rollout全batch有效PPO比例92.93%，不能转写成每种恢复lesson都有92.93%的学习曝光。下一版复用这些结束计数，仅补原生脚本完成、首纯RL、连续接稳、放行的事件漏斗及最终policy mask按lesson的曝光。

证据：[现有计数的同一步归约](../reports/v6_full_training_audit_20261009/source029_tb002.recovery_summary.json)。

| 项目 | 当前证据 | 改进方向 |
|---|---|---|
| 普通大台阶 | v26普通上阶池改为5/10/20/40mm三阶上下8组。200mm+再250mm的普通搭台收腿组已不在该实际训练池；台阶上恢复/跳阶仍有 | 小阶是已授权的优先课，合理保留；在完整计划中显式补回普通大台阶的后续难度层。与主动跳阶分别训练/验收，不能互代 |
| 恢复交接 | 54恢复组+41基础复习组；成熟L0/L1/L2比例20/30/50。已有真实lesson结束分母；完整链进入RL后还需连续1s接稳才能放行后续运动 | 复用结束分母，补脚本完成、首接管、连续稳定段、放行与运动曝光；真正倒地姿态子集另行报告。失败定位到脚本、交接、接稳、运动哪一段 |
| 恢复条件的地形适用性 | support/body_clear使用FK/IMU复合估计而非直接真实接触；左右高度差等条件可能在跨台阶姿态中更苛刻 | 先用同帧原生接触/净空与估计对照，证明误判后修几何/估计；不直接缩短1s或把未完成写成完成 |
| 速度/姿态联合难度 | speed的0/120/240/360 actor更新放开到2/3/4/5m/s、1至3圈/s与.23–.43m。普通连续命令映射仍主要±2.7m/s/±1圈/s，特殊组另覆盖极限 | 根据继承模型的同条件成绩和有效样本，分开扩展速度、高度、加速度、转向、延迟；高姿态×高速×强扰动不能只靠均匀全组合。360是否太快目前是待验证假设 |
| 后期复习有效性 | v26各域189/233/248/262组，4096静态分配最小11/5/3/2环境；脚本段还不进入PPO | 报告每个能力族真实可学习帧、结束回合和条件覆盖。合并重复教学规格或按宏观能力分配人口，保留固定评测锚点；不要只增加更多细碎组 |
| 普通运动无指令离地 | 普通mode1旋转不触发mode4跳跃请求，air专项支付也有模式约束；dense普通vertical_velocity目前固定为0 | “旋转时跳”尚不能归因状态机误触发。先记录轮接触、vz、腿速度、力矩/饱和和实际命令；普通运动vz代价可做单项候选，不施加到合法跳跃相位 |
| 跳跃定性→定量 | 已有辅助概率/力度退火及顶点bin课程；跳高、垂速、收腿reward核宽仍固定 | 加入明确actor时钟的宽→紧候选，宽核覆盖实际未收腿/未达到顶点的误差，后期精调，同时保留大误差可区分的成本。先用误差/付款分布定宽度 |
| 负载与质心鲁棒性 | 本项目COM x±1cm/y±.5cm/z=0，base质量.9–1.1 | 按实际载荷/装配误差扩大并渐进加入；不要把较宽范围直接视为一定更好。腿侧bag仍未qualified，是实车响应可信度缺口 |

原地跳、行进跳、跳阶已在代码/课程中存在，但存在课程不等于完成离地→收腿→落地→接稳。需要分别核验实际跳跃请求、完整回合和落地接稳，配置存在不能代替实测成绩。

## 高速与台阶的具体安排

1. 速度课持续混采静止、低中高速、正反旋转、弧线、加速、松杆制动、反向切换，并覆盖低/中/高姿态。稳定速度时要求良好姿态；加减速阶段要允许机构必要的瞬态倾斜，并检查加速度/jerk与恢复水平的时间，不能只以全回合平均pitch掩盖制动峰。
2. 保留平地足够跑道；已实现flat半长160m、半宽8m，普通地形越障后的正向边界5.6m、支撑延伸6.2m。场地已扩展，不应重复把当前问题归因小平地。
3. 普通小阶先做5–40mm上下、坡、粗糙面与障碍接触；用地形×命令表限制为有用的直进/退、合适高度和小yaw。source031已冻结八行普通单阶/双阶渐进配置：单阶50/100/150/200mm，双阶最终再升200+250mm、首层800mm踏面；仅连续向前速度、mode0、无STEP/JUMP触发。它们在terrain的有效recipe内逐级解锁，并在jump/stepjump保持复习；新版本仍等正常speed边界，实际原生准入和完整爬阶效果待验证。
4. 主动跳跃按离地、收腿、落地吸收和站稳分项训练/记录；跳阶最后增加平台边缘定位与连续两级任务。正常爬阶不需要JUMP请求，跳阶使用明确状态机请求，两者定义保持分离。
5. world-reference旋转平移每步重算body命令，部分路径绕过普通命令投影。世界系匀速与body-vx变化不同，不能直接用vx×yaw判定错误；需记录完整坐标系命令、轮速、滑移和离地做专项可达域核查。

## 历史问题的原因与仍未证明的内容

已证实：普通静止错误领取落地正奖励；adaptive LR恢复后超出期望；短worker和停车同相位导致曝光集中；特定恢复脚本接触冲击及闭链误差；部分原生地形四元数接口错配；本次native CLI参数漏传、completion文件名读取不符，以及串行主调度在已认证REC提前封存后仍重跑旧REC的选课错误。对应修复/负证据已有记录，不能把它们重新列成当前全部仍未修。

历史同条件均值策略对照确有部分运动/yaw回归，停车同时改善，说明存在任务间权衡。当前两个不同难度/噪声窗口的统计不能独立证明遗忘，更不能断言“继续训练自然会全部收敛”。历史A/B的两个seed是评测重复，并非两次独立PPO训练。

静止宽核、精核与L1速度代价，姿态、yaw和接稳成本已经支付。精核在大误差区接近饱和不代表广域成本也无信号；reward对误差的导数也不是PPO直接反传的梯度。不能以惩罚数量或总reward判断收敛。增加权重需先排除错误命令/掩码、交接状态、未曝光、探索与部分可观测性。

## 训练机制的语义边界

- 预训练继承、迭代技能解锁和相位奖励均不能保证不遗忘。模型目录数量不能作为部署时切换策略的依据。
- flat_orientation_x/y_v是速度命令相关的投影重力姿态代价，不是角速度惩罚；后者另有ang_vel_xy。
- 当前actor为单帧35D；地形查询不在actor输入，不能将仿真地形信息冒充部署侧观测。
- 延迟随机化按5ms物理步推进，应以毫秒记录，不能混作50Hz策略帧。观测、动作、USB传输和执行器响应必须分别定义，避免重复累计。
- 静止命令重采样概率与固定静止环境份额是不同指标。低指令平面速度惩罚是否生效应检查实际任务配置、掩码和支付值。
- 主动跳跃源码能力、交付配置启用状态、训练轮数和实车成绩是四种不同证据。

## 接续实施优先级

**P0：当前已授权串行接续已实际运行。** source029从误派发后最新REC14188完整状态进入speed，全部19次新增actor消费保留；CPU准入、一次origin更正、唯一owner及首actor恢复已通过。固定4096、35D/81D、50Hz策略/200Hz物理反馈PD160/2.5、轮响应和原预算；继续本段并按冻结配方自动串行到terrain/jump/stepjump，不把未validated的reward建议热改到本次运行，不增加重复启动或阻断gate。新效果等待完整训练窗与旁路评估。

**P1：下一版本统一补齐可解释性与覆盖。** 基于本报告一次冻结：恢复按真实L0/L1/L2分桶、脚本/接管/放行/结束分段计数、pitch/roll与轮接触/vz统计、各能力真实PPO曝光；普通大台阶明确后续课程位置；速度×姿态×地形×制动覆盖表。复用原FSM/logger，不增45项阻断晋级。

**P2：用窄对照决定reward/难度调整。** 同一封存模型、同初态/随机化比较mean/sample；固定少量静止、停车、正反旋转、高低姿态、恢复、阶梯用例比较旧新模型。在此基础上一次只测普通vz代价、核宽退火或难度节奏中的一个。核宽取实际误差分布，不凭感觉同时翻倍所有权重。评测保持离线/旁路，不每40次重建训练。

**P3：针对实测环境时间提速。** 最新19.17s/update中环境约16.01s、优化器0.127s，优先查host同步/重复状态读取、闭链物理和日志。已有host同步合并候选尚无Kaiser实测收益。维持物理语义作短profiling和未profiling对照，不把CPU event解析时间算成PPO暂停，也不凭低利用率认定线程竞争。剩四域若各15k、速度不变约13.3天，这是条件估计，不能在缺少吞吐实测时承诺一天完成。

## 证据索引

- [历史配置语义核查](../reports/v6_full_training_audit_20261009/scut_source_comparison.md)
- [因果审计与最新100窗](../reports/v6_full_training_audit_20261009/causality_audit.md)
- [实际v26课程提取](../reports/v6_full_training_audit_20261009/curriculum_audit.json)
- [启动参数修复独立审查](../reports/v6_full_training_audit_20261009/native_startup_repair_review.md)
- [一次命名启动修复计划](V6_NATIVE_CLI_RECOVERY_20261009.md)
- [source029真实父树与276合同/非owned文件对照](../reports/v6_full_training_audit_20261009/source029_true_base_verification002.json)
- [错误暂存隔离与0PPO回执](../reports/v6_full_training_audit_20261009/source029_quarantine001.stdout)
- [source029跨源码目录准入独立审查](../reports/v6_full_training_audit_20261009/SOURCE029_PREFIX_ALIAS_REVIEW.md)
- [归档权限拒绝诊断](../reports/v6_full_training_audit_20261009/source029_mode_rejection002.json)
- [第二次失败暂存和CPU日志隔离](../reports/v6_full_training_audit_20261009/source029_quarantine002.stdout)
- [source029最终stage回执](../reports/v6_full_training_audit_20261009/source029_stage003.stdout)
- [实际CPU预检与四域dispatch](../reports/v6_full_training_audit_20261009/source029_preflight003.stdout)
- [一次origin更正执行](../reports/v6_full_training_audit_20261009/source029_correct003.stdout)
- [实际launch](../reports/v6_full_training_audit_20261009/source029_launch003.stdout)
- [首critic/首actor完整恢复与唯一speed owner](../reports/v6_full_training_audit_20261009/first_actor029_001.json)

## 本次模型交付

已从同一REC14169封存权重导出REC41374候选，ONNX SHA `63574953c97aad374fa1554c53486cd540a3019857af72e0c06cc010214299bf`。源PT与合同保持原字节、数值导出通过。它作为恢复优先候选，不可称为已验证的恢复+高速最强模型；其普通课程范围仍是|vx|≤1m/s、|yaw|≤1rad/s。新联合速度续训的权重需另行封存评估后才能替代此文件。

对照的speed19837来自speed←flat分支，不能合并其高速经历到REC；旧REC38000高速回放失稳证据保留。详见[模型选择审计](../reports/v6_full_training_audit_20261009/model_selection_audit.md)与[本次交接文档](V6_RECOVERY_PRIORITY_MODEL_HANDOFF_20261009.md)。

## 本次同条件候选回放的实测结论

REC41374和REC38000在相同原生合同的左侧90°倒地场景均5.48s进入纯RL、6.48s连续接稳后放行、7.46s完成完整链。最新完成姿态略优，但普通运动明显有取舍：0–5s加速vx MAE新/旧0.393/0.556m/s；6.34–7s实际命令归零后的33帧，世界平面均速新/旧1.965/0.083m/s、最大倾角35.43°/10.54°。最新7.06s越过侧界，非跌倒；因此后续高yaw只有3帧，不能给高yaw通过或失败结论。旧模型高yaw虽150帧，角速度MAE1.509rad/s、最大倾角28.98°也未良好。

这提供了同条件的恢复制动权衡证据；只有各一次固定出生，未定位因果，不能称综合最强或实车合格。本次优先恢复/制动回放选择旧REC38000，最新REC41374另存，不让“最新”代替选优。普通上台阶与主动跳阶各自未测，不能借左倒地或跳跃接口授予爬阶信用。

第二次回放仅增补位置和共同时间窗口；两次场地实际均为8m宽，不能把它描述成拓宽场地。累计行程不是零命令静止漂移。详见[原生同条件结果](../reports/v6_full_training_audit_20261009/model_compare/native_comparison_addendum.md)。

## 2026-10-09 首100次高速actor窗口与下一版组合

06:08（UTC+8）实读 source029 原三owner仍为4141407→4141412→4141576，本段完成135次＝10critic＋125actor；消费41528／actor消费41348／剩79472。进度与账户分开读取，仍不声明原子性。本次未写远端文件、未触发额外PPO或重启。

首100次实际actor对应TB41403–41502，前一个TB41402只作累计结束计数的窗口基线。100点完整且有限，静止230／305／430mm组的平均世界平面速度3.19／2.53／2.13cm/s，目标高度平均绝对误差3.02／3.08／5.88mm；start_stop_1实际驻停窗66980帧的条件速度2.58cm/s。它们是探索和域随机化下的训练统计，不能称确定性10秒漂移、机械或实车通过。

本版连续命令映射会给forward_05/backward_05/rotate_1重采连续vx与yaw；这些组名不再代表固定±0.5m/s或纯旋转考试。首100点前后组平均实际命令vx为−.092／−.039m/s，实际body-vx均值+.122／+.113m/s；差值是带符号偏置，不能称速度MAE，更不能拿它们与历史固定命令评测直接比较。当前缺该窗口的逐帧绝对误差和确定性高速评测证据。

现有producer累计计数作窗口差后，L0／L1／L2成功分别47/387＝12.14%、60/638＝9.40%、47/1065＝4.41%；各级结束分解与success/failure/collection_timeout吻合。L2仍包含非倒地出生，不授四倒地整链完成率。全batch有效PPO帧比例95.52%，不等于各恢复组都获得该比例。

100更新wall均23.791s：环境18.516s、诊断.743s、原生logger.442s、optimizer.127s、未归属3.939s。此前CUDA报告块的57个实际AST等价例已通过，局部中位耗时旧5.223ms／新1.759ms；它只验证小块，不证明整更新三倍提速。新状态快照候选四字段批量取回和专用state_publish计时均已实现，尚未远端采用。

隔离组合cda955e纳入普通大阶、恢复事件审计、host批读取与快照发布，148项必要CPU检查通过（40.41s）。独立审查发现step_climb.py未列入worker逐文件来源清单，root仅补v27清单和断言并冻成0d6294b；该修复1项CPU／27.26s通过，独立回执002确认闭合。v27静态配方保持旧24–26及speed字节，future terrain/jump/stepjump显式打开恢复阶段审计；普通大阶从单50mm逐步到双200＋250mm，仍为无跳请求的前向爬阶。正式completed-speed→terrain准入、真实父PT／预算／引入clock、原生场景验证与部署尚未完成，不将静态0/15000当实况，也不扩大source029八文件alias装入新功能。

证据：[首100actor完整归约](../reports/v6_full_training_audit_20261009/source029_tb100_001.summary.json)、[对应只读活进程](../reports/v6_full_training_audit_20261009/source029_live010.json)、[组合与验证边界](../reports/v6_full_training_audit_20261009/source029_next_candidate_composition002.json)。新普通阶与主动跳阶分别记录覆盖、姿势和成功，不能互代。

### 行进高跳覆盖更正

真实e5/v26的TrainingSchedule.sample_jump不是只按组名里的03/06/10cm训练。三条running行在jump component actor0/2000/6000均能采3–30cm的七个bin，其中15–30cm条件概率25%／55%／80%；180000个CPU CDF目标与真实SkillCommands→FullTask目标接线吻合，critic-only时钟保持。固定前进仍为0.4m/s，水平目标0.15m。stepjump后段中running_03/06继承jump clock，running_10由stepjump拥有，不能把jump阶段的同钟结论泛化到所有后段。

因此不增加重复的行进高bin课程。剩余工作是实际bin曝光、无辅助整链成功和高bin固定评测，另行扩大行进速度域；CPU可采不是动作已学会。更正与复算见[采样核验报告](../reports/v6_full_training_audit_20261009/RUNNING_JUMP_SAMPLER_COVERAGE_CORRECTION.md)。

## source029 第二100actor窗口：静止与恢复复习回落

日期：2026-10-09。真实 Kaiser source029 / speed attempt_003 的现有 TensorBoard HTTP 标量；不扫描event、不更新优化器、不修改远端项目。冻结源码 `e5bd9c1367b875d7f229b9a1930647013d023372`。

这两个不重叠窗口是带探索/域随机化的训练读数，不是同checkpoint确定性评测，也不是实车或功能合格证明。第二窗静站/停车与恢复复习下降是需要跟踪的质量信号；当前合同未改、未停机、未退款。


三档静站平面均速3.19/2.53/2.13→4.11/3.06/2.82 cm/s，停车条件均速2.58→2.99 cm/s。L0/L1/L2 ended完成分别47/387、60/638、47/1065→2/485、4/770、2/1187；L2含非倒地出生，未授倒地链通过。每更新wall23.79→20.21s、全batch有效PPO份额95.52→95.58%，不是未部署优化因果。已保存完整两窗和分母；下一实际500完整seal做同条件均值回放。详见 `reports/v6_full_training_audit_20261009/SOURCE029_FIRST_TWO_SPEED_WINDOWS.md`。

## 下一版正式合同与正常边界接续准备

2026-10-09 07:16（UTC+8）实际 source029 的同一三owner仍在运行：本段完成349次更新＝10critic＋339actor，消费41742、actor消费41562、余79258。这是分开读取的进度与账户，不是原子事务，也不是后续新能力成绩；当前运行源码仍为e5bd9c，未改成下一版。

下一版代码与12份正式v27静态合同已冻结为 `81ad124554c4ab45cc274e2d482d46c76ba125b0`。36份旧v24–26合同与真实source029 Git字节保持，独立审查核对54个本地引用无差异。speed配方保持v26；terrain加入无跳请求的普通单级50/100/150/200mm与双级50＋62.5/100＋125/150＋187.5/200＋250mm，首踏面800mm，最后一级顶面450mm。普通5–40mm小楼梯继续保留。ordinary与active stepjump分别持有命令、接触/动作记录和成功判定，不借用彼此成绩。源terrain池中的大阶10%并非所有混合域都占10%；实际联合分配已单独审查。

恢复阶段诊断在未来terrain/jump/stepjump启用，保留原FSM与连续1s接稳条件；speed不加这批诊断。host批量读取、状态发布和专用日志计时属于下一版工程改动，其整更新收益尚未实测。原物理/控制、轮响应与毫秒域随机化保持，腿bag仍未qualified。

具名completed-speed→v27接续保留原REC来源、speed full-state及所有付费时钟；只在speed自然完成后由原生账户与准入绑定真实父。新增29个独立CPU生命周期用例与68个旧兼容用例、组合6项定向用例均有各自回执；24与16两批有11个重叠，不能合计成40个独立新用例。部署helper的19项CPU仅验证guard/适配器，不能代替真实场景或full-state学习结果。正常接续不使用45项精度晋级门，功能评测仍在训练外独立进行。

已在本机逐Git blob、SHA、大小和完整权限核验4341文件的归档；正式使用显式 `tar.umask=0022` 的freeze002，SHA `7afdf9e0fdb08c217d1f034f88ad52f7dad1eb7af1f86c9d710347cf5b0a9dab`。默认权限001包未部署。静态暂存与真实边界后的动态receipt/admission分开：此时没有未来speed父、实际terrain入场或新增能力资格，0/15000只是静态schema占位，不是实际课程时钟。

证据：[组合定向CPU](../reports/v6_full_training_audit_20261009/source030_composite_cpu001.json)、[正式合同独立审查](../reports/v6_full_training_audit_20261009/source030_contract_independent_review001.json)、[完整归档](../reports/v6_full_training_audit_20261009/source030_freeze002.json)、[部署helper与验证范围](../reports/v6_full_training_audit_20261009/source030_deploy/README.md)。后续实际远端stage/边界/首actor仍必须分别核验，不以本节代替。

### Kaiser 实际暂存与 CPU 预检

2026-10-09 07:39（UTC+8）读取原三owner4141407→4141412→4141576仍存活，高速本段完成417次更新＝10critic＋407actor；消费41810／actor消费41630／余79190。当前仍运行原e5bd9c／v26，同一学习链继续；此进度不代表确定性高速、恢复或新地形成绩。

freeze002的4341文件已经实际暂存到Kaiser `source_serial_dispatch030_20261009`，文件SHA清单与本机归档逐项相同，source/archive/helper身份一致。新源原生 `run_task_campaign.py --preflight-only --device cpu` 实际退出0，原始日志为 `{"status":"passed","roles":30}`；44.834秒仅为静态CPU预检和传输耗时，没有场景、额外PPO或训练信号。该检查覆盖合同与角色配置，不能代替新普通大阶的PhysX动作与学习效果验证。

新源controller只读检查已认证真实旧supervisor和唯一speed child、原native allowance，目标14897actor、当次剩14487。此时没有 `source030_speed_boundary.json` 意图或 `source_revision_030.json`，没有冻结虚构未来父。一次性有限等待接续队列仍在准备；不得现在用默认3600秒抢先切换任务。真实完整speed结束、原wrapper封包及native零PPO收尾完成后，才发布实际父与v27准入，随后核查第一terrain actor的全状态继承。评测继续在训练外，不新增45项阻断gate。

证据：[实际暂存](../reports/v6_full_training_audit_20261009/source030_static_stage001.json)、[远端CPU预检](../reports/v6_full_training_audit_20261009/source030_static_preflight001.json)、[原始日志读回](../reports/v6_full_training_audit_20261009/source030_static_remote_readback001.json)、[只读正常边界检查](../reports/v6_full_training_audit_20261009/source030_controller_check001.json)、[实际高速进度](../reports/v6_full_training_audit_20261009/source029_live017.json)。

### 第301–400次speed actor训练窗口

现有Kaiser TensorBoard HTTP读回94条所需标量，各100个actor日志点完整且有限，基线41702、窗口41703–41802。没有扫描本机event、更新优化器或重建训练场景。第201–300窗没有在本次读取，不假称四窗连续审计。

| 带探索与随机化的训练指标 | actor1–100 | actor101–200 | actor301–400 |
|---|---:|---:|---:|
| 230mm静站世界平面均速，cm/s | 3.19 | 4.11 | 4.73 |
| 305mm静站世界平面均速，cm/s | 2.53 | 3.06 | 3.11 |
| 430mm静站世界平面均速，cm/s | 2.13 | 2.82 | 3.75 |
| start_stop_1驻停条件均速，cm/s | 2.58 | 2.99 | 3.33 |
| L0结束回合完成数 | 47/387 | 2/485 | 2/442 |
| L1结束回合完成数 | 60/638 | 4/770 | 1/602 |
| 混合L2结束回合完成数 | 47/1065 | 2/1187 | 0/1048 |
| 实际wall秒/更新 | 23.79 | 20.21 | 19.40 |

第三个已读窗口中的L2分解为327失败与721collection timeout；它仍含非倒地出生，不能称四倒地完成率0。停车分母为67234帧；三个静站组各100800帧。静站速度和停车读数恶化、恢复复习成功减少，是实际质量警讯，但不同难度/探索条件的窗口不单独证明遗忘或reward缺项。下一真实500完整seal的相同出生、均值动作、真实零命令10s窗口对照仍待执行；此前短回放结论不替代新端点评测。

第301–400窗每更新环境14.663s、诊断0.666s、原生logger0.341s、未归属3.591s，全batch有效PPO帧95.593%。source030尚未采纳，不能把19.40秒归因于下一版性能改动。证据：[完整窗口归约](../reports/v6_full_training_audit_20261009/source029_tb400_001.summary.json)、[三窗身份和分母](../reports/v6_full_training_audit_20261009/source029_tb400_comparison001.json)。

### 回放端点的原生封存时钟更正

实际e5源码按总消费 `completed % checkpoint_interval == 0` 封存，speed v26的间隔为500；`evaluation_interval=0` 不会在task actor500另造快照。因此消费41903／task actor500不是已有原生产物，回放工具早前的精确500要求需修正。保持生产规则后，下一个周期快照为消费42000／task actor597／actor消费41820／stage28153／完整谱系28253，来源仍必须认证真实父REC41393；不能热插检查点或把文件名当学习时钟。

07:52（UTC+8）实际worker4141576/start113428247继续训练，task actor439／消费41842；当前原生seal有41394（critic1）、41404（actor1）、41500（actor97），42000尚不存在。此时不下载或宣称新的speed597已封存。旧回放工具f4的接口/物理bootstrap准备证据保留，真实端点要求正在更正；新物理比较仍未完成。证据：[实际封存就绪检查](../reports/v6_full_training_audit_20261009/source029_checkpoint_readiness001.json)、[独立端点更正审查](../reports/v6_full_training_audit_20261009/rec_speed_mean_native_driver_endpoint_review002.json)。

### 一次正常边界队列冻结准备

source030 的外部协调器已经完成16项CPU guard/调用顺序检查，独立审查确认完整4341文件冻结依赖在执行前及长等待后都会认证。它明确等待最多7天，仅由原生控制器在高速剩20–100 actor时暂停认证的supervisor；当前speed worker不会被停止。正常最终封存、原wrapper封包和原生0PPO终局闭合后，再由既有helper发布真实父、terrain准入与唯一wrapper。旧交付文件按实际原子替换语义硬链接保留；异常保留意图、日志、PID，不自动重试或退款。该队列不是新奖励、训练门禁或第二套PPO调度。

根运输工具分开记录transfer、只读dryrun与detached enqueue，逐SHA核对脚本/原runtime/真实stage，在入队前核对原三owner的PID/starttime/父子链及唯一集合。代码冻结提交后才执行；本段仅说明已完成的静态准备，不冒称远端队列已启动。协调器的CPU验证不代替真实终局父/首terrain actor或任何新增机器人能力。证据：[协调器与16CPU范围](../reports/v6_full_training_audit_20261009/source030_queue/README.md)、[独立审查](../reports/v6_full_training_audit_20261009/source030_queue_independent_review001.json)。

42000/task597均值对照工具已更正并独立审查，真实父41393已CPU导出；七文件下载与原生认证流程已准备。子模型仍须按实际封存SHA取回，不能把准备回执当作物理成绩。原CPU父验证的RNG仅核对保存结构，未安装并读回CPU/CUDA RNG；原交付的过强字段由新增errata更正，ONNX/PT/合同/原清单字节保留。当前没有新增能力资格。

### Kaiser 自动接续队列已实际入队

2026-10-09 08:10（UTC+8）外部协调器按冻结提交 `f859b141db41a672ef6000c8b3856df0010c1303` 完成传输、真实全树只读dryrun和一次detached入队，三步均退出0。实际协调器PID4166067/start114488334（PPID1），控制器子PID4166078/start114488781；原三owner4141407→4141412→4141576继续正常speed PPO。真实/proc/locks确认协调器持有唯一队列flock，status为`waiting_for_normal_speed_boundary`。原训练未收到信号，future边界意图/prepare/launch/source_revision030均不存在，这是尚未到达正常speed终点的预期状态。

等待进程各1线程，实读RSS协调器57,568KiB、控制器705,892KiB；没有新PPO或TensorBoard服务。7天等待是有限上限，当前speed仍以原生14897actor正常目标继续；不以精度45项门阻断，也不将此入队称为已开始terrain或已有大阶能力。实际终点后将保留旧交付、绑定真实父并提交同一学习链的新wrapper，首terrain actor须另行核验。

最新独立live019实读本段492完成＝10critic＋482actor、stage28038；账本消费41885／actor消费41705／余79115，进度/账户为分别读取。42000/task597检查点在readiness002仍未封存，当前同一worker存活，不因缺待生成文件重启。源码仍e5/v26，source030尚未采纳；新能力、回放和性能收益均保持未验证。

证据：[实际dryrun](../reports/v6_full_training_audit_20261009/source030_queue_dryrun001.json)、[一次提交](../reports/v6_full_training_audit_20261009/source030_queue_enqueue001.json)、[活进程/锁/无提前产品读回](../reports/v6_full_training_audit_20261009/source030_queue_readback001.json)、[最新原训练进度](../reports/v6_full_training_audit_20261009/source029_live019.json)。

### 第401–500次speed actor：质量分化与共享资源

2026-10-09 08:19（UTC+8），原三owner继续运行，本段513次完成＝10critic＋503actor，消费41906、actor消费41726；队列及controller仍为同一活PID和真实锁，未来边界产品无。新HTTP窗口94条×101点（含前基线）已独立重算，actor401–500对应TB41803–41902、实际完成消费41804–41903，不能将actor500称作新PT封存。

本窗230/305/430mm静站世界平面均速4.36/3.24/3.54cm/s，驻停条件3.20cm/s（51274帧）；高度MAE5.38/4.31/15.05mm。相较上一已读窗，230/430速度与停车稍改善，305速度及305/430高度恶化。L0/L1/混合L2完成分别1/520、4/724、2/1172，后者仍不是四倒地率。全部是含随机化/采样动作的训练读数，不能称完整功能通过或直接证明灾难性遗忘；同端点均值物理对照仍待实际42000/task597原生seal。

真实更新19.40→25.25秒，约88.60%的增加来自环境执行，窗内checkpoint计时全0。WSL路径资源读回确认同GPU另有活frame-curriculum进程4160719；其项目cwd与本PPO不同，没有对它发信号。两个等待队列进程CPU约0%/0.5%单核，不以进程存在推断调度挤占全部开销；共享GPU只是一项证据，未隔离变慢原因。下一版性能修改仍未采纳。详见[第五窗口与资源](../reports/v6_full_training_audit_20261009/SOURCE029_FIFTH_SPEED_WINDOW.md)及[独立算术审查](../reports/v6_full_training_audit_20261009/source029_tb500_independent_comparison001.json)。

完整功能信用增量审查确认**跳阶落台停车与锁定的起跳vx落地代价相冲突**，正在独立兼容修复，未热改source029或已排队81ad。普通大阶已有有限底盘支撑/抬轮/逐平台信用，主动跳跃也已有收腿项，不能继续概括为“没付费”；尚待原生验证的是承重收腿顺序、每腿几何收腿及后段实际曝光。行进跳确覆盖3–30cm，但当前仅+.4m/s，不冒称高速行进跳。具体源码流、量纲和真实配额见[增量审查](../reports/v6_full_training_audit_20261009/FULL_FUNCTION_CREDIT_COVERAGE_ADDENDUM.md)。当前仍无新能力qualification或实车资格。


## 跳阶落台目标修复：source031 冻结与暂存

已从 source030 的 81ad 冻结树实现并审查最小修复：实际跳阶环境接受起跳请求时，另存落台前向/转向零目标；接触前轮速仍用起跳命令，普通行进跳保持原目标。原物理成功条件、三秒成本窗、六秒任务期限、FSM、奖励权重与采样份额均未改。旧奖励在起跳 vx=0.4 m/s、yaw=−0.2 rad/s 时，对正确落台停止仍收取约 −1.16125 与 −0.241046/秒成本；修复后的对应跳阶两项为零。这是可复现的目标一致性修复，尚不能当作跳阶完成率提高。

源码提交 0f4b5f673e5f7e2cb382a8fcbd5766b0e13d8a9b 与完整合同提交 ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a 已冻结。12 个 v27 对象重生，276 份旧合同逐字节保持；仅 jump/stepjump recipe、独立 design 与实际 runtime 声明一致启用，奖励与独立声明模块加入来源摘要。原 81ad/030 字节与当前 source029 高速 PPO 不变。

实际归档4343文件；Kaiser完整静态 stage 与原生 CPU 30-role preflight exit0，0新场景、0PPO。第一次 transfer 因路径替换遗漏 dispatch030 在排他保留前被拒绝；保存拒绝与只读无新路径证据后，以独立 transfer002 完成正确031传输，未覆盖拒绝。此时031仍未入场，旧030等待队列仍活跃；安全等待观察器替换及新一次性队列正在CPU跨模块审查，不能把暂存写成terrain/jump已开始。

证据：`reports/v6_full_training_audit_20261009/stepjump_stop_target_source031_independent_review001.json`、`source031_freeze001.json`、`source031_static_remote_readback001.json`及当前部署回执新增 `source031_static_candidate`。


## 最新实际接续：031 唯一观察器与 speed42000 封存

2026-10-09 09:04（UTC+8）实际读回：原 source029 三 owner 4141407→4141412→4141576 及完整 PID/starttime/argv/父子关系保持，speed 本段619完成＝10critic＋609actor；独立账本读取消费42012、actor消费41832、余78988。仅旧等待 controller4166078 收到一次PIDFD TERM，coordinator4166067自然退出；旧边界、准入、prepare、launch均未产生，原训练信号0、PPO0、退款0。取消实际回执与独立审查通过。

新冻结031协调器4173993/start114815081及等待控制器4174000/start114815433已唯一实际运行，queue FLOCK由该协调器持有，两者各1线程。当前状态waiting_for_normal_speed_boundary；扩展只读核对031 revision、prepare、launch和共享原生v27准入目录均不存在。它等待原speed正常预算终点，再走既有原生prepare/完整父状态/launch；7天等待上限、零自动重试。031尚未实际采纳，不能记为terrain或新跳阶能力通过。

原生周期检查点消费42000／speed actor597已真实封存和下载，PT SHA7d7d53ad86898614da3dd74794c37fd3bea0c42976c1426e449e40fd5eca38be。七份输入前后远端stat/SHA与本机副本一致；native全状态严格CPU验证002与原生导出exit0：9actor、8critic、17Adam，actor消费41820、stage28153、学习谱系28253。新ONNX为models/v6_speed_periodic_42000_actor597_20261009/policy.onnx，SHA4ddf8d4245f344645af9c45d732af11393b6c263bf02f6c37462cd04e6ecc5a3；35ABI向量最大误差1.0728836e-6。仅数值/身份核验，未授物理或实车资格。

准备工具把campaign SHA当作integrated plan SHA，被真实封存输入拒绝；failed001与原工具保留，纠正002仅使用原生binding.sha256＝86311f7ab75644d6be44e79c2ccf427648e5cabec22e4e75ed60f64f472f7ffd，不修改任一检查点、原训练源码或合同。相同错误也存在回放输入适配器，已最小两文件修复并通过1项真实来源哈希正反对照，冻结61e283f；原物理/FSM/预算/资格条件不变。相同出生的两端点原生均值回放尚待执行。

最新第501–600actor HTTP窗口含94条×101点，actor日志步41903–42002，对应实际完成消费41904–42003。静站230/305/430速度4.36/3.24/3.54→3.95/2.90/3.36cm/s，停车条件3.20→3.14cm/s（本窗32560帧）；高度5.38/4.31/15.05→3.96/5.03/21.94mm。速度略改善、高姿态高度恶化，不能统称趋势全面变好。L0/L1/混合L2完成9/446、10/689、8/1047；L2含非倒地出生，不能当四倒地通过率。更新25.33秒，环境19.99秒、未归属3.92秒、logger0.50秒；这是含随机化和探索的训练统计，与均值物理表现分开。

实际接续证据：reports/v6_full_training_audit_20261009/source031_waiter_cancel_apply001.json、source031_cancel_actual_independent_review001.json、source031_queue_enqueue001.json、source031_queue_readback002.json、source029_live022.json、source029_speed42000_download001.json、source029_speed42000_verification002.json、source029_tb600_001.summary.json。原030暂存与拒绝回执保持历史身份，部署主回执新增source031_once_queue。

### 实际均值回放与当前等待句柄

首版原生回放001在环境构造前失败：独立design表的recovery角色被误作实际环境阶段名，实际环境仍应为mixed。Kit关闭时SystemExit(0)又掩盖原异常，外层只凭退出码错误地继续五协议；五个内层报告实际均failed、0物理步，无能力证据。原始失败记录保留。最小工具修复选择已冻结配置的唯一enabled stage，并要求父进程读取真实native完成状态；仅自己launcher的正常close退出不再掩盖异常。23项工具检查通过，冻结fbf937effb2ec88461b75cf4af2379417db077ac，未改原训练源码、合同、物理或判据。

修复后nominal002真实完成：两端点、7场景×4副本，CPU，0PPO，exit0/88.480s。三档静站平面速度REC41393→speed42000为2.28/2.19/2.37→2.41/2.47/2.53cm/s，高度MAE2.09/1.32/5.87→3.42/4.60/26.15mm；左/右/俯卧/仰卧完整链为2/4、1/4、4/4、4/4→0/4、1/4、4/4、4/4。闭链峰均小于0.681mm，仅覆盖本次名义回放。nominal没有停车专项窗口，其quiet字段0帧不能作停车分数。结果未建立新模型更强或实车就绪结论。

这两个候选在同一env按固定顺序复用，已列明初态/RNG指纹相同，但稀疏trace在脚本/BLEND前已出现倒地物理前缀差异；尚无逆序或独立世界对照，不能将11/16→9/16唯一归因策略遗忘。真实报告为`rec_speed_mean42000_nominal002/native_replay.json`，执行回执在该目录外的`rec_speed_mean42000_nominal002.execution.json`。正反向运动与旋转协议另行串行执行，未结束的协议不计作已完成。

进一步只读核查430mm场景：两个候选在所查0.02/1.02/3.02/5.02/9.82s的真实高度命令均为0.430m、FSM均INACTIVE；1s后父模型约0.425m，新speed约0.404m。该误差不是只有初始升高瞬态，也不是恢复逻辑仍保持0.305m命令所致。核查保存于`rec_speed_mean42000_height430_readonly001.json`，不从稀疏trace反推未保存的50Hz分子，也不据此唯一归因reward或actor。

2026-10-09 09:21（UTC+8）新的SSH只读证据确认原029三owner仍活，完成658＝10critic＋648actor；独立账本42051/41871、余78949。14秒后的队列003读到649actor，031协调器4173993及控制器4174000仍唯一活跃且FLOCK实际持有，revision031/prepare/launch/shared v27入口仍无。两个读取不是原子快照。按本段14907总更新含10critic，正常actor终点14897，尚余14248actor；最近约25.45秒/actor的条件估计为约4.20天，7日队列等待仅在后续速度相近时足够，不保证主机永无延迟。

### 四个运动协议完成与第601–700次actor窗口

正反行驶和正反旋转003均已真实完成，两原候选认证通过、0PPO，未构建优化器。请求段为5Hz稀疏trace统计，停车为50Hz完整10s窗口，不混用两者的分母。REC41393→speed42000的±0.5m/s速度MAE为0.141/0.170→0.102/0.130m/s，±1rad/s角速度MAE为0.052/0.056→0.741/0.416rad/s。归零停车平面均速为正行驶2.87→2.96cm/s、反行驶2.64→2.81cm/s；所有停车副本完整10s，窗口完整不等于停车达标。新模型正向旋转明显过转，305mm场景最大倾角2.21→11.05度，不支持实车升级。四协议仍同env固定顺序复用，尚未排除求解器内部状态差异；下一步使用两个全新Kit/FixedCaseEnv进程、相同固定projection字节和两候选认证做对照。原始结果及范围见`rec_speed_mean42000_remaining_protocols003_comparison.json`/`.md`，不把中低速小样本当全高速或完整技能验收。

最新第601–700次实际actor训练HTTP窗口：94条序列各有基线加100点，TB42003–42102，对应完成消费42004–42103；均有限且分项分母核对。较501–600窗，三档230/305/430mm静站速度3.95/2.90/3.36→4.60/3.33/3.21cm/s，高度MAE3.96/5.03/21.94→2.49/5.96/25.01mm。停车条件均速3.14→3.79cm/s，本窗15114帧，前窗32560帧；条件窗曝光不同，不作同初态确定性比较。L0/L1/混合L2结束成功为23/499、21/701、31/1270；后者包含非倒地出生，不能记作四倒地完成率。有效PPO比例95.13%，每更新25.44s，其中环境20.14、未归属3.82、logger0.50、优化器0.14s。高姿态高度继续恶化，静站速度与停车走势分化，不用总reward替代行为结果。证据：`source029_tb700_001.json`及严格完整窗口摘要`source029_tb700_001.summary.json`。

2026-10-09 09:43（UTC+8）只读live025/026核对原029三owner完整PID/starttime/argv及父子关系保持，live025完成707＝10critic＋697actor，live026完成710＝10critic＋700actor；live026独立账本42103/41923、余78897。09:42的队列004另读031协调器4173993与控制器4174000存活，唯一queue FLOCK仍持有，revision031/prepare/launch/shared v27入口均不存在。当前仍为正式speed PPO和正常边界等待，尚未采纳031；全部读取为分开的快照，不声称原子一致，也没有重启、热改或退款。

另对source031完整目标覆盖做只读重算，以worker加integrated recipe patch后的有效配置为准，不能只查worker表或按名称搜step/stair：terrain、jump、stepjump均保留全部八行ordinary body_climb；最终行明确+0.200/+0.250m、0.800/0.600m踏面、mode0、无trigger_source。静态模板使用terrain component actor时钟，相对引入点1000–4000渐进；真实边界准入将按已消费账本一次绑定origin和actor_allowance，后续阶段保持同一个引入时钟。它与主动double_step_jump分别采样/评测，恢复出生在平台上也不是爬阶成功。覆盖配置不等于实际曝光或学会；待实际terrain读回几何、命令、PPO曝光和结束结果。只读证据`source031_complete_scope_readonly002.json`固定ef2cf6cd源码和recipe/module SHA；001手写结论与自身抽取结果冲突，已明确拒绝，未据此改合同或训练。

### 独立进程名义对照004：恢复变化未复现、高姿态问题复现

隐藏candidate-index工具仅为旁路回放增加严格单候选入口，原两候选认证、共享ABI、默认paired行为均保留；同一真实nominal002 projection四文件与外部SHA固定。88项CPU检查/外部只读审查后冻结bcc65de，父runner冻结3c71954。两个原生进程PID314747/315691串行各构造全新Kit/FixedCaseEnv，真实退出0且逐个通过native完成状态、PID、source/controller/物理读回和结果真实性检查；0PPO、无优化器、无训练信号。

原始结果位于`rec_speed_mean42000_independent_world004/world_00/`及`world_01/`；最新两端点四倒地完整链均左2/4、右1/4、俯卧4/4、仰卧4/4，总11/16。先前same-world复用的speed左0/4/总9/16没有复现，不把它归因于遗忘。静站三档指标则复现：速度REC→speed为2.28/2.19/2.37→2.41/2.47/2.53cm/s，高度2.09/1.32/5.87→3.42/4.60/26.15mm；独立世界未消除430mm高度问题。

独立只读审查已逐lane核对：16倒地副本在首BLEND前的532个lane/时间样本中，位置、速度、角速度、重力投影和高度均逐值相同，首BLEND时间也相同；旧复用协议的16副本此前均出现物理前缀差异。该53个共同trace时刻的证据不覆盖全部物理子步、接触求解器内部状态或完整RNG，因此仍不唯一归因于任何缓存或actor。两端点各5次原生`script_timeout`保留为失败，无右删失；名义静止专项0帧不能作停车成绩。运动003尚为same-world协议，其旋转过转不能被本次nominal替代验证。未授完整功能、实车或新模型整体更强。详见[独立实际审查](../reports/v6_full_training_audit_20261009/rec_speed_mean42000_independent_world004_independent_review001.md)及[工具与校验范围](../reports/v6_full_training_audit_20261009/rec_speed_mean_independent_world_tool_review001.md)。

2026-10-09 10:06（UTC+8）只读live027核对原029三owner PID/starttime/父子链保持，本段完成771＝10critic＋761actor、stage28317；独立账本42164／actor消费41984／剩78836。队列005另读031协调器4173993与控制器4174000仍唯一存活、实际FLOCK保持、未来revision/prepare/launch/shared入口不存在。仍是正式speed连续PPO与正常边界等待，031未采纳；分开读取不作原子快照，未热改或重启。

### source031完整覆盖的独立复核

实际冻结`ef2cf6c`的worker＋integrated override独立CPU展开，四课有效组数speed/terrain/jump/stepjump为189/241/256/270，未发现整类目标课程丢失。后三课各保留8行普通单/双阶、5–40mm三阶上下、54恢复行及高速复习；普通最高双级为+.200/+.250m、首踏.800m、mode0连续前进无触发。行进跳名称03/06/10不能当高度上限：实际七bin覆盖.03–.30m，jump和stepjump的component clock归属各自继承；原生辅助在300 actor后及固定evaluation硬关闭。没有新增重复高跳组或更改现队列。

这些仅证明冻结配置和路由完整，未证明后课已部署、实际曝光或能力。普通搭台收腿的具体动作链、行进高跳15–30cm固定成绩、每族真实policy-owned帧和无辅助ended分母仍须分别核验；配置的COM上升高度不能换成轮净空。详见[独立覆盖审查](../reports/v6_full_training_audit_20261009/source031_complete_scope_independent_review001.md)，运行状态仍以上述真实live/队列为准。

### 独立旋转005与第701–800次actor窗口

已从根冻结`dcc32ad`的runner实际完成yaw±1四个全新串行native进程，PID330303/331662/332733/334365，各自exit0并通过严格single结果认证。原生物理、控制器、同协议固定投影和两模型身份保持、0PPO。请求统计限定`.02+n*.2`的5Hz栅格、active/非零请求、stand305四lane，各180样本；额外FSM事件采样不混入。正/反请求角速度MAE为REC父0.0521/0.0562、新候选0.7415/0.4156rad/s，与原003逐值复现；新模型实速+1.7017/−1.3694rad/s。作用命令MAE也为0.7248/0.3955，父0.0311/0.0338，过转不是仅由slew均值造成。各停车统计仍是原生50Hz、完整4×10s=2000帧；不能当停车通过。两方向结果证明该名义条件下的问题真实复现，尚不证明唯一训练原因、全部高速或实车资格。见[执行完成](../reports/v6_full_training_audit_20261009/rec_speed_mean42000_independent_motion005/completion.json)与[根限定口径读回](../reports/v6_full_training_audit_20261009/independent_motion005_root_readback001.json)。

最新完整随机化训练第701–800次actor窗口为94条有限序列各基线+100点，TB42103–42202、对应完成消费42104–42203。三档静站速度为6.72/4.10/3.41cm/s，较上窗4.60/3.33/3.21均升；高度MAE2.10/4.92/27.11mm，低中档改善、高档继续恶化。停车条件3.68cm/s、39930帧，上窗3.79/15114帧，曝光不同。L0/L1/混合L2 ended成功28/470、27/658、30/1108，不当四倒地确定性成功率。每更新wall21.89s、env16.95、未归属3.67、logger.41、optimizer.13s；耗时改善不能替代质量。证据：[实际原服务读取](../reports/v6_full_training_audit_20261009/source029_tb800_001.json)、[严格窗口摘要](../reports/v6_full_training_audit_20261009/source029_tb800_001.summary.json)。

2026-10-09 10:23（UTC+8）原三owner仍存活，本段823完成=10critic+813actor、stage28369；账本42216/actor42036/余78784。队列006的031协调器与控制器仍唯一活跃并持FLOCK，未来准入产物不存在；原speed连续PPO仍在运行，未借本机回放暂停、重启、热改或授通过。分开读取非原子快照。

005独立CPU审查再次认证四个actual single结果、PID/starttime/完整argv与各自constructor、串行过程及原始SHA。两协议各16倒地lane的527个固定栅格BLEND前比较五物理字段逐值一致、首BLEND时间一致；四个world仍各11/16成功＋5次`script_timeout`，所有28初始回合均结束、没有右删失。故当前明确有两类缺口：已导出597模型的名义双向旋转过转，以及两模型均有的五条侧倒脚本超时。后者须定位脚本阶段与交接状态，不能把还在脚本控制的失败全交给增加PPO或reward权重。完整功能继续保持未证实。详见[四进程独立审查](../reports/v6_full_training_audit_20261009/rec_speed_mean42000_independent_motion005_independent_review001.md)。


### 18:03 最新实际状态与第1901–2000次actor窗口

原029三owner继续运行，live031于2026-10-09 18:02:59（UTC+8）实读本段2019完成＝10critic＋2009actor、stage29565。progress完成消费43412，分别读取的账本43413含1个在途付费预留；actor消费43232、speed域计费7113、余77587，不退款或把预留称已学习。031队列009仍为同一活协调器4173993/控制器4174000、唯一flock，等待正常speed边界，revision/prepare/launch仍未产生。后续terrain→jump→stepjump继续一个actor完整状态串联并保留复习；目前031尚未采纳。

HTTP第1701–1800窗口已被TensorBoard reservoir抽样，53/101一类缺点不能作完整100均值；保留拒绝回执。复用既有冻结CRC019流式parser，实际读取原speed PID event限定1GiB区域，117.46秒CPU只读exit0，认证1,071,597,998B完整记录区域、没有不完整尾/冲突重复，旧区域未认证。94条选择序列均含baseline43302及TB43303–43402完整101有限点；这是actor1901–2000、实际完成消费43304–43403。无本机TB、event下载、native、优化器或PPO。根另用math.fsum重算分母、成本和计数；首次摘要源码标签被数值文本替换污染，原负结果保留，只纠正标签读取原raw来源，物理/训练与所有数值未改变。

| 静站高度 | 世界平面均速 | 高度MAE | 与701–800窗相比 |
|---|---:|---:|---|
|230mm|6.05cm/s|14.33mm|速度略改善，高度恶化|
|305mm|3.70cm/s|3.35mm|两项改善|
|430mm|8.76cm/s|9.29mm|高度改善，漂移明显恶化|

停车条件均速4.019cm/s，72068帧，较旧窗3.682/39930帧升高；不同条件曝光，不称同初态因果对照。L0/L1/混合L2 ended成功62/516、73/757、71/1212，混合L2=5.86%，较旧窗2.71%增加，但含非倒地出生，不能当四倒地或全部恢复能力。valid PPO帧94.64%；每更新26.21秒，其中env20.91、未归属3.86、diagnostics.776、logger.500、optimizer.137，最近窗口无checkpoint计时，不把117秒读取时间算进一次更新。

源码审计未发现可证实的yaw单位、mode或目标错配；精细核之外还有无截断宽域成本，不能据精细项饱和直接说PPO梯度为零。五个侧倒超时在六fresh worlds失败集合一致：12/16/17在CAPTURE、13/19在PREPARE，均8秒外clock、无BLEND/RL。lane19末参考仅轮速条件未过；其他仍有姿态/gyro/支撑未过。该证据定位脚本上游，未证明守卫冲突或饱和原因，不用延长超时/缩短稳定作为未经验证的修复。已导出597模型的±1过转仍是其名义回放结果，不能冒充当前2009模型的确定性表现。

证据：[完整最新窗口](../reports/v6_full_training_audit_20261009/source029_tb2000_exact001.summary.json)、[实际CRC读回](../reports/v6_full_training_audit_20261009/source029_tb2000_exact001.json)、[yaw源码核查](../reports/v6_full_training_audit_20261009/yaw_response_source_audit001.md)、[五脚本超时定位](../reports/v6_full_training_audit_20261009/recovery_five_script_timeouts_readonly001.md)。完整功能和实车资格仍未建立，主线没有停训、热改或重复启动。


### 18:52 连续训练状态与原生自起诊断

2026-10-09 18:52:06（UTC+8）live033实读原source029三owner仍是4141407→4141412→4141576。本轮2168更新完成＝10次critic适配＋2158次actor更新，stage actor29714；独立账本消费43561、actor消费43381、speed域消费7261、剩77439。进度与账本分开读取，非原子快照。队列011仍由原协调器4173993和控制器4174000等待正常speed边界，未来revision031/prepare/launch/v27入口尚未产生。正式训练没有因本机诊断暂停、重启或修改运行合同。

最新完整效果窗仍为actor1901–2000，数值见上表。230/305mm静止速度略改善，430mm静止速度恶化；停车4.019cm/s，混合L2结束回合71/1212。该L2包含非倒地出生，不能作为四方向自起成功率，也不能以课程组名代替实际运动指令。当前结果不足以证明整体收敛或全部技能完成。

诊断008使用原REC41393候选、同一冻结原生配置，在一个全新CPU世界记录五个既有超时副本的1600次物理反馈；0PPO、不构建优化器、不改变FSM/物理/稳定条件。工具在原生关闭前导出，26项CPU检查和独立复核通过。007原回放虽然完成，但关闭后诊断导出缺失，保留原始失败；008才建立完整诊断。父进程严格认证完成状态、实际PID、原生源文件和输出，并逐值核对004的reset、完整策略trace、事件及回合/停车指标相同；不由此声称隐藏求解器状态完全等价。

五条失败lane12/13/16/17/19均没有由脚本交接进入BLEND或纯RL，ready累计峰均为0；12/16/17停CAPTURE，13/19停PREPARE。后两条在PREPARE的23/104次参考计算中，八项传感条件从未同时通过。最后1秒前三条的倾角条件持续不满足，lane19的轮速条件187/200次不满足；PREPARE还包含配对角误差、轨迹完成及连续ready要求，不能仅按最后单帧决定放行。起始阶段约1.005s的RL标签属于被动零力矩释放，释放后的blend为0、learning mask全为false，不计为接管。PREPARE分母使用参考更新前的23/104行，不能与更新后的24/105行混用。FK支撑/净空估计与真实接触的一致性未在本诊断中证明。

释放后1399个有效物理区间，四腿轴请求峰值均低于当时力矩界；轮轴请求峰22.74–26.17N·m，实发和物理applied峰均4.5N·m，轮请求超界约8.5–11.9%。这证实限幅实际发生，尚未证实其为超时的唯一原因。所录“请求→实发→applied”层与反馈时点分别保留，气簧力独立记录，不把脚本阶段失败归因于未执行的RL奖励，也不据此热改上限或缩短连续1s稳定要求。

证据：[实时训练](../reports/v6_full_training_audit_20261009/source029_live033.json)、[后课队列](../reports/v6_full_training_audit_20261009/source031_queue_readback011.json)、[诊断完成](../reports/v6_full_training_audit_20261009/recovery_script_diagnostic008/completion.json)、[五副本逐物理步分析](../reports/v6_full_training_audit_20261009/recovery_script_diagnostic008/analysis001.json)。008独立审查实际核验51字段、各stage1600次、562项导出数值重算相同，PD/轮P+probe与力矩各层精确重建；详见[实际输出独立审查](../reports/v6_full_training_audit_20261009/recovery_script_diagnostic008_independent_review001.md)。新当前actor2158尚未做确定性完整能力评估；已导出的actor597过转结果不能作为它的现成绩。

19:05:09（UTC+8）live034复核本段2208完成＝10critic＋2198actor，原三owner保持；账本43601、actor消费43421、speed域7301、余77399。queue012仍等待正常speed边界，没有下一版准入/launch产物。训练完整效果窗口未刷新，仍为上述actor1901–2000，不能用较新进度补成较新行为成绩。
