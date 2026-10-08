# 完整训练审计与接续改进

日期：2026-10-09（UTC+8）。审计对象为实际 source023/v21 恢复训练、误派发后完整封存 REC actor14188，以及已实际进入speed PPO的source029同 v26 单模型串行接续。参考华南虎官方固定提交 b8ff79f3df855faf9dc92f4a282bd80c42649466 的源码与三份保存配置。区分实际运行、冻结候选和建议；本文件不是新的训练合同。

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

## 对照后真正需要补强的内容

| 项目 | 当前证据 | 改进方向 |
|---|---|---|
| 普通大台阶 | v26普通上阶池改为5/10/20/40mm三阶上下8组。200mm+再250mm的普通搭台收腿组已不在该实际训练池；台阶上恢复/跳阶仍有 | 小阶是已授权的优先课，合理保留；在完整计划中显式补回普通大台阶的后续难度层。与主动跳阶分别训练/验收，不能互代 |
| 恢复交接 | 54恢复组+41基础复习组；成熟L0/L1/L2比例20/30/50。完整链进入RL后还需连续1s接稳才能放行后续运动 | 按实际lesson统计脚本完成、首接管、连续稳定段、放行、运动曝光与结束原因；纯L2单独给分母。失败定位到脚本、交接、接稳、运动哪一段 |
| 恢复条件的地形适用性 | support/body_clear使用FK/IMU复合估计而非直接真实接触；左右高度差等条件可能在跨台阶姿态中更苛刻 | 先用同帧原生接触/净空与估计对照，证明误判后修几何/估计；不直接缩短1s或把未完成写成完成 |
| 速度/姿态联合难度 | speed的0/120/240/360 actor更新放开到2/3/4/5m/s、1至3圈/s与.23–.43m。普通SCUT映射仍主要±2.7m/s/±1圈/s，特殊组另覆盖极限 | 根据继承模型的同条件成绩和有效样本，分开扩展速度、高度、加速度、转向、延迟；高姿态×高速×强扰动不能只靠均匀全组合。360是否太快目前是待验证假设 |
| 后期复习有效性 | v26各域189/233/248/262组，4096静态分配最小11/5/3/2环境；脚本段还不进入PPO | 报告每个能力族真实可学习帧、结束回合和条件覆盖。合并重复教学规格或按宏观能力分配人口，保留固定评测锚点；不要只增加更多细碎组 |
| 普通运动无指令离地 | 普通mode1旋转不触发mode4跳跃请求，air专项支付也有模式约束；dense普通vertical_velocity目前固定为0 | “旋转时跳”尚不能归因状态机误触发。先记录轮接触、vz、腿速度、力矩/饱和和实际命令；普通运动vz代价可做单项候选，不施加到合法跳跃相位 |
| 跳跃定性→定量 | 已有辅助概率/力度退火及顶点bin课程；跳高、垂速、收腿reward核宽仍固定 | 加入明确actor时钟的宽→紧候选，宽核覆盖实际未收腿/未达到顶点的误差，后期精调，同时保留大误差可区分的成本。先用误差/付款分布定宽度 |
| 负载与质心鲁棒性 | 本项目COM x±1cm/y±.5cm/z=0，base质量.9–1.1；华南虎所核验COM更宽 | 按实际载荷/装配误差扩大并渐进加入；不要把较宽范围直接视为一定更好。腿侧bag仍未qualified，是实车响应可信度缺口 |

原地跳、行进跳、跳阶已在代码/课程中存在，但存在课程不等于完成离地→收腿→落地→接稳。华南虎本次核验三份保存交付配置主动JumpTakeoff均关闭，不能援引这些配置证明主动高跳已经实证解决。

## 高速与台阶的具体安排

1. 速度课持续混采静止、低中高速、正反旋转、弧线、加速、松杆制动、反向切换，并覆盖低/中/高姿态。稳定速度时要求良好姿态；加减速阶段要允许机构必要的瞬态倾斜，并检查加速度/jerk与恢复水平的时间，不能只以全回合平均pitch掩盖制动峰。
2. 保留平地足够跑道；已实现flat半长160m、半宽8m，普通地形越障后的正向边界5.6m、支撑延伸6.2m。场地已扩展，不应重复把当前问题归因小平地。
3. 普通小阶先做5–40mm上下、坡、粗糙面与障碍接触；用地形×命令表限制为有用的直进/退、合适高度和小yaw。之后单独引入普通大阶及200+250mm双级/首层800mm踏面，真实机械回放验证后冻结。
4. 主动跳跃按离地、收腿、落地吸收和站稳分项训练/记录；跳阶最后增加平台边缘定位与连续两级任务。正常爬阶不需要JUMP请求，跳阶使用明确状态机请求，两者定义保持分离。
5. world-reference旋转平移每步重算body命令，部分路径绕过普通命令投影。世界系匀速与body-vx变化不同，不能直接用vx×yaw判定错误；需记录完整坐标系命令、轮速、滑移和离地做专项可达域核查。

## 历史问题的原因与仍未证明的内容

已证实：普通静止错误领取落地正奖励；adaptive LR恢复后超出期望；短worker和停车同相位导致曝光集中；特定恢复脚本接触冲击及闭链误差；部分原生地形四元数接口错配；本次native CLI参数漏传、completion文件名读取不符，以及串行主调度在已认证REC提前封存后仍重跑旧REC的选课错误。对应修复/负证据已有记录，不能把它们重新列成当前全部仍未修。

历史同条件均值策略对照确有部分运动/yaw回归，停车同时改善，说明存在任务间权衡。当前两个不同难度/噪声窗口的统计不能独立证明遗忘，更不能断言“继续训练自然会全部收敛”。历史A/B的两个seed是评测重复，并非两次独立PPO训练。

静止宽核、精核与L1速度代价，姿态、yaw和接稳成本已经支付。精核在大误差区接近饱和不代表广域成本也无信号；reward对误差的导数也不是PPO直接反传的梯度。不能以惩罚数量或总reward判断收敛。增加权重需先排除错误命令/掩码、交接状态、未曝光、探索与部分可观测性。

## 华南虎旧解读的更正

- 它有预训练继承、迭代技能解锁、相位奖励和门控，不能称“无相位所以不会遗忘”。多个保存目录也不能证明实车必然切换多个ONNX。
- flat_orientation_x/y_v是速度命令相关的投影重力姿态代价，不是角速度惩罚；后者另有ang_vel_xy。
- 三份actor都是单帧35D；raycast不在actor输入，不能说对方actor有地形扫描而我们没有。
- 所核验延迟按5ms物理步推进，观测5/10/15ms、动作5/10ms；不是50Hz帧10–40ms。本项目毫秒DR已更宽，需避免与USB传输/执行器响应重复累计。
- 10%静止是命令重采样概率，不是永久固定10%环境。stand_still_lin_vel在flat为−2、rough_rotation为−1，所核验rough_dash配置没有该reward。
- 主动跳跃源码能力、交付配置启用状态、训练轮数和实车成绩是四种不同证据。

## 接续实施优先级

**P0：当前已授权串行接续已实际运行。** source029从误派发后最新REC14188完整状态进入speed，全部19次新增actor消费保留；CPU准入、一次origin更正、唯一owner及首actor恢复已通过。固定4096、35D/81D、50Hz策略/200Hz物理反馈PD160/2.5、轮响应和原预算；继续本段并按冻结配方自动串行到terrain/jump/stepjump，不把未validated的reward建议热改到本次运行，不增加重复启动或阻断gate。新效果等待完整训练窗与旁路评估。

**P1：下一版本统一补齐可解释性与覆盖。** 基于本报告一次冻结：恢复按真实L0/L1/L2分桶、脚本/接管/放行/结束分段计数、pitch/roll与轮接触/vz统计、各能力真实PPO曝光；普通大台阶明确后续课程位置；速度×姿态×地形×制动覆盖表。复用原FSM/logger，不增45项阻断晋级。

**P2：用窄对照决定reward/难度调整。** 同一封存模型、同初态/随机化比较mean/sample；固定少量静止、停车、正反旋转、高低姿态、恢复、阶梯用例比较旧新模型。在此基础上一次只测普通vz代价、核宽退火或难度节奏中的一个。核宽取实际误差分布，不凭感觉同时翻倍所有权重。评测保持离线/旁路，不每40次重建训练。

**P3：针对实测环境时间提速。** 最新19.17s/update中环境约16.01s、优化器0.127s，优先查host同步/重复状态读取、闭链物理和日志。已有host同步合并候选尚无Kaiser实测收益。维持物理语义作短profiling和未profiling对照，不把CPU event解析时间算成PPO暂停，也不凭低利用率认定线程竞争。剩四域若各15k、速度不变约13.3天，这是条件估计，不能承诺照抄华南虎即可一天完成。

## 证据索引

- [华南虎源码/实际保存配置对照](../reports/v6_full_training_audit_20261009/scut_source_comparison.md)
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
- [华南虎rough保存配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml)

## 本次模型交付

已从同一REC14169封存权重导出REC41374候选，ONNX SHA `63574953c97aad374fa1554c53486cd540a3019857af72e0c06cc010214299bf`。源PT与合同保持原字节、数值导出通过。它作为恢复优先候选，不可称为已验证的恢复+高速最强模型；其普通课程范围仍是|vx|≤1m/s、|yaw|≤1rad/s。新联合速度续训的权重需另行封存评估后才能替代此文件。

对照的speed19837来自speed←flat分支，不能合并其高速经历到REC；旧REC38000高速回放失稳证据保留。详见[模型选择审计](../reports/v6_full_training_audit_20261009/model_selection_audit.md)与[本次交接文档](V6_RECOVERY_PRIORITY_MODEL_HANDOFF_20261009.md)。

## 本次同条件候选回放的实测结论

REC41374和REC38000在相同原生合同的左侧90°倒地场景均5.48s进入纯RL、6.48s连续接稳后放行、7.46s完成完整链。最新完成姿态略优，但普通运动明显有取舍：0–5s加速vx MAE新/旧0.393/0.556m/s；6.34–7s实际命令归零后的33帧，世界平面均速新/旧1.965/0.083m/s、最大倾角35.43°/10.54°。最新7.06s越过侧界，非跌倒；因此后续高yaw只有3帧，不能给高yaw通过或失败结论。旧模型高yaw虽150帧，角速度MAE1.509rad/s、最大倾角28.98°也未良好。

这提供了同条件的恢复制动权衡证据；只有各一次固定出生，未定位因果，不能称综合最强或实车合格。本次优先恢复/制动回放选择旧REC38000，最新REC41374另存，不让“最新”代替选优。普通上台阶与主动跳阶各自未测，不能借左倒地或跳跃接口授予爬阶信用。

第二次回放仅增补位置和共同时间窗口；两次场地实际均为8m宽，不能把它描述成拓宽场地。累计行程不是零命令静止漂移。详见[原生同条件结果](../reports/v6_full_training_audit_20261009/model_compare/native_comparison_addendum.md)。
