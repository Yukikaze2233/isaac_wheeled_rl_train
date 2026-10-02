# V6 静止支付与毫秒延迟核对

本页记录用户2026-10-02“漂移严重、轮速/静止惩罚是否未生效”和扩大毫秒延迟覆盖的要求。初次审计对象为v5源码07be0f6、4096环境，实际部署身份以launch.json为准。原始曲线来自Kaiser TensorBoard attempt_005，保存为 reports/v6_task_campaign_20261001/learning_audit/stationary_reward_audit_v5.json；没有复制event或在本机启动TB。

05:42已迁移到冻结v6源码34138176d33d／attempt_006并核验真实PPO。上文v5数据继续保留历史时点；最新启动、性能与完整驻留窗口另见文末，不能把两个worker的累计均值或max拼成一条趋势。

## 已确认的支付与指标边界

- 实际启用dense_tracking。静止广域核为未截断pseudo-Huber（尺度0.10m/s，权重1.5），精核尺度0.04m/s、权重0.75；零vx/yaw且普通接地时两项均生效。旧precision_tracking的0.03宽度已被dense替换，不能沿用旧“6cm/s以上完全饱和”的归因。
- 冻结报告TB前119轮：230mm静止两项最近10轮合计−0.1781 reward/s，305mm−0.09424，430mm−0.09064；wheel_velocity同窗分别约−4.41e−5、−7.36e−5、−6.44e−5/s。均非零；普通静止落地高度/速度bonus均为0。轮速项是−1e−5×两轮轴速平方的小成本，与SCUT相同量级，不承担位置驻留的全部责任。
- 报告第120轮累计行为：230/305/430mm平面速率均值2.15/1.24/1.37cm/s，最大出生点位移1.11/0.292/0.353m。方向偏置分别+0.490/+0.088/+0.584cm/s。确有剩余漂移，430mm偏置未消除，不能从总reward较高认定站稳。
- Behavior/stand_drift_max_m是每组所有随机策略环境、全部历史回合的最大值；新attempt120轮对应57.6s单环境仿真，不是固定10s确定性评测。EpisodeMetrics origin从各回合第一步取值，max跨回合保留；observed_sim_seconds是环境时间求和，不能当单车时长。动作仍随机探索，轮侧std约0.16；不得与确定性回放混记。
- 合同没有stationary_tracking位置anchor。瞬时速度成本已支付，但同样的小速度偏置积分后的位移没有单独定价。这是目标覆盖差异，尚不能证明为全部漂移的因果根源；原B臂静止权重×2无益的结论继续有效，不盲目翻倍或增加强制零轮速项。没有位置/历史输入的35D策略也不能保证漂远后找回出生点。
- LearningAudit/continuous_zero_10s还要求高度命令逐帧不变，当前为0；它不是“零速驻留没有采样”的证明。新增独立的2s进入后固定10s位移窗口，显式记录样本数、mean/max和速度，保持旧累计最大指标以便追溯。

## 华南虎逐项纠正

核对本地及GitHub冻结源码b8ff79f3，实际rough_dash/2026-07-19参数：

- use_obs_delay/use_act_delay均开启，四传感器组分别1–4、腿/轮动作1–3；缓冲在_apply_action物理子步推进。dt=5ms时是观测5–20ms、动作5–15ms，不是按50Hz策略帧算成20–80/20–60ms，也不是10–40/10–30ms。
- flat_orientation_x/y_v罚的是速度命令调制的投影重力分量平方，不是pitch/roll角速度。我们pitch_velocity、roll_velocity已经都有；horizontal_angular_velocity为−0.05×pitch/roll角速度平方，与SCUT ang_vel_xy对应。
- SCUT wheel_vel=−1e−5与我方已有项同量级；该rough_dash实际stand_still_deadzone_enabled=false、stand_still=−0.0，不能称“实际激活了强静止专项罚”。他们仍通过零速命令的全局速度跟踪付费。
- 我方mass/COM并非“几乎相同”：当前底盘质量0.9–1.1、COM x±10mm/y±5mm/z0；SCUT归档x±40mm/y/z±20mm。放宽COM和前置推扰均应独立审核，轻推不会给无位置输入的策略凭空增加位置感。本轮不夹带改力学随机化或reward。

## 延迟补强方案

已有command_transport覆盖USB下行0.04–0.16ms、抖动和1kHz丢包；signal_perturbations只开噪声、max_delay_steps=0。确实缺少较宽的策略传感器年龄和设定值执行滞后。

新增独立的物理时钟延迟层，复用Isaac Lab原生CircularBuffer：观测只延迟IMU角速/投影重力/主动轴角度和速度，不延迟当前命令、模式、时钟或previous-action；critic真实状态及底层PD反馈保持当前物理子步。动作延迟腿角设定值与轮速设定值，再用新反馈PD，避免将“反馈延迟”和“保持旧力矩”混同。USB短丢包仍由既有链路层处理，不重复注入。

v6候选覆盖：25%无额外延迟；其余环境按60/35/5%选常规/宽域/压力档。常规obs0–20ms、动作0–15ms；宽域obs20–40ms、动作15–30ms；压力obs40–80ms、动作30–60ms。各传感器/腿轮组独立抽取回合固定基延迟，另加±5ms抖动；压力是未测工程长尾，不能称实测USB延迟。采用200Hz网格，误差最多半个5ms子步，日志同时给请求范围和实际lag毫秒。强度按本域actor时钟渐增，flat从1500至2500，其他域0至1000；零延迟样本始终保留。

仅准备v6冻结迁移，测试通过前v5持续PPO；不热改正在训练的源码/合同。迁移继承完整actor/critic/Adam、所有消费和课程时钟，并计入一次10更新critic适应；其后常驻PPO，每500封存、非阻断review、原20k每域上限不变。全部入口断言新模块启用与参数，真实物理推进后核验非零lag和fresh PD时钟，再更新实际部署回执。未qualified腿bag仍未接入。

## 实际曲线复核（05:02冻结样本，05:13浏览器查看）

已在远端TensorBoard打开attempt_005的LearningAudit曲线。静站速度存在周期尖峰，峰间回落，不能用累计max的单调性描述持续恶化。近期20轮与前20轮的比较要同时看均值和方向偏置：230/250/305/430mm静站均速分别2.50→1.86、1.91→1.53、1.20→1.16、1.52→1.29cm/s；430mm前向偏置0.41→0.84cm/s仍有反弹。尖峰与约20s回合周期相近，但未按reset事件逐帧配对，暂不把出生过渡定为唯一根因。

41行为组原始快照和停车窗曲线在`reports/v6_task_campaign_20261001/learning_audit/all_behavior_audit_v5.json`，快照metrics270／progress277（原子文件发布时间不同，不捏造成同一时刻）：

| 行为 | 名义高度速度MAE | 非名义高度速度MAE | 名义／非名义高度MAE | 非名义累计失败／已结束回合 |
| --- | ---: | ---: | ---: | ---: |
| forward_05 | 1.95cm/s | 10.06cm/s | 1.42／15.05mm | 0／390 |
| backward_05 | 1.14cm/s | 2.31cm/s | 1.43／2.92mm | 2／386 |
| forward_1 | 2.35cm/s | 4.59cm/s | 1.56／23.20mm | 2／439 |
| backward_1 | 4.80cm/s | 7.93cm/s | 2.46／3.52mm | 1／409 |

非名义高度泛化差距是真实的，尤其前进高度和偏快；但“多数环境50°+／整体塌陷”不是max倾角的含义。所有记录的失败为nonwheel_contact，fall=0；例如height_scan是14／2470（0.57%），curve_low高度变体10／506（1.98%）。这些低失败率也不代表任务精度达标。部分最短失败0.44–0.62s提示需查出生过渡，但单个最短值不能确定所有失败时机。curve_low高度变体另有232边界截断，不能混成跌倒。

停车要用LearningAudit的真实settled_stop掩码。最近20轮名义start_stop_05/start_stop_1/velocity_curve_1窗口均速1.30／1.22／1.31cm/s，超过3cm/s的样本占3.26／1.87／2.94%；高度变体1.63／1.48／1.46cm/s，占9.32／6.07／5.33%。start_stop_05高度变体占比前20轮6.71%→9.32%，局部恶化应持续跟踪，不能用总体改善抹掉。无采样rollout必须先看samples，不能把0当通过。

原Behavior/settled_stop_speed约0.075–0.138m/s是本attempt历史最坏值；Behavior/stand_drift_max_m在停车/曲线组保留行进前出生点，因此8–11m包含有命令运动，不能称停车后漂移8–11m。新增完整10s静止窗口只记录合格完整样本，帮助把出生最大位移、瞬时速度和驻停位移分开；没有据此加位置reward或修改判据。

v6候选342项CPU测试与30角色预检通过（2026-10-02）。原生探针将使用冻结4096合同，工程时钟只在探针设为课程终点2500，执行24策略步／96新反馈PD，检查重复读取观测不推进延迟、最高80/60ms和零延迟样本；不训练、不消费更新、不授予能力通过信用。运行回执完成后另补部署证据。

来源：[SCUT物理子步与reward源码](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py)、[实际rough_dash参数](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml)。

## v5封存与v6当前读数（05:45）

v5在2432总消费完整封存。05:29快照的短窗曾显示230/430mm反弹；封存前最后20轮四档静止均速又回落至1.73／1.50／1.17／1.40cm/s（230／250／305／430），前20轮为2.17／1.82／1.23／1.60。430前向偏置仍1.01cm/s。周期性尖峰和不同短窗可给相反方向，不下“持续恶化”结论。非名义高度forward_05末20轮实际0.677m/s、高度20.79mm，比前窗0.624／15.33mm更差；名义前进0.528m/s／1.27mm。原始TB及最新累计41组在learning_audit/v5_sealed_curve_metrics.json与summary；累计69失败／41620结束回合（0.166%），其中68 nonwheel_contact和1 fall，boundary3102另计且可能与timeout重叠。唯一fall在yaw_continuous_low__reverse_train__height_train，不能继续将此前“fall=0”搬到封存时点。

v6原生工程验证已通过：4096×24策略步、96新反馈PD，35D81D／reward全有限、重复观测读不推进历史；最高obs80ms、目标60ms，启用人口75.49%，clean年龄0。工程时钟2500仅用于探针，正式flat从1422完整恢复，1500才渐入。首PPO核验保留Adam和九actor张量真实改变；10次critic适应全部计消费。证据performance/native_delay_probe_v6.json与deploy/v6_first_ppo_verification.json。

05:45 v6进度40更新（10 critic＋30 actor），总消费2472／flat1472，flat actor1452、actor谱系1820／完整谱系1850。behavior原子文件到40更新，TB最新2469，对应38更新；它们发布时钟不完全相同。下表behavior为20→40两份累计统计的frame加权差分，只含正常PPO；驻停与10s完整窗口使用各自最近20轮TB、按samples加权，排除无采样的0值。

| 行为 | 速度指标 | 高度MAE | 额外说明 |
| --- | ---: | ---: | --- |
| stand_230mm | 平面均速1.90cm/s | 2.21mm | 连续10s峰值均值6.70cm、最大33.00cm，81窗口 |
| stand_250mm | 平面均速1.80cm/s | 1.55mm | 10s峰值均值5.94cm、最大24.06cm，79窗口 |
| stand_305mm | 平面均速1.33cm/s | 2.04mm | 10s峰值均值3.02cm、最大14.63cm，81窗口 |
| stand_430mm | 平面均速1.65cm/s | 4.65mm | 前向偏置1.36cm/s；10s峰值均值13.96cm、最大25.96cm，81窗口 |
| forward_05 | vx MAE2.84cm/s，实速0.528m/s | 1.35mm | 命令0.5m/s |
| backward_05 | vx MAE1.26cm/s，实速−0.510m/s | 2.28mm | 命令−0.5m/s |
| backward_1 | vx MAE2.21cm/s，实速−0.979m/s | 9.15mm | 高度偏差仍大 |
| forward_05高度变体 | vx MAE16.60cm/s，实速0.659m/s | 19.22mm | 命令均值0.497；存在边界后重置，不能当纯0.5稳态增益 |
| rotate_1／反向 | yaw角速度MAE0.087／0.084rad/s | 2.47／3.64mm | 平移vx MAE1.86／1.22cm/s，不与yaw混用 |
| start_stop_05／1 | 驻停均速1.306／1.308cm/s | — | >3cm/s占3.35／2.60%，窗口帧24329／32327 |
| velocity_curve_1 | 驻停均速1.261cm/s | — | >3cm/s占1.95%，8766窗口帧；只11个rollout有样本 |

10s窗口数还少且含随机探索、当前高度资格与新回合出生；这是新训练诊断，不能授予固定10s漂移验收信用。430mm的均值位移与持续前向偏置一致，是明确剩余问题；低姿态和不同高度运动也有弱项。不把最近40轮多数静站未结束回合、failures=0误作100%存活验收。v6前38轮全部标量有限、LR维持1e-4；actor-PPO近20轮wall中位11.50s，env9.825s／optimizer0.133s。短run相对v5末期6.187s开销增加，主要时段仍在环境步进，尚未做严格配对的delay/diagnostic性能消融。

当前仍训练flat；其他任务域未启动，不宣称自起／高速／台阶／跳跃已学会。没有阻断45项gate，无新固定评测通过分数；现有轮响应保留，腿响应bag仍未qualified。最新证据learning_audit/v6_metrics_snapshot.json与deploy/launch.json。

## 延迟渐入后的跟踪（07:45–07:47）

07:47实读仍为同一3413817冻结源码、tmux r6、worker2486394与supervisor2486378连续PPO。总消费3066／flat消费2066，flat actor时钟2046、actor谱系2414；其他五域消费0。第3000不可变checkpoint的模型SHA与seal匹配，完整谱系2378／actor谱系2348、flat actor1980；review_available只表示可进行有证据的复核，不表示已验收或应停机。campaign_state中finished_at仍保留v5旧封存时间，当前running、推进中的progress和存活PID证实不是停止；此次未重启、未热改或晋级。

远端TensorBoard07:45快照至消费step3055；行为原子统计至本attempt620更新。下表比较相邻100更新：TB2856–2955与2956–3055，behavior420→520与520→620。行为采用累计frame加权差分，完整10s窗口和驻停采用各自samples加权、排除无样本0。两个发布时钟不强行对齐。

| 指标 | 前100轮 | 最近100轮 | 含义 |
| --- | ---: | ---: | --- |
| 230mm完整10s驻留峰值均值 | 17.47cm（158窗口） | 20.05cm（228窗口） | 精度退步；最大值98.52／102.54cm仅作尾部观察 |
| 250mm完整10s驻留峰值均值 | 14.31cm（154） | 18.60cm（180） | 精度退步 |
| 305mm完整10s驻留峰值均值 | 9.45cm（157） | 10.67cm（240） | 精度退步 |
| 430mm完整10s驻留峰值均值 | 12.55cm（161） | 13.27cm（236） | 仍有明显漂移 |
| start_stop_05／1驻停均速 | 3.234／3.198cm/s | 3.627／3.572cm/s | >3cm/s帧占48.12／46.68%→55.17／54.88% |
| velocity_curve_1驻停均速 | 3.296cm/s | 3.645cm/s | >3cm/s占49.40%→56.89% |
| 名义forward_05／backward_05 vx MAE | 2.917／2.483cm/s | 3.396／2.954cm/s | 含出生和边界截断后的重置，不能当纯稳态速度 |
| 正／反rotate_1 yaw角速度MAE | 0.1580／0.1479rad/s | 0.1865／0.1638rad/s | 转向精度退步，不是平移误差 |
| forward_05高度变体vx／height MAE | 16.20cm/s／23.09mm | 14.65cm/s／21.62mm | 有改善但仍弱；实速均值0.626，命令均值0.496m/s |

延迟课程强度TB中位0.3855→0.4855，最新0.535；启用人口约74.6%，最近100最高观测年龄45ms／目标年龄30ms。新反馈PD保持200Hz，延迟施加于传感器与目标缓冲，不把反馈PD变成延迟反馈；这些是工程管线年龄随机化，不是声称USB实测45ms。带探索、不断增加延迟难度的训练读数显示精度回退，但未做同冻结checkpoint的delay on/off配对回放，不能把它断言为永久策略退化或归因于reward未生效。

最近100行为结束回合9026、failure93（1.03%），前100为106／8145（1.30%）；最近100均记nonwheel_contact，fall／closure／knee／spring原因计数0。累计409／54504（0.75%），原因计数可重叠，不能直接相加；boundary6173独立列示，不当跌倒。较低failure比例不抵消漂移和停车精度退步，也不授予固定评测通过信用。

最近100 PPO更新wall中位12.472s，env_step10.919s、optimizer0.119s、diagnostics0.320s、logger0.164s；前100wall12.339s。LR仍fixed1e-4、所选标量全部有限。相对v5末期6.187s开销仍明显；当前主要时间在环境步进，没有评测等待阻断，也没有证据把时间归因于Adam或多个PPO worker竞争。新增延迟／诊断的具体性能占比尚待同源配对剖析，不热改健康进程。候选保持unassessed，腿响应bag仍未qualified。快照与原始远端TB系列SHA见[07:45跟踪回执](../reports/v6_task_campaign_20261001/learning_audit/v6_followup_20261002T0745.json)。

## 延迟实现提速与全强度读数（10:07）

总消费3723／flat2723，flat actor2703、actor谱系3071；延迟强度已经1.0。最近100轮带探索训练：230／305／430mm静止均速9.33／9.27／7.07cm/s，名义前进／后退速度MAE13.07／12.36cm/s，rotate_1 yaw角速度MAE0.370rad/s。10s完整驻留峰值均值33.06／32.87／41.98cm，305mm仅3个完整窗口，须连同样本数解读；未完成的窗口不当作0位移。名义0.5／1.0启停驻停均速9.62／9.83cm/s，>3cm/s帧占91.38／91.47%。高度变体前进height MAE5.65mm，比07:45的21.62mm改善，但速度仍10.48cm/s。不能用单项高度改善掩盖速度／停车退步，也不能把训练探索读数称固定验收。原始冻结TB快照位于原root的delay_optimization_before.json，SHA-256为3fe4449afb2c602272edaf057c575679b3783c17b0768e74f4b4c4b7959fd3ca。

最近50 PPO wall中位13.319s、env11.704s；不是物理dt=5ms过大。每轮24策略步×4物理子步=96次，每次新增2次历史append和6次延迟读取。原isaaclab.utils.buffers实现append会整段torch.roll，读取max_length会将GPU标量.item搬回主机，首次append判断又用GPU布尔条件；静态路径约576次标量读取加192次布尔检查／PPO。另有完整35D观测重复构建和静止窗口布尔索引导致动态形状同步。

本次只修实现开销，冻结v6合同不变：

- 复用现有isaaclab_experimental.utils.buffers.CircularBuffer的游标存储、计数与reset，增加小型GroupedDelayBuffer适配。容量留在Python常量；reset的已知ID用于首次填充，六组读取合并为每种buffer一次按特征gather。保持完整历史及各环境首样本复制语义，不在物理步中roll或读取设备标量。
- 延迟观测只构造18个传感器槽；命令、前动作、模式上下文和81D critic仍当前。年龄指标按六列合并归约，tag与统计窗口不改。
- 10s驻留anchor更新用固定形状torch.where，2s进入／10s计时／完整样本统计不改。

CPU随机轨迹逐值对照原线性buffer，覆盖零／半／满强度、超过历史容量、多个局部reset及全reset，反馈／目标／完整历史／计数／RNG一致；热路径拒绝.item、Tensor布尔转换和torch.roll，sensor18与旧35D切片逐值一致。18项针对性测试通过；扩展485项中484通过，另1项历史SCUT specialists测试用全局新增SKILLS检查旧v5合同，在未修改父提交b976d23独立归档中已复现同一失败；不以其失败修改运行合同。30角色启动预检通过。

部署顺序：验证和提交→新commit独立冻结目录→认证source-only revision→旧worker最新完整状态合法封存→同合同接续。保留全部消费、actor／critic／Adam、fixed LR1e-4、flat actor时钟、35D／81D、4096、50Hz策略／200Hz新反馈PD、轮响应、延迟覆盖与随机化；不再做10更新critic适应、不回A、不退款。原生4096×24步探针与配对GPU延迟微基准不含PPO／技能信用；实际首PPO需确认actor改变及完整状态恢复，再以去冷启动后的真实wall报告整轮提速，不能把微基准倍率当整机倍率。

## 提速实际结果与华南虎对照（10:36）

源码修复已提交为`2778206b5905c2760ccc381f2592b08e1cad611a`，独立归档SHA-256为`f68378de5dc773d7dcbf283ac1c31476725906e809cf435a6dc34aba370391dc`。认证source revision 007只替换冻结源码根目录；v6 plan／worker合同SHA、reward、随机化分布、反馈控制和消费上限不变。父attempt_006总消费3753／flat2753合法封存，模型SHA为`659637d01ac9abf33da649534d36a39eb95f3893441e481bc69dc8c28b0b2566`；其完整学习谱系3131、actor谱系3101、flat actor2733。新attempt_007初始actor／critic／Adam逐值一致，环境以新回合启动；首3754九个actor张量改变，全部时钟分别加1，无重复critic适应。不能将完整学习状态恢复称为场景物理状态逐值恢复。

配对GPU延迟微基准覆盖4096环境、96物理步／块、六次交替，零／半／满强度及部分／全reset的反馈、目标、年龄、RNG和指标逐值一致。旧路径768次`aten::item`（包含192次布尔检查）、192次`roll`，新路径这两项均为0。延迟层中位0.3331→0.0722s；基准与准备期间的旧PPO共享GPU，仅证实该层实现提速，不能据此声称整轮4.6倍。独立原生场景探针4096×24策略步／96新反馈PD通过，35D／81D有限、重复观测读取不推进历史、clean年龄0、最高obs80ms／腿轮目标60ms、轮响应有效。探针不消费PPO，不授予技能或硬件等价信用。

| 真实PPO时段 | 更新墙钟中位 | 环境步进 | optimizer | diagnostics | logger |
| --- | ---: | ---: | ---: | ---: | ---: |
| 修复前最后50轮 | 13.319s | 11.704s | 0.134s | 0.334s | 0.166s |
| 修复后丢弃首15轮的23轮 | 12.456s | 10.993s | 0.135s | 0.320s | 0.165s |

短窗墙钟减少6.48%；两时段为先后连续训练窗口、回合不同，不是同一场景严格配对，也不是质量保持或稳态吞吐的结论。新PPO延迟强度仍1.0，最大obs约80ms／目标60ms；所有选取的计时、loss和延迟标量有限。修复前静止7–9cm/s、停车约9.6–9.8cm/s、正向yaw角速度MAE0.370rad/s的精度缺口尚未有固定回放证据证明消失；本次实现等价优化不作为行为问题修复验收。

对照华南虎冻结提交`b8ff79f3df855faf9dc92f4a282bd80c42649466`的实际rough_dash参数：4096环境、dt=0.005、decimation=4、rollout24、20k最大更新、save_interval=500，观测／动作lag缓冲在物理子步更新。我方已采用相同物理／策略节拍、并行数和连续PPO流程；这不意味着机械模型或每轮耗时应相同。华南虎该资产求解器position／velocity iterations为12／6，我方闭链模型为64／32；不能直接降低到12／6。剩余约11s环境步进须拆分物理求解、状态读取、控制、观测和奖励耗时；在闭链残差、气簧行程及瞬态接触逐轨迹验证之前，求解器只能是待测候选，不能以未测差异代替归因。当前Adam约0.135s、监督进程CPU约0.3%，没有多PPO worker竞争或评测阻断的实证。

10:36存活核验：唯一PPO worker2557022／supervisor2557006，tmux r7；总消费3839／flat2839，flat actor2819、actor谱系3187／完整谱系3217、actor消费3799；其余五域尚未启动。GPU瞬时90%／12804MiB。正常PPO继续；结果文档提交不触发再次封存或重启，TensorBoard只在Kaiser，自动化仍PAUSED。

证据：[源修订认证](../reports/v6_task_campaign_20261001/deploy/source_revision_007.json)、[完整恢复与首PPO](../reports/v6_task_campaign_20261001/deploy/delay_first_ppo_verification.json)、[GPU逐值对照](../reports/v6_task_campaign_20261001/performance/delay_gpu_equivalence.json)、[原生场景探针](../reports/v6_task_campaign_20261001/performance/native_delay_probe_2778206.json)、[真实PPO计时](../reports/v6_task_campaign_20261001/performance/delay_optimization_after_first_window.json)、[存活与时钟核验](../reports/v6_task_campaign_20261001/deploy/delay_optimization_live_verification.json)。

## 连续训练跟踪（18:26–18:34）

同一冻结源码2778206、v6合同和attempt_007仍运行，未修改求解器或另开PPO。18:34实读总消费6799／flat5799，flat actor时钟5779、actor学习谱系6147；其余五域均0。唯一worker2557022、supervisor2557006及tmux r7存活。第6500不可变checkpoint的模型SHA与seal一致；其完整学习谱系5878、actor谱系5848、flat actor5480，全部消费继续保留。没有新固定评测、晋级或qualified结论；未qualified腿bag仍未接入。

本次核查发现TensorBoard scalar HTTP接口使用1000点reservoir，最近100个step范围实际只有32–35个返回点，不能将它们称完整100轮或直接当完整驻留样本数。已在Kaiser通过TensorBoard原生TFRecord reader对event尾部进行CRC验证，读取1704866条记录，选取61项稳定tag，并确认6525–6624和6625–6724两个100更新窗口的计时、静止／停车样本tag各有100点。远端原始有界系列SHA与文件路径保留于跟踪回执，不启动额外TensorBoard服务；本机只保存摘要。以下驻停和完整10s指标按真实samples加权、排除零样本记录。行为数据另用原子history累计frame加权差分2770→2870及2870→2970，两个发布时钟不强行对齐。

| 训练诊断指标 | 前100轮 | 最近100轮 |
| --- | ---: | ---: |
| 230mm静止平面均速／高度MAE | 8.19cm/s／6.80mm | 8.42cm/s／6.26mm |
| 250mm静止平面均速 | 8.49cm/s | 8.96cm/s |
| 305／430mm静止平面均速 | 7.84／5.97cm/s | 7.66／5.90cm/s |
| forward_05／backward_05 vx MAE | 10.27／9.06cm/s | 10.35／9.17cm/s |
| rotate_1正／反yaw角速度MAE | 0.344／0.351rad/s | 0.348／0.351rad/s |
| start_stop_05／1驻停均速 | 8.12／7.86cm/s | 8.09／8.02cm/s |
| start_stop_05／1超过3cm/s帧占比 | 87.94／87.16% | 88.13／87.93% |
| velocity_curve_1驻停均速 | 7.64cm/s | 7.56cm/s |
| PPO wall／env／optimizer中位 | 6.527／5.566／0.0848s | 6.536／5.572／0.0851s |

最近名义前进／后退实际vx均值0.400／−0.418m/s，命令均值0.496／−0.495；出生与边界重置也进入该统计，不能将其当纯稳态增益。高度变体forward_05速度MAE9.59cm/s、高度3.96mm；名义前后高度约3.38／3.30mm。高度距离较小没有修好速度、旋转和驻停；yaw单位为rad/s而不是角度误差。

完整10s驻留峰值均值230／250／305／430mm为64.83／13.26／20.59／28.79cm，合格窗口分别8／16／69／171；前窗分别4／15／65／190。230mm样本极少，不能用该条件化小样本独自证明整体持续恶化。frame覆盖充足的静止均速显示低姿态局部恶化、305／430略有改善，整体精度仍弱。230mm近期失败15／207结束回合，原因均nonwheel_contact；430mm11／221，其中fall原因9、nonwheel_contact3可能重叠，不能相加成12次失败；boundary截断另外统计。

41组总失败从258／9726（2.65%）至239／9802（2.44%），其中fall原因67→57、nonwheel_contact197→184；原因可重叠。boundary2214→2036是独立截断账目，不算摔倒。总体失败率下降与部分精度恶化可以同时成立。

延迟强度仍1.0、最高obs80ms／目标60ms，LR固定1e-4，所选标量有限。最近100轮diagnostics／logger／progress I/O中位0.213／0.065／0.011s，主要开销仍在环境步进；没有评测等待或多个PPO worker竞争的证据。当前6.54s是较晚连续运行窗口，与10:36冷启动后短窗不同，不能将全部变化归因于某一优化。求解器仍position64／velocity32，未运行32／8或12／6对照，也没有通过该改动获得提速。此次仅更新结果文档和部署观察回执，PPO持续运行。

证据：[完整200步TB及41组差分](../reports/v6_task_campaign_20261001/learning_audit/v6_followup_20261002T102629Z.json)、[18:34存活和时钟](../reports/v6_task_campaign_20261001/learning_audit/v6_followup_20261002T102629Z.live.json)。
