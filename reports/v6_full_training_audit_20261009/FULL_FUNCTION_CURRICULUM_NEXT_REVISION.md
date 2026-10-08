# 完整功能覆盖与下一版最小补课建议

日期：2026-10-09。只读审查实际 source029，未写合同、未运行仿真/PPO、未远端操作。本文是候选设计，不是部署回执或能力验收。

审查根目录：`/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009`，冻结 commit `e5bd9c1367b875d7f229b9a1930647013d023372`。课程是 `contracts/v6_task_campaign_v26.json`、`contracts/v6_integrated_runtime_recipe_v26.json`。计划 SHA `db693283b744400ed5b6b586055f1514b56aea9106580c40a3bc3ee882d36d92`，配方 SHA `d1a1de9829bb1c798f627e6605d4cc57ca669b400136b568e46b9b73708df2c5`。source029 仅修派课、恢复来源认证与跨源码目录绑定，没有增加新的课程。

## 1. 最重要的结论

当前单一 actor 的串行主链是 **speed → terrain → jump → stepjump**，后段混采前段复习。普通大台阶训练却仍缺席：四个 composed 域的 `body_climb` 组数量都为 **0**。已有 200/450 mm 平台上的恢复出生，以及 200+250 mm 主动跳阶，不能代替从原地面连续前进、抬车体、搭台、收腿的普通上阶。

这是 v23 小阶修改明确留下的待补专项：`small_stair_revision.py::sampling_patch` 把整个旧 `untriggered_steps=0.42` 池替换为 5/10/20/40 mm 小阶，设计记录 `high_step_training=deferred_not_deleted_from_fixed_evaluation`。旧 v15/v17 大阶代码/固定评测仍在盘上，并不等于 v26 actor 正在练这些组。

当前 source029 已证实进入 speed 的真实 PPO。2026-10-09 首 actor 核验只证明训练执行正确；不能据此声称 terrain、jump、stepjump 的这条新共同谱系已经完成。历史各专项分支的能力成绩也不能拼接到当前一个 checkpoint。

## 2. 已冻结的覆盖与缺口

| 用户功能 | v26 配方真实覆盖 | 不能据此得出的结论 / 缺口 |
|---|---|---|
| 静止、低/中/高速、加减速和停车 | speed 189 组，80% speed+20% REC；命名静止池合计15.2%，另有普通命令站立概率；release-stop 占4%，平移2/3/4/5 m/s、正反向旋转和高度变体 | 组名/份额不是实际目标曝光；还有slew、初态接稳、探索和脚本PPO排除。需要均值动作同协议速度/姿态/停车结果 |
| 5–40 mm 普通小阶上下 | 8组：5/10/20/40 mm各上下方向；每条3阶、踏面600 mm。小阶占terrain全batch **24.9231%**、jump **14.9148%**、stepjump **11.8700%** | 40 mm指每级增高，三阶顶面120 mm；不覆盖单级200 mm。配方已映射SCUT低速楼梯区间0.5–1.5 m/s，但必须看实际slew/限制后命令，不能仍引用旧v23模板0.3–0.75为当前真实命令 |
| 普通单级200 mm、双级200+250 mm | **没有body_climb训练组** | 当前不应标为已训练或已学会。大阶评测保留、平台REC出生、跳阶均不是该任务 |
| 倒地自起→RL接稳 | REC54组包含侧卧、前后趴、非对称腿、腹/背腿上举、平台/台阶初态；speed/terrain/jump/stepjump均复习 | 使用encoder/IMU恢复脚本FSM，满足prepare条件后0.2s力矩混合再pureRL；不是“脚本结束瞬间全力矩切RL”。脚本/混合不进PPO；接稳1s后释放后续运动。成熟prefix配比20/30/50。现有RecoveryOutcomes已有按等级计数；original_chain/L2还含upright和platform出生，不能把全部L2称为完整倒地链。真正倒地要再按出生姿态筛分母 |
| 原地跳 | 3/6/10/15/20/25/30 cm；已有2000/6000 actor的bin概率课程、300 actor助跳衰减和物理落地成本 | 有动态任务分布/辅助，不应泛称“没有动态reward”。预载/推蹬/收腿核宽仍固定；辅助样本不能算无辅助验收 |
| 行进跳 | 命名行为`running_jump_03/06/10`，但真实训练采样器对三行均采3/6/10/15/20/25/30cm；jump component actor0/2000/6000时15–30cm条件概率合计25%/55%/80%。固定前进0.4m/s、水平目标0.15m | 高bin目标已有，尚缺真实按bin曝光、无辅助成功率和高bin固定评测证据，速度域仍窄；`move_stop_jump_resume`先停车再跳，不是行进跳。CPU目标可采不等于实机或仿真动作达成 |
| 跳上台阶 | 单级3/6/10/15/20/25 cm，manual/laser；double200+250，踏面[0.8,0.6]、顶面450 mm | 现有双级完整难度从该阶段开始就有正配额；不是先学普通大阶，更不是普通大阶成绩。需要分第1/2级、未辅助整链单独结果 |
| 特殊地形 | slope_up/down、cross_slope、rough、material，easy/medium/full固定几何行及高度联合组 | 这不是成功率晋级或几何实时课程；当前非小阶表面命令域较保守，没有完整SCUT高速楼梯1.5–2.7 m/s对应；actor35D没有地形raycast |
| real2sim和DR | 轮侧响应、物理/策略延迟抖动/丢包、摩擦、质量/部分COM等已有 | 未qualified的腿侧bag不能标成腿响应已接入。载荷/质心范围、闭链气簧和关节响应是否充分仍要独立证据；本文不修改这些参数 |

实际有效component份额因重复组去重和永久静止floor而不同于名义mix：terrain=(0.604396 terrain,0.247253 speed,0.148352 REC)；jump=(0.4375 jump,0.234375 terrain,0.1875 speed,0.140625 REC)；stepjump=(0.393782 stepjump,0.139896 jump,0.186528 terrain,0.139896 speed,0.139896 REC)。新增任务后需重新认证，不能沿用旧sum/allocator回执。

## 3. 推荐下一版第一修：普通大阶，保留小阶和原复习

### 3.1 配额来源明确，不增加总体环境数

保持4096环境、单一35D/81D actor/critic、50 Hz策略/200 Hz物理PD、现有160/2.5、闭链与轮响应。保持整个speed段已有配方；新普通大阶只进terrain及其后续复习。

建议在terrain**原生source池**内做以下一次冻结变更，保持总和1：

| source池 | 当前份额 | 候选份额 |
|---|---:|---:|
| untriggered_steps：原8个小阶 | 0.42 | **0.42**，逐组份额也保持 |
| surface_traversal：坡/粗糙/材质 | 0.10 | 0.02 |
| surface_height_joint | 0.04 | 0.02 |
| 新 ordinary_body_climb | 0 | **0.10** |
| 其余所有standing/height/translation/yaw/transition/release-stop复习 | 0.44 | **0.44**，逐池保持 |

若component选择/静止floor机制保持，terrain全batch小阶仍24.9231%，大阶约 **5.9341%≈243 lanes**；表面遍历+高度联合从8.3077%降为2.3736%。小阶/平地/恢复复习不为大阶让路，坡面/rough学习预算确实减少。这是明确的资源取舍，不是“免费补功能”。后续jump/stepjump新增大阶复习约3.5511%/2.8262%，须由新实际compose结果确认，所有整数lane和有效PPO帧重新计数。

坡/rough30条现有行不删，按原相对份额缩放。此安排使一些细分行lane很少；旁路重点看五个地形族的总体有效样本。若粗糙地形已有能力未保持，先在同actor后段地形巩固窗口把surface权重恢复，不额外声称已qualified，也不偷偷新增预算。若要求旧rough的**原份额也完全不降**，在固定4096/固定总和1下不能同时新增大阶份额：必须明确增加训练预算或另分时段；不能让所有保底份额相加超过1。

### 3.2 一次建场、八个预建任务，有限时钟逐步放行

不要恢复旧v17“全部难度立即正向冲阶”的原样配方。复用其物理几何、原生普通技能和 `StepClimbTask`，预建以下8组；候选大阶池内部暂用**各12.5%**，约terrain每组30 lanes；不要扩成高度×正反向×触发源的大笛卡尔积。

| 组 | rises_m / 顶面m | treads_m | 新大阶课程actor相对时钟：开始→完成正向命令ramp |
|---|---|---|---:|
| 单级50 | [.05] / .05 | [.8] | 1000→1200 |
| 单级100 | [.10] / .10 | [.8] | 1400→1600 |
| 单级150 | [.15] / .15 | [.8] | 1800→2000 |
| 单级200 | [.20] / .20 | [.8] | 2200→2400 |
| 双级50+62.5 | [.05,.0625] / .1125 | [.8,.6] | 2600→2800 |
| 双级100+125 | [.10,.125] / .225 | [.8,.6] | 3000→3200 |
| 双级150+187.5 | [.15,.1875] / .3375 | [.8,.6] | 3400→3600 |
| 双级200+250 | [.20,.25] / **.45** | **[.8,.6]** | **3800→4000** |

以上时钟是提案值，按**新普通大阶引入时冻结的terrain component actor origin**计相对更新；不是用消费时钟、episode次数或旧terrain专项2206直接从成熟末端跳过去。旧speed/REC/height/delay时钟仍按已冻结继承，不能reset退款。实际实施须把origin和component clocks写入新manifest；未来阶段不重置大阶引入时钟。

所有高阶lane未放行期间必须有真实普通预训练：在实际起点的平地approach pad做小幅前后移动/制动/高度保持（可复用已有命令sequence/slew/停车评价），保持pure-RL样本进入PPO；或明确配置有限的站立预教。建议周期5s、+0.1 m/s 1s、−0.1 m/s 1s、0 m/s 3s，名义高度305 mm。必须记录实际位置，验证轮/车体没有提前碰到未解锁的台阶；若有，标为提前困难曝光，不能用传送/位置投影掩盖。不能把“未启用”实现成永远零指令、脚本控制、或无效PPO mask。

到了各组开始时，停止approach-pad循环，向正常前向命令0.30 m/s ramp，保持前向行进，无JUMP、无主动台阶命令FSM。20s单级、32s双级；初始vx用0.3以便归因，后期再拓到0.15–0.6，不能初期同时随机全高度、加大推扰和摩擦极端。大阶策略自行利用腿角抬车体/收腿，**不由StepClimbTask生成动作**。本提案不改气簧曲线、PD或力矩限值。

最小新增代码应只有命令采样处按明确clock选择有限pretraining sequence/已有`scheduled_command`的普通前进模式，复用当前采样与参考逻辑；不引入新机械求解器、奖励框架或每40步worker。`SkillCommands.sample`当前`scheduled_command`用global actor时钟且随后planar采样可覆盖命令，不能直接贴`start=1000`而假设已按terrain时钟生效。新增绑定须显式指定component/relative origin，且pretraining与activated都走同一个命令入口，避免双重schedule。几何只startup预建，episode内不移动平台。

预算短时的闭集规则：根据实际已扣账的domain余量B，在入场时冻结总有效actor allowance，最多10%预教、约25%完成八组ramp，末级完成放行不晚于该allowance的35%；至少后60%让完整200+250实际练到。上表适用于B足够容纳≥4000引入+完整巩固。若余量不足，不承诺完整训练；减少冗余中间行/缩放时钟要形成新冻结表，不能让末级直到cap才第一次接触，也不能追加未授权消费。

### 3.3 复用现有普通上阶成功定义，单独报告姿势

`StepClimbTask`是物理评估/奖励，不是命令FSM。它支持1/2级累计顶面、合法车体底部接触、每级双轮上平台连续settle0.35s；连续失去轮或合法车体支撑≥0.08s标成flight_violation，不能以主动跳替代普通爬阶。合法顶面车体支撑按总上向力≥20 N与0.1s记录；该支撑奖励是有限进度credit，不能驻留一直刷分。当前普通completion**允许车体支撑但不强制必须发生**。

对用户要求的“先搭台再收腿”应额外报告该姿势是否出现：body_support_s/peak、各级双轮support、收腿/伸展轨迹、飞行违规、横向偏移与结束姿态；如果必须执行这一具体接触姿势，再独立冻结最终姿势评判，不能把普通完成率直接称为“已执行搭台”。小阶滚过去不必被迫车体摩擦。原/新增reward复用进度、宽/精轮抬升、有限body支撑、姿态和横向项，首轮先不同时调权重，避免失去覆盖变化的归因。

## 4. 其余最小补强，分开冻结

1. **高速**：目前已混采静止/低速/高速，不应再分出“只高速actor”。问题候选是0→360 actor内已经放大到5 m/s、3 rps和230–430 mm全高度，较快；可将后续新引入高幅边界改为约0/1000/2000/4000 actor的2/3/4/5 m/s与对应yaw渐进，并把已学会速度保底保留。正式修改前用同checkpoint有效命令trace和均值动作旁路验证；不能因核验器首PPO通过就宣告姿态改善。release-stop现有4%并非缺项，需看真实零指令驻留后的速度/姿态，不用移动任务累计位移当漂移。
2. **自起**：保留真实恢复脚本→0.2s blend→pureRL。成熟的20/30/50 prefix复习已有，不再发明第2套自起器。现有RecoveryOutcomes已有按等级计数，需要补的是script进入BLEND、pureRL接稳、1s连续通过、运动释放的事件与有效曝光。original_chain/L2包含upright、platform等初态，真正倒地整链必须用L2×倒地出生姿态的交叉分母；现有原生fallen固定用例仍须保留。单纯放松连稳判据或把轻松站稳出生算倒地链会美化结果，不能解瓶颈。
3. **原地/行进跳**：实际sampler已让三条running行采完整3–30cm；先补按实际请求bin的有效曝光、无辅助完整成功和15/20/25/30cm固定评测，不再新增第二套同目标课程。未来再独立冻结行进速度0.2–0.8m/s的渐进扩展；现有vx固定0.4m/s，不能宣称已覆盖更宽速度。jump阶段三行均用jump component时钟，stepjump复习中的03/06用jump、10由stepjump拥有，报告条件概率必须写明所在domain和实际component。`move_stop_jump_resume`始终单列“停车后跳”。详见[真实采样核验](RUNNING_JUMP_SAMPLER_COVERAGE_CORRECTION.md)。
4. **主动台阶跳**：manual/single-beam触发和双级200+250已有，但相应source阶段一开始就上完整双级。复用同样预建易/中/满几何与有限全局actor放行，单级→双级，再归还既有平地/恢复/普通阶复习。普通body_climb与step_jump的completion、flight、触发、模型成绩分开。
5. **宽到紧reward**：当前预载/推蹬高度核35 mm、vz核0.5 m/s、收腿核40 mm仍是固定值；落地已有broad+fine成本，不是只有窄核。若实测早期收腿/发力响应远超核宽，可新增按jump component actor时钟从宽到紧的少量参数（例如tuck80→40 mm），保留宽尺度误差项，并做同状态reward探针和真实回放。先补任务覆盖，不把覆盖与多项奖励/探索/观测同时改完后声称某项因果成立。

## 5. 预算和验收：不要再漏账或漏难度

`first_actor029_001.json`保存的同一核验时点为消费41406、actor41226，总余量79594；domain消费flat13020、REC14502、speed5106、terrain2206、jump5572、stepjump0。它是2026-10-09核验快照，不是持续更新状态。

按每域20000封顶，上述时点剩余paid allowance：speed14894、terrain17794、jump14428、stepjump20000；critic适配/故障/历史误派课也已扣费。新共同各段目标15000 actor并不保证每段实际拿满15000：speed/jump会更早触及消费cap。下一版要从实际封存account重算，而不是旧8500、源028旧41374、或v26静态PPO_updates=0字段推断资源。首次预检只做必要的新命令/新大阶几何验证，沿用未变机械工程先验；不为所有旧组重跑大场景。

保留500 actor非阻断旁路review，训练按冻结时钟/预算自动推进，不恢复45项全过才能继续的门禁。旁路小套件至少各测：静止3高度、±低速/±高速/yaw、真正释放后的停车、原完整REC、5/40 mm普通阶、200 mm普通阶、200+250普通阶、stationary/moving jump、200+250 active jump。报告样本数和有效曝光；普通/跳跃分别判，短回放脚本未结束按截尾，未测记未测，qualification仍null直到正式证据支持。

新源、新recipe、新plan和迁移回执需一起冻结，旧合同/回执原字节保留，actor/critic/Adam/std从最新共同谱系恢复；不热改source029、无退款、不得并行重复owner。

## 6. 可核验入口

本报告读源不执行测试。源码均相对上面的e5bd9冻结根：

- `src/wheeled_tasks/chassis/small_stair_revision.py::sampling_patch/revised_design`：原大阶42%池换成小阶；明确后置大阶。
- `src/wheeled_tasks/chassis/integrated_sampling.py::COMPONENTS/_compose`：各段复习、去重和静止floor；须重核新配额。
- `contracts/v6_integrated_runtime_recipe_v26.json`：4段实际scene_groups/skill_specs/row_ownership；body_climb为0。
- `src/wheeled_tasks/chassis/step_climb.py`：普通爬阶的支撑、有限credit与80ms腾空排除；无命令输出。
- `src/wheeled_tasks/chassis/skill_curriculum.py::skill_spec`：body_climb禁止trigger_source，mode0；跳阶mode4分别构造。
- `src/wheeled_tasks/chassis/skill_commands.py::sample/scheduled_command`与`training_schedule.py::actor_update_clock`：现命令ramp具体clock/采样顺序，不能错接时钟。
- `src/wheeled_tasks/chassis/recovery.py`与`recovery_training.py`：脚本、blend、pureRL与稳定释放；脚本/混合排除PPO。
- `src/wheeled_tasks/chassis/rewards.py::reference_jump_terms`：固定jump核与raised landing替换；`jump_landing.py`有限物理落地成本。
- `src/wheeled_tasks/chassis/scut_observation.py`：35D encoder/IMU/command布局，不含terrain/raycast/root速度。
- 本审计目录既有`scut_source_comparison.md`、`curriculum_audit.json`、`first_actor029_001.json`为历史审查/首PPO证据，本文不覆盖它们。
