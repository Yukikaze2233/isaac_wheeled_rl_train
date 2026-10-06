# V18 姿态出生与推扰补强及实际接续

日期：2026-10-06。状态：**已在 Kaiser 部署并确认实际 actor PPO 学习，尚未获得新增行为达标或实车资格**。冻结源码为 `4ba339299c203cf2aba77183ec81f2f017dc9bf9`；07:09（UTC+8）启动，07:14 的原生核验通过 25 项。实际身份以主运行回执 `reports/v6_task_campaign_20261001/deploy/launch.json` 为准。

## 需要补齐的机制

SCUT 公开保存的 rough_dash（2026-07-19）与 rough_rotation_stair（2026-07-30）启用了普通出生 roll/pitch ±0.15 rad，以及间隔 5–10 s 的平面速度推扰 ±0.25 m/s。它们也有 pitch/roll 指数奖励、姿态平方成本与真实的横向角速度成本。`flat_orientation_x/y_v` 是速度相关的姿态角成本，不是角速度成本。

我们的 speed v14 已实际接入 pitch/roll、角速度、动作变化、静止速度奖励，毫米到数十毫秒的信号年龄及抖动随机化、非标准腿姿态与短暂零力矩、大场地及低高速混采。普通出生已有 5% 的速度边缘初态（前向速度与三轴角速度），仍缺普通车体小倾角；推扰执行路径存在，但现有 skill 幅度全部为零。补强应针对这两个采样缺口，不能再次写成“姿态奖励未付费”。

## 本轮已部署范围

五个任务 worker 增加 `ordinary_posture_curriculum`，使用各域独立 actor 时钟、训练出生和原生地面推扰。出生只在 reset 写入，需核查机构闭链、轮子/地面净空与车体支持高度；推扰限定普通地面动作，不覆盖主动跳跃及脚本自起接管。普通出生倾角采样概率为 50%，roll/pitch 从 ±0.03 rad 渐入至 ±0.15 rad；平面推扰幅值从 0.05 m/s 渐入至 0.25 m/s，独立间隔 5–10 s。幅值使用各任务接续起点之后的 actor 时钟，前 1000 次保持起始难度、1000–3000 次线性加大，critic 适应不推进该时钟。碰撞检查拒绝的倾角组合明确记录，不借助运行中姿态重写救场。取消倾角后，若合法原姿态仅缺保守包围盒的 1 mm 净空余量，出生高度保留该余量，允许轮子离支撑面最多 1 mm 后由正常物理落定；真正非轮碰撞仍拒绝。

原生坡地检查另发现：flat-ground 的非标准腿库不保证每个坡度上都能由轮子先支撑。取消车体倾角后若腿库仍有非轮碰撞，出生写入前回退认证的名义闭链坐标，并取消该样本的非标准保持／接管状态；采样次数与取消次数分开记录。禁止用放宽碰撞阈值或运行中重写姿态掩盖该问题。

v18 从实际 v15 计划直接接续，v17 仅作为未启动的课程修复生成配方。本轮同时部署基础复习、地形与命令绑定、跳跃重复周期证据清零与恢复前缀课程修复；已消费更新不退款，学习来源仍为各域实际封存状态。speed 以原 v14 worker 为基础，不借用跳跃配方。

恢复按原生前段逐渐教学：近直立接管 → 中等倾角释放 → 原始困难整链，显式 SCRIPT、BLEND 与短暂零力矩区间排除 actor 学习；保留物理终止的收费和禁止 bootstrap。地形按易、中、完整几何行采样，并绑定相应命令；任务经验之外保留完整基础行为复习。跳跃与台阶跳每次接受请求时清空上一周期的腾空、COM、离地和落地证据，避免连续请求借用上一跳成绩。原地与行进跳跃、普通上阶与腾空台阶跳仍是不同任务与评测口径。

固定评测恢复实际 v15 来源的原始行为目录、配额、基础奖励和恢复设置；新增姿态出生/推扰、非标准出生、助跳与训练松杆实验全部停用。原 case、阈值和典型出生不变；主动跳跃保留接受周期的观测证据清零，不能将上一跳证据当下一跳成功。

## 停车与高速审计结论

speed v14 实际为 135 个平地行为组：基础 61.75%、高速及过渡 33.25%、独立松杆停车 5%。松杆有随机等待、4–8 s 驱动、12–16 s 零请求驻留，并保证制动预算后 2 s warmup 与完整 10 s 驻停统计。重采样不会重启这段轨迹，驻停统计不会因实际速度过大而排除坏样本。场地半长 160 m、宽 8 m，实际边界半长 156 m、半宽 3.7 m。没有找到“停车相位未采到”“零速奖励没接”或小场地默认截断的现行代码证据。

已有 pitch/roll 速度尺度下限 3/4，比 SCUT 高速时趋近 2 的尺度更严格。本轮不凭主观观感继续加大姿态权重，也不强制每次加速车体精确水平。

同一封存模型的独立 mean/stochastic clean 物理诊断显示，探索噪声增加抖动，但 mean 策略仍有零速偏差与累计漂移。actor 完整转移和 resume 保留 learned std，`initial_noise_std=1` 只控制新构造器初值。本轮不清零或热改 std；35D/81D、网络结构、50 Hz 策略与 200 Hz 仿真反馈 PD、轮响应及未 qualified 腿 bag 的状态保持。

## 验证与部署回填

- 固定评测的五域 × 四角色逐项与实际来源匹配，拒绝改源 SHA、改 case 阈值、串域及借用未启动 v17 学习来源。
- 启动前源文件审计绑定新增 helper、配方、历史依赖、实际源合同与独立 v18 design。
- 原生工程探针需确认实际启用、姿态出生合法、地面推扰生效及跳跃/恢复路径排除；工程通过不代表动作达标。
- 部署后用不可变同模型的 mean 策略核查静止、完整停车、高速姿态与基础保留；带探索训练曲线不能替代固定行为结果。

冻结归档 SHA-256：`f93a7ef910a51bb151de13401e01947c075688f5479181c8ff45c753de935715`；计划 SHA-256：`027de0e63832b3821d7a1686e8bbc3e1a2662094476c06c3f71193fed59ec022`。

- 同一冻结归档的 68 个选定测试文件：1643 通过、3 跳过；原有 32 项历史排除及其理由保留，不声称整个历史测试仓库全绿。新增四套测试完整纳入。
- 本机与远端 30 角色预检通过：五域训练实际启用新增机制，20 个固定评测角色保持原评测隔离，flat worker 不变。
- 158508 个有限几何组合通过，包含实际坡地失败样本与非连续环境 ID 回退复现；保留真正碰撞拒绝。有限枚举不代表所有连续姿态都已证明。
- 五域各完成 4096 环境 × 600 策略步、2400 物理步、12 s 仿真；倾角 root quaternion 原生回读、推扰速度原生回读、实际人口、源码快照及每周期四次反馈 PD 通过。terrain/recovery 在 Kaiser，jump/stepjump/speed 在本机独立 GPU 完成；冗余远端工程进程已单独结束。这些工程检查不消费 PPO，也不认证动作性能。
- jump/stepjump 的接受周期清零均实际观察到；工程报告完整保留，不能把退出码为零或 preflight-only 当成 rollout 通过。

完整入场证据：`reports/v6_scut_posture_completion_20261006/verification/native_population_018.json`。入场后四份固定证据保持不变；运行状态另写回执，不修改冻结输入或启动第二条训练链。

审计机器文件：`reports/v6_scut_posture_completion_20261006/high_speed_stop_posture_audit.json`。

## 预算与学习来源

旧运行已完整封存：累计消费 29205，恢复 actor 2294、高速 5063、地形 2186、跳跃 5552。v18 从每条线自己的最新不可变封存状态恢复完整 actor、critic 与 Adam；stepjump 在 jump 新分支完成后接续。五条修订线各增加一次 10 更新的 critic 适应，既有消费和历史结果全部保留。计划 actor 终点为 15000，flat 已完成的 13000 及模型保持封存。按没有额外故障消费、修复或人工提前审核的口径，新增计划消费 59955，累计 89160，低于 121000；每域仍受 20000 消费上限约束。

同合同故障恢复使用最近不可变 seal 的剩余 warmup；硬崩溃丢失的未封存更新不退款，不能承诺任意故障下历史 critic 实际消费严格只有 10 次。

## v18 首次接续与自动推进（历史记录）

原 run root 保持：`/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358`；tmux `v6-task-campaign-20261001T110045Z-282358-r18`。唯一 wrapper/supervisor/worker 为 3471693 / 3471702 / 3471715，当前 `recovery/attempt_002/train`。后续自动顺序为 recovery → speed → terrain → jump → stepjump；各线保留自己的策略，不把恢复课程权重覆盖到高速线。

恢复父模型 SHA `2119674b64db65c0570a4b093067430cba90ceecdcd28c540d97c76b7f35572f`，其学习封存计数 16334、恢复 actor 2294 与当前总消费起点 29205 是不同的时钟。完整 actor/critic/Adam 初态逐值一致；第 29206 次消费只改变 critic，actor 保持；第 29216 次消费完成首个 actor 更新，9 个 actor 张量改变。07:14:30 实读：新增成功更新 13（10 critic + 3 actor）、恢复 actor 2297，总消费 29218。

`deploy/first_PPO_verification_018.json` 的 25 项检查全部通过，CRC 合法 event 尾窗确认实际姿态出生、推扰、既有静止速度及姿态成本、正姿态奖励、delay/transport 与 Timing 指标。此为实际训练学习，不是只跑推理测试；新行为是否改善仍需同模型确定性回放和后续完整行为窗口验证。

保留 4096 环境、35D/81D、6 动作、50 Hz 策略、200 Hz 仿真反馈 PD、160/2.5、fixed LR 1e-4、轮侧响应与延迟随机化；未 qualified 腿侧 bag 没有接入。固定评测阈值不变，训练不由在线 45 项 gate 拦停。各任务到冻结 actor 终点或合法提前审核后封存并自动接续；`qualified=null` 不因完成迭代而变为通过。

TensorBoard 在 Kaiser 运行，本机仅端口转发：[6006](http://127.0.0.1:6006/#scalars)。6007 属于其他实验，不作为本次曲线入口。

## 时间开销的当前证据

首轮核验的 CRC 尾窗只有 7 个早期更新（含 critic 适应）：wall 17.02–19.67 s，环境 13.32–14.71 s，optimizer 0.08–0.18 s。Kaiser GPU 同时存在另一项 frame 实验；这些样本不能当成熟吞吐或把时间差全归因于某个模块，也不承诺本轮已提速。

只读源码对照确认：本轮 200 Hz 物理子步循环 AST 与实际 v15 相同，没有将 FK、mesh 加载或新增姿态 helper 放入每个物理子步。新增工作在 reset 几何审核及推扰采样/回读；timer 统计的 CUDA 标量同步和空样本分支存在进一步 profile 的候选，尚未量化占比。当前归档保持不变，性能分析不以热改合同或重复启停实施。

## 首个完整学习窗口与源码修复 019

首个 100 actor 窗口为恢复 actor 2295–2394、event 29215–29314；排除前十次 critic 适应，每个环境经历 48 s、共 9830400 个观测帧。CRC 顺序读取确认全部必需指标完整，Behavior 按生产频率只有 11 个累计端点，不能误称 100 个 Behavior 端点。带探索训练读数尚未达到基础精度：九档静止平面均速 7.72–10.67 cm/s，前后 0.5 m/s 速度 MAE 29.52 / 28.28 cm/s，正反 yaw **角速度** MAE 0.589 / 0.458 rad/s；六个真正 settled-stop 条件窗口均速 8.66–10.17 cm/s。100 更新 wall 中位 17.878 s、环境中位 15.073 s。策略 std 均值 0.373，不据此认定确定性策略遗忘或新增达标。

230–400 mm 没有满足生产选取条件的完整 10 s 静止窗口；430 mm 有 72 个，峰位移均值 43.21 cm。零样本不能补成零漂移；样本窗口可能跨越 actor 窗口边界。恢复前段、释放前段与 original-chain 单独记账，静态恢复组的结束计数不能作为困难整链完成率。完整原始回执和统计口径见 `learning_audit/recovery_first100_018.json`、`recovery_first100_018.summary.md`。

独立覆盖审计发现 v18 的 recovery 保留旧 pool 名 `translation/transitions/yaw`，普通姿态白名单只有 `basic_*`。因此 30 个普通运动复习组、872 个环境遗漏倾角与轻推；它们的任务、奖励和 PPO 样本仍存在。原实际普通资格只有 577，不是完整基础人口 1449；其他四域实际 3928/4096/1231/1228 不受别名缺失影响。恢复当前没有 `nonstandard_start` 腿库实例，不能把修复静态资格称为新增实际腿库 hold/capture。

修复源码冻结于 `712d701473f38ef6797725f8efdda9e6beb85e87`，补齐三旧别名，脚本恢复、旋转特殊模式和跳跃请求仍排除。独立明确目录与五域实际人口断言覆盖上述范围；201 个相关 CPU 测试通过，20 个显式 CUDA 微测试未在该 CPU 集合运行。另已完成 41 项 CPU/CUDA 语义对照，验证 timer 统计合并传输及空普通推扰分支优化保持随机数序列、全量统计、原生回读断言与状态写入。

该修复使用命名的 **source revision 019** 接续，保持 v18 计划、六个 worker、actor 终点、消费上限和所有课程 origin 原字节；从最新 own-recovery 完整封存状态恢复 actor/critic/Adam，同合同不重复 critic 适应，不回滚、不退款。新冻结源码需要自身的 30 角色、受影响测试和 4096×600 原生验证，不能拿旧 4ba 报告换顶层 SHA 充当新入场证据。候选与实际部署回执分开，实际是否接续以主 launch.json 和 `source_repair_019/` 的运行记录为准。

## 实测性能归因

Kaiser 运行中采样因 ptrace 权限被拒绝，未改系统权限、未暂停 PPO。随后在独立本机 4060 上对同一冻结 v18 恢复 worker 做 4096×600 原生采样：环境步进约 67.6% 的采样时间位于 PhysX step/fetch-results，约 6.4% 位于原生 COM 读取。这些调用包含同步和等待，不能把它们全部归因为求解迭代数；inclusive 分类重叠，不可相加。普通姿态 helper 约 0.06%、apply_push 约 0.14%，不是主瓶颈。

最小同步优化的单次 A→B 对照为 181.194→180.620 s，差约 0.3%，不足以证明有效提速，更不能外推 Kaiser PPO 吞吐。当前求解器是 64/32，尚未建立该配置在 V6、5 ms、闭链接触下的最低必要性；后续降低迭代必须以独立实验合同核查闭链、接触、冲击、姿态与制动质量，不能直接采用旧 1 ms 或其他模型的 12/6 结论。性能证据见 `performance/baseline_profile_summary_018.json` 与 `candidate_profile_pair_018.json`。

## 019 入场核验与当前封存

新冻结源码自身的 30 角色在本机与 Kaiser 通过；恢复原生 4096×600 策略步／2400 物理步确认普通覆盖 1449、出生倾角 root quaternion 与 2471 次轻推扰原生回读，最大闭链残差 1.897 mm，未消费 PPO、未认证行为资格。最终完整入场报告与归档字节逐项绑定，独立只读部署审核通过。

2026-10-06 08:04:06（UTC+8）旧 v18 recovery attempt_002 于 actor 2458、总消费 29379／actor 消费 29239 完整封存；该段新增 174 更新（十次 critic、164 次 actor）。唯一信号发给已认证 supervisor，未触碰其他实验。原生 wrapper 自然执行 finalize 全量交付归档，prepare 会拒绝尚活的 wrapper；目前等待归档，不把候选或封存状态写成新 PPO 已启动。实际接续结果后附。

019 于 08:07:30（UTC+8）在同一 run root、tmux r19 启动 `recovery/attempt_003/train`。完整父模型为 actor 2458 的最终封存 SHA `1410b6bbe7edcd4268ef3cabcdd871c114a6b3d9908d923e252717fdfe59f195`。新唯一 wrapper/supervisor/worker 为 3480202 / 3480207 / 3480218，实际包装日志编号为 `train.resume_016.log`（不是 source revision 编号，不能由 019 拼造日志名）。

初态 actor/critic/Adam 逐值一致且 warmup=0；第 29380 次消费是首个 actor+critic PPO，九 actor 张量、critic 与 Adam 均变化，学习谱系与 actor 时钟各增加一次。当前已验证实际普通人口 1449 与保留 origin，仍等待第一个原生轻推周期；首PPO验证回执此时 19 passed／1 pending／0 failed。不得把这些实现验证或首步学习当成漂移、欠速或接稳性能已修好。

08:12:20（UTC+8）首PPO完整核验 **20 passed／0 pending／0 failed**。实际 CRC event 中普通覆盖一直为 1449、原生推扰速度增量最大约 0.04995 m/s，保留课程相对时钟和既有基础姿态／静止成本／延迟；不把 logger 平滑后的 push_events 值当精确整数事件数。最新只读进度为 recovery actor 2470、本段 12 次全部 actor 更新；账本消费 29392 可能含下一步预留，实际完成消费 29391，两者单独记录。

旧 worker 于 08:04:06 完成学习封存，旧 wrapper 于 08:06:23 完成交付归档，新 tmux 于 08:07:30 启动，首 actor 更新于 08:09:25 落盘；这是本次实际冷停与恢复时间线，不能全算成 PPO 每步时间。全量归档入口每次对历史训练文件重哈希、重压缩，后续批索引交接模式可单独优化，但本次没有跳过归档。

## 019 首完整100更新窗与接稳诊断

完整窗口为 task actor 2469–2568（event 29389–29488），每环境48 s、9,830,400物理观测帧；最新50窗包含于此100窗，不能相加。两窗按原生生产 cadence、TFRecord 双CRC与全部必需tag认证，缺项为0。原始数据 `learning_audit/recovery_window_019_20261006T0042Z.json` SHA `18e4aa8e0b1c1644d63d9bee47fb549acd6e138f25e64fd175bd3b19c934a2d3`，完整表在同名 `.summary.md`。

带探索读数：230/305/430mm静止平面均速11.05/9.68/8.62cm/s，前后0.5m/s速度MAE28.5/27.2cm/s，正反yaw角速度MAE0.538/0.478rad/s；六个真实settled-stop窗口约9.00–11.13cm/s。400mm平面速度较低但高度MAE24.70mm，不能称达标。430mm完整10s合格驻留74窗，峰位移均值38.84cm、最大206.97cm；305mm没有合格10s窗，不能补零。lifetime最大倾角不是此100窗最大倾角。与旧018窗的差值含actor年龄、难度与探索变化，不作别名修复的因果增益。

现有逐门与命令原生tag另读为 `learning_audit/recovery_hold100_019_20261006T0059Z.json`，SHA `e412b1b25b84a50bf871efa438b144559c0d3d098563b0a6a28e680e07d1ccea`。100个rollout均有非零holding人口，均值2548；轮速、腿速、gyro门的单项通过均值分别约0.194/0.320/0.329，support/body_clear约0.615、tilt0.869、height0.997。这些是先按24物理观测帧平均的边缘比例，不是联合连续1s通过概率，不能相乘或与单独平滑的人口相乘反推精确计数。stable_peak_seconds_mean包含全部4096环境，均值14.23ms不能当恢复lesson的成功率。

全部54恢复静态组实际command vx/yaw均值均为零；其中26个原规格forward followup=0.3m/s也尚未实际收到行进命令。渐进lesson约88.30% prefix_stand、10.59% prefix_release、1.12% original_chain，静态组名与动态lesson并非相同分母。motion_release前命令保持零，零速接稳与后续行进曝光必须分别记账，不能用组名代替实际命令曝光。未证明探索或某奖励是唯一原因，下一步用同一新seal、实际v18出生和完整探索状态作均值/探索零PPO对照。

固定均值回放已在只含评测入口修复的隔离源码完成，两父模型保持不变：flat actor13000 SHA `7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f`，恢复actor2458 SHA `1410b6bbe7edcd4268ef3cabcdd871c114a6b3d9908d923e252717fdfe59f195`。相同63例、seed290929、每例4次：前进速度MAE1.73→26.01cm/s，后退1.23→25.75cm/s，正yaw角速度MAE0.0103→0.3095rad/s；驻停却3.18→1.52cm/s。故存在运动能力退化，但不是所有能力都退化，也不是当前带探索曲线的同义词。quick单seed通过27/63→17/63不是S1旧45项或全322项结果。

恢复native结束成功111/216→57/216，按固定时钟严格完整110/216→57/216；flat有一例完成已开始BLEND后稍超8s预算，保留两种计数。新模型52条失败已释放并取得历史稳定资格，停在post-handover任务未完成（48前进、4左转），不能仅归因为从未接稳。两文件没有replica ID，不逐行配对；案例人口只支持44–52条原完整案例转入此失败域的范围。证据 `mean_comparison_019/comparison_guard_candidate_004.summary.md` 与独立review。

CPU双时钟14例中，12个稳定正例在1ms/5ms、不同BLEND尾相下均完整通过；2个实际首20ms接触缺失的负例正确拒绝。未复现系统性20ms提前释放bug；这是恒定传感器CPUoracle，不替代原生接触与策略诊断。

评测入口原来在ordinary disabled时仍解析新训练目录，使固定用例rotate_1_reverse报KeyError；最小修复只在enabled且非evaluation_exact_cases时求eligibility，保留disabled对象、运行时schema和启用训练的严格目录检查。永久6项回归已加入，相关7文件260 CPU通过；隔离原生两模型63例实际完成。当前Kaiser仍source712持续PPO，入口修复未热改运行源码。

08:58（UTC+8）只读实时进度：唯一worker3480218，恢复actor2621，本段163次全部actor+critic；累计实际完成29542、actor消费29402。主launch的ordinary运行回读已由误留的旧577替换为实际attempt_003 startup1449（694出生采中），明确这是startup快照、不是此刻推扰累计。当前质量仍未恢复，不新增qualified信用，不导出为新可用实车模型。

求解器64/32的最小必要性未被证明；SCUT已认证配置12/6，但其资产也有六条闭合约束，不能以开链/闭链差异作解释。降迭代是物理变更，后续单独冻结诊断合同、逐5ms闭合/接触/冲击/驻停/接管对照后再测时间。COM候选已完成4096×600原生语义等价（局部reset、推扰、IMU与RNG）。四个串行fresh-process ABBA环境循环分别为179.953/167.854/168.462/180.312s，候选相对基线均值缩短6.648%，两组范围不重叠；这是本机4060环境工程循环，不是Kaiser正式PPO提速。合并固定评测边界修复后347项CPU通过（2显式跳过），同63用例/两个原检查点原生回放697次reset均验证两次数据epoch推进和world/body速度逐位一致，所有逐case指标与未加COM补丁完全相同。初次回执source集合遗漏四份静态轮响应证据，随后按原不可变archive补核字节，不改原生运行或源码。


## COM源修订020候选与部署边界

只优化startup机械随机化完成后的root COM读取，复用Isaac Lab原生Warp力学映射，按既有robot.update推进缓存epoch；保留同epoch原生写入/已读缓存语义。policy state、通信、全机构跳跃COM、高度/姿态/奖励、64/32求解器与v18全部合同字节保持原值。新模块纳入训练和固定评测源码manifest；关闭ordinary模块的固定评测不再解析不适用的训练group别名，启用模块仍严格检查。模型重写工具须invalidate→完整native setters→refresh，不能缓存旧模型。

实际部署前继续核验冻结archive、30独立roles、原生机制与唯一原root来源。源修订从届时最新完整actor/critic/Adam/RNG封存接续，所有消费与课程origin保持、同合同零新critic预热；不能退回较早的均值较好模型，也不能凭测试结果授予qualified。当前Kaiser仍为source019/原tmux r19。


## 同一封存的均值/探索接稳对照

诊断用不可变update29500/task actor2579完整seal（PT SHA7f91d355…dbcca），原source712/v18/4096×600策略步/2400物理步。两臂初始root/q/qd/速度/课程lesson/场地顺序与env RNG逐位相同，六轴保存的Gaussian std逐值不变，私有native采样流相同；均值臂使用mu，探索臂使用保存的native sample。0 PPO，不消费正式预算。原生terminal-before-close报告验证完成，根进程观测两次exit0；Kit关闭后的sidecar没有产出，不能声称app.close已返回。

| 首出生lesson | 人口 | 均值放行/成功/未完 | 探索放行/成功/未完 |
|---|---:|---:|---:|
| 简单接稳L0 | 2284 | 979/908/1277 | 0/0/2005 |
| 短零力矩接稳L1 | 327 | 140/66/229 | 0/0/278 |
| 原自起链L2 | 36 | 15/9/12 | 0/0/21 |

“未完”是12s右删失，并非20s回合失败。均值首出生失败分别99/32/15；探索273/47/15，场地截断另计。983均值成功逐lane核查真sensor/contact/followup≥1−1e−5；5条sensor放行但真实contact不足1s都被code7拦截，不能报门禁漏放。L0原非零goal人口1185，其中502放行/459成功仍followup零，这些459只完成prefix站稳，绝不授予原行进/yaw成功。L1原静止155成功59、行进159成功5、yaw13成功2；L2静止21成功8、行进15成功1。

该对照直接支持“保存的探索动作阻断此12s样本中的持续接稳和后续命令曝光”，不保证所有20s回合必失败，也不证明它是全部遗忘唯一原因。后续需要同时核查actor-owned的hold reward付款和简单目标桥：只增加motion组名或比例不能解决探索臂零放行。保持最后1s稳定/实际接触/后续跟踪门与保存std/Adam，修复方案先冻结并验证，不热改当前合同。


## 接稳奖励启用边界与下一步修复范围

独立核查纠正：`apply_recovery_takeover`与hold腿速/轮速的专用成本虽存在于源码，v18 recipe明确删除`recovery_takeover`配置，实际worker无该key，env不会调用该函数。因此不能把旧v8专用成本当成本轮已支付；普通gyro/腿速/轮速/静止稠密项仍在。两臂wrapper没有保存完整逐项reward付款，历史付款与PPO优势未知，不能由零放行直接宣布某项奖励缺失或权重不足。

后续先只读捕获actor完整拥有区间的实际最终reward components、原生单次事件和返回reward，核对`reward = policy_dt × Σdensity + events`，按lesson与hold/释放区间分箱。证据充分后优先复用现有非饱和pseudo-Huber broad+fine，考虑只补保持期的腿速/轮速/gyro目标；不重新造核、不整体重启旧takeover清零基础tracking、不手工压探索std或重置Adam。L0非零原目标一直变零的教学桥另行修复；最终真实接触/连续1s/行进跟踪门保持。source020只包含工程语义优化与固定评测边界修复，未部署新的质量奖励。

source020实际冻结348项CPU通过、2显式跳过；本机和Kaiser30独立roles通过；恢复/跳跃/跳台阶各4096×600策略步/2400物理步完成，均0PPO、qualified=null。旧平地actor工程探针的jump最大闭链20.2804mm/49次guard终止、stepjump57.2737mm/37次guard终止在新旧源码逐值一致；不得将工程入场写成所有闭链小于3mm或跳跃功能通过。现有3mmguard未放宽。

2026-10-06 09:41:34（UTC+8）source019 recovery attempt_003正常封存：新增300次全部actor+critic，task actor2758、累计消费29679/actor消费29539。信号仅发给已认证旧supervisor3480207，旧wrapper自然归档；此时新PPO尚未启动，最新部署后附。


## 020真实接续结果

2026-10-06 09:45:48（UTC+8）在同一run root、tmux r20启动recovery attempt_004。父为source019最新own recovery actor2758/总消费29679完整封存，PT SHA `7ed1d83675ba408d7558a5b61ab88be66d7be0608491c16d7b19919221e03582`；保留全部历史消费、origin、std/Adam，warmup0。唯一wrapper/supervisor/worker为3494082/3494087/3494096，实际日志编号train.resume_017.log不能由source020拼造。

实际首PPO于09:47:42完成，第29680消费九actor张量、critic及Adam均改变，初态三者与完整父逐值一致。父/初态/首seal含完整CPU/env/CUDA RNG，冻结恢复路径保留；reset/resampling会消耗随机流，不宣称post-reset RNG逐位等于父。最终首PPO21 passed/0 pending/0 failed，实际root adapter及全部源码快照、35D81D/6动作、4096/50Hz/200Hz、1449普通人口、原生轻推与奖励/延迟都已核验。

09:51:09只读实测actor2771、新增13次全部actor+critic、累计完成及账本均29692、actor消费29552。最初8条完整CRC计时wall15.016–17.751s/更新、env12.367–14.957s；这是早期共享GPU实测范围，不是Kaiser配对提速证据，也不代替本机ABBA6.648%局部结果。阶段仍recovery，原生后继speed/terrain/jump/stepjump、终点15000/各域消费20k/总121000不变，不新增能力资格。TensorBoard仍仅Kaiser6006，本机6007属于其他实验。详见source_repair_020/first_PPO_verification_020.json和主launch.json；结果提交不得重复启动。

09:55（UTC+8）继续只读核验：recovery actor2787，本轮29次actor+critic；完成消费与账本均29708、actor消费29568，原三进程身份不变。主launch中学习来源字段已明确仅表示2758初始化父状态，不把父快照称作当前最新训练输出。

## 020完整训练窗口与真实付款核查

10:16（UTC+8）只读确认原三进程持续PPO：恢复actor2861，本轮103次actor+critic，累计实际完成及消费29782、actor消费29642。其后原生事件首完整100窗已完成，actor2769–2868／event29689–29788，全部CRC与必需tag完整，每环境48s；最近50窗包含其中，不相加。每次更新wall均值17.073s、中位16.804s，环境13.988s、优化器0.128s；共享GPU和actor条件未配对，不能把时间差全归因于COM。

230／305／430mm静止平面均速11.86／10.29／7.63cm/s，高度MAE5.38／5.32／5.76mm；前后0.5m/s MAE26.35／27.20cm/s，正yaw角速度MAE0.554rad/s。真实settled-stop约9.53–11.29cm/s。恢复场景仍无成功，场景名混合动态课级，不能把这个计数直接当原自起链完成率。工程修复不等于质量修复。完整结果见 `learning_audit/recovery_window_020.read001.summary.md`；真实100窗由现有TensorBoard事件生成，不运行本机TensorBoard或额外训练进程。

同seal2579的两臂真实reward观察已经串行完成，均4096×600个20ms区间；共各2457600区间严格对账零残差，事件只支付一次、密度只乘一次dt，首次出生行为与原对照逐值相同。0PPO，不消耗正式训练预算。初级静止、首出生、完整actor接稳区间，均值／探索动作的轮缘最大速度均值0.227／0.549m/s、腿轴最大速度1.006／2.830rad/s、全轴gyro模长0.432／1.079rad/s；实收奖励0.08149／0.03220每20ms。静止车体平动罚实际为−0.07978／−0.25506 reward/s，确实生效；现有轮速罚仅−0.000343／−0.001466，腿速罚−0.01805／−0.09941，横轴gyro罚−0.01477／−0.06352 reward/s。不能再描述成全部静止惩罚未接。

上述付款支持按接稳门尺度重新定标运动成本，但没有测量GAE优势或PPO梯度，不能保证加权即可修复，也不能把正tracking baseline单独解释成激励漏洞。最后参考tick八门同时通过比例57.47%／2.97%，不是连续200个5ms物理tick的通过概率；12s未完成出生仍为右删失。两次子进程实际exit0且原生preclose结果完整，App.close是否返回仍未知。

下一修复候选限定两件事：复用既有广域pseudo-Huber＋精度核，在完整actor接稳窗口按轮缘／腿轴／全gyro门尺度替换已有弱运动税并避免yaw重复付费；初级接稳后保留原前进／旋转目标。普通环境、释放后的运动跟踪、所有正baseline与终止事件保持；1s连续传感器／真实接触稳定、1s原目标跟随和仅L2计完整链均不放松。目标桥已有独立21项CPU核验，奖励和新合同在本段审计时仍需原生验证、尚未部署、无新模型资格；后续021真实部署结果见下节。真实付款完整表见 `learning_audit/native_payment_pair_019.summary.md`，压缩保存在 `learning_audit/preserved_payment_pair_019/catalog.json`，可按SHA恢复原字节。

## 恢复质量修订021／v19

冻结源码为 `d0dce6181df83b03fe166abcbadef3c81de2e57e`，归档SHA `2c019ba1fd56053af75bf631258397176de8c5b40f36d35ccdf54d32f5b21ee9`。新增四份v19合同，只改变恢复域；原219份合同和其他五域worker逐字节保留。课程origin、人口、父树、自动终点和消费预算保持。冻结源码实际362项CPU通过、3显式跳过；本机与远端30角色逐项相同，独立迁移及观察器审计已完成。4096×600策略／2400物理步原生工程002已通过：新hold成本实际付款726560个区间，严格density×dt加事件的对账残差为0，普通和排除区间逐值保持，L0原前进及正反yaw目标均实际出现。首观察001因bound-method身份审查误拒而失败，原始结果保留；002只修证据观察器，生产冻结源码未改。原生验证有1次护栏终止，峰闭链误差4.28467mm，因此只授工程准入，不授接管或实车质量资格。

接稳成本使用现有广域／精度核，以实际稳定门尺度归一化：四腿轴最大绝对速度除以2rad/s，双轮最大轮缘速度除以0.25m/s，全轴gyro模长除以0.75rad/s。启动核查原生FSM及左右机械模型的0.06m轮半径；只在完整actor拥有且policy interval起点仍hold的区间替换已有腿速、轮速和横轴gyro成本，并去掉重复yaw成本。事件和其他reward不变；脚本、混合、普通组及释放后区间不受此补强。新增稳定TensorBoard tag记录实际hold人口、归一误差与两核付款，不能由函数存在推断实际支付。

初级L0在原连续1秒稳定和实际接触门通过后，下一policy interval继续跟踪原前进或正反yaw目标；原1秒followup时长保留。首释放interval仍执行原hold语义。简单姿态接稳只计prefix教学，完整自起资格仍仅来自L2原链；不用降低稳定门或把零命令成功转换成行进成功。原生工程验证须实际出现三个非零目标，缺任一项拒绝准入。

本次奖励合同改变，所以采用原生10次critic-only适应，全部计消费、actor课程时钟不前进；随后恢复actor＋critic。必须从切换时最新own recovery完整actor／critic／Adam／RNG封存接续，保存的探索std不重置，不退回旧平地学习状态。工程观察可使用固定平地母模型和独立2294教学时钟，但它不是正式学习父状态，不授予质量或实车资格。正式首PPO另核初态逐值恢复、10次真实计费和首actor更新；中间没有逐更新封存的actor张量不声称已观察。

4096／35D81D／50Hz策略／200Hz新反馈仿真PD160/2.5、轮侧响应与毫秒延迟随机化保留。求解器仍64/32，较低迭代尚未通过冲击／制动／接管对照；腿侧bag仍未qualified、未接入。TensorBoard仅Kaiser6006，本机6007不动。结果提交不触发重启；实际切换、首PPO与后续质量以部署回执和原生event为准。

2026-10-06 11:19（UTC+8）已由原生wrapper从v18最新恢复域封存接续v19，tmux r21／recovery attempt_005。父actor3075、总消费29996、actor消费29856、父SHA `21c3cf70caadaafe94e16a8b8d0174939694017c569b0498e59b06480cf6be45`；全部原账本保留。唯一wrapper3506636→supervisor3506645→worker3506652，实际内层日志为 `train.resume_018.log`。首PPO只读强验10/10通过：父／初态actor、critic、Adam逐值一致；29997–30006为10次真实critic-only消费，actor3075及两个课程相对时钟781保持；30007的首actor3076九张量及critic／Adam改变。只读独立抓取原生1–11次progress并核对SHA；中间actor张量未逐更新封存，因此仍明确未观察。初次账本读取撞到shared claim与campaign文件写入窗口，原失败样本保留；新只读read003一致，没有重启、热改或退款。11:24:19实读成功更新14／actor更新4、task3079、完成消费30010、actor消费29860；学习仍在继续，新质量窗口未完成，不授予行为资格。

首PPO证据：[first_PPO_verification_021.json](../reports/v6_scut_posture_completion_20261006/source_repair_021/first_PPO_verification_021.json)，SHA `e7af7199d4bd64ef7dd527d2cc56cc83d46dbac9a7e6c0fdc5c0fd5fa56aa30e`。主launch已切到021实际身份；旧020验证和计时独立保留历史，不能套用为新版本质量或提速。下一次质量观察使用现有原生TensorBoard完整actor窗口，分开L0教学、L2原整链和普通基础行为；不另起并行训练或自动放宽标准。

11:28:00（UTC+8）实时只读核验：actor3094，本段29次成功含10critic＋19actor；完成progress、campaign state及消费账本均30025，actor消费29875，原三owner和v19身份不变。独立首PPO审查通过（SHA `3c39d29713e3ac96fb62a3538ec915ccb91582c91bee2515d39a83a9be8f8521`），所有11份原进度字节与哈希一致；仍不新增行为资格。
