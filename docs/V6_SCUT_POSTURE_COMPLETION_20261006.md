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


## 021完整actor窗口的质量观察

2026-10-06 12:00（UTC+8）只读实测原三owner3506636/3506645/3506652仍在recovery attempt_005连续PPO：task actor3210、本段135次actor加10次critic，实际完成与账本均30141，actor消费29991。没有重启、热改或退款；当前最新模型与初始化父状态分开记录。

新版本首个完整稳态观察窗为attempt actor11–110／task3086–3185／TB30016–30115，排除10次critic-only。两个不重叠50窗及其合并100窗均有完整CRC和必需tag，95组的760个整数计数及855个按frame加权均值由独立审查重算；100窗与50窗不能相加。每环境48秒物理数据中9389210个转移有PPO资格，441190个脚本／混合转移被排除。

| 指标 | 前50actor | 后50actor |
|---|---:|---:|
| 230mm静止平面均速 | 11.80cm/s | 12.18cm/s |
| 305mm静止平面均速 | 11.22cm/s | 10.72cm/s |
| 430mm静止平面均速 | 8.61cm/s | 9.23cm/s |
| 前进0.5速度MAE | 26.79cm/s | 27.25cm/s |
| 后退0.5速度MAE | 26.23cm/s | 27.76cm/s |
| 正向yaw角速度MAE | 0.632rad/s | 0.580rad/s |
| 反向yaw角速度MAE | 0.533rad/s | 0.554rad/s |
| 0.5启停驻停均速 | 12.22cm/s | 12.52cm/s |

前／后实际纵向速度为0.23253→0.22501与−0.23453→−0.22085m/s，指令约±0.496，欠速仍明显。0.5驻停条件分别有17554／17494frames，1.0驻停为21050／21968，不能用含运动区间的累计位移代替驻停误差。54个恢复命名组累计7255次已结束回合，success0、failure2323、timeout4932，其中80场地截断另分类；场景混合动态L0/L1/L2，不能称为原自起链最终成功率。新hold成本确实支付，轮门归一误差约2.38基本不变，腿和gyro略降；未观察到整体恢复质量改善。

support/body_clear两门来自FK／IMU观察器决策，不是实际接触。真实接触另参与最终判分；连续1s联合通过不能由各门边际比例相乘推算。静止10s窗口230／305均无完整样本，430共有71个条件窗口，不把未完成记零漂移。所选TB reader没有独立L0 motion-release及非零目标曝光统计；静态零命令不能证明目标桥实现失败。全轴mean_std是六轴聚合值，六轴DiagnosticUpdate/action_std_i既有tag另行读取，不需要新增生产日志。

100次wall均值15.412s／中位14.598s，环境12.592s、optimizer0.117s、诊断0.423s、logger0.281s；两个50窗wall14.675→16.148s。共享GPU且未配对actor条件，这不是新源码整体加速或减速的因果证据。详细结果见 `learning_audit/recovery_window_candidate_021/quality_assessment_021.md` 与 `first100_summary.json`，独立分区review SHA9097fbc1…59e402。当前功能资格仍null，腿bag仍未qualified。

下一步已在独立本机GPU串行执行同一own REC3076不可变seal的均值／采样0PPO原生对照。PT SHA286b9e26…b671c31，消费30007、task3076；这是修订后首个actor模型，不是当前最新模型。原生出生、保存的std／RNG、50Hz／200Hz、PD160/2.5、奖励／命令／接触判据保持，只有应用mu或sample不同。正式Kaiser继续训练，诊断不消费训练预算。两臂完整结果、出生及私有采样流待原生readback，12s未完成属右删失；不得由单臂早期输出预判结果或热改std／Adam。


## 021同封存配对与六轴探索读数

12:11（UTC+8）Kaiser原三owner仍连续PPO：task3239，本段164次actor＋10critic，完成及账本均30170、actor消费30020；source021／v19未改。

独立本机的same-own-REC3076两臂均真实完成600策略步／2400物理步并正常exit0。初始出生／lesson／profile／env RNG、六轴保存std和私有Gaussian采样流首尾一致；仅应用mu或sample不同，后续reset的env RNG允许随行为分叉。原生付款逐lane残差0，独立float64分箱汇总的舍入差另列，不篡改原结果。App.close是否返回仍未记录；process exit0与原生preclose完整性已经根进程实际观察。

| 首出生课级 | 固定人口 | 均值释放／成功 | 采样释放／成功 |
|---|---:|---:|---:|
| L0简单接稳及原目标跟随 | 1883 | 443／216 | 0／0 |
| L1短零力矩接稳及原目标跟随 | 675 | 157／63 | 0／0 |
| L2原完整自起链 | 89 | 12／2 | 0／0 |

恢复固定人口2647、ordinary1449独立计数。mean恢复281成功／276失败／2场地截断／2088右删失；stochastic0成功／459失败／1场地截断／2187右删失。12s未完成不是20s超时失败；281含教学prefix，原完整链成功仅2。mean首出生L0实际原前进／正yaw／反yaw目标曝光77615／2894／1008个50Hz whole-actor区间，stochastic三类均0，后者作为有效完整负结果保留。采样臂首出生sensor连续稳定峰值最多L0 .06s／L1 .05s／L2 .03s，不能用瞬时八门边际比例乘积解释连续通过率。配对证实该seal的采样应用阻断持续接稳及后续曝光，不证明entropy唯一根因，也不代替最新模型或实车资格。

两臂全窗各5次闭链guard，峰gap为mean31.374mm／stochastic730.632mm。首出生可定位4／3次，含两臂逐值相同的两个L2台阶脚本终止；这些首次guard帧whole-actor mask为false，env closure_gap终止不bootstrap。现观察器未保存全窗argmax lane／frame及所有后续出生guard逐事件，最大gap位置不可可靠定位。物理机械资格未过，不隐藏大gap，也不把全部异常归因探索或熵。原3mmguard未放宽；后续须单独核查台阶出生／脚本的闭链动态可行性。

六轴std/KL从既有TensorBoard同100actor完整103tags×100原值读取，独立重算309段统计和物理映射。两半std为：LH .25632→.25709、LK .25920→.25587、RH .27136→.26945、RK .24839→.24729、LW .82817→.83898、RW .85899→.86414。腿3降1升、轮两轴均值上升；每轴均有逐更新波动，不能称全部std单调增大。全窗四腿目标σ约.062–.068rad，轮目标σ8.336／8.616rad/s，轮缘目标σ.500／.517m/s；这些是限幅之前的目标尺度，不是实际运动噪声。global KL均.006916／最大.012181，4/100点超过.01，fixedLR不会因此自动刹车；个别组KL有尖峰，但每组probe只有16观察且未按actor资格筛选，不能当实际梯度贡献。

最小学习补救候选是原生entropy单变量消融，保留std／Adam／原采样和log-prob／所有奖励与1s门；不使用mu收集伪装PPO，也不硬夹噪声。当前运行entropy仍.005。optional合同参数的本机接线与聚焦测试正在实现，尚未生成新worker／冻结部署或开始PPO消融；后续须明确独立RNG、真实消费与判决，不能只改JSON或seed标签。当前只提交证据不触发正式重启。

原始配对与压缩字节保全见 `learning_audit/exploration_gate_review_021/REC3076_pair_001/`；独立复核JSON SHAde41e6e5…87f2b81。六轴实读在 `exploration_gate_review_021/per_axis_std_021_001/report.json`（SHA13c2ce9c…8e03d8），独立review SHA047a87fc…26d1c6c。补救候选 `learning_audit/exploration_gate_review_021/learning_remedy_design_021.md` 明确了PPO概率一致性及待验证边界。

## 021第二个完整窗口与探索系数入口

2026-10-06 12:32（UTC+8）原三owner仍在连续PPO：task3299，本段224次actor加10次critic；实际完成和账本均30230，actor消费30080，恢复域消费3339。新本地结果与代码提交不改变冻结source021、运行合同、保存的std／Adam或自动课程。

又取得不重叠的actor111–210／task3186–3285／TB30116–30215完整100窗，实际95组baseline为30115。首次读数因尚差更新而pending，原样保留；第二次完整CRC读数成功。两次100各48秒物理经验，100窗与其尾50不叠加。与前一个完整100窗比较：

| 指标 | actor11–110 | actor111–210 |
|---|---:|---:|
| 230mm静止平面均速 | 11.99cm/s | 12.41cm/s |
| 305mm静止平面均速 | 10.97cm/s | 11.32cm/s |
| 430mm静止平面均速 | 8.92cm/s | 7.79cm/s |
| 前进0.5速度MAE | 27.02cm/s | 28.95cm/s |
| 后退0.5速度MAE | 27.00cm/s | 29.95cm/s |
| 正向yaw角速度MAE | 0.606rad/s | 0.591rad/s |
| 反向yaw角速度MAE | 0.543rad/s | 0.561rad/s |
| 0.5启停驻停条件均速 | 12.37cm/s | 12.22cm/s |
| 1.0启停驻停条件均速 | 12.21cm/s | 12.27cm/s |
| 每次更新wall均值 | 15.41s | 21.24s |

第二窗前／后实际纵向速度0.2074／−0.1963m/s，欠速加重。54个混合恢复组0／9092个结束回合成功（2439失败、6653超时，其中79场地截断）；不是L2原整链最终率。0.5／1.0驻停条件分别34242／42728frames，只统计驻停段。轮／腿／gyro瞬时门率约16.05%／23.10%／24.87%，连续稳定历史峰值的全体均值.00966s；不能据边际门率推算1s联合通过。整体质量仍未改善，不以reward或个别改善授予能力资格。

12:30只读资源快照确认Kaiser GPU99%且两个compute PID，正式worker仍3506652，另一PID3511069不属于本回执三owner；未停止或调整另一进程。32逻辑CPU的load约2.7，worker约99线程。这只是单次共享主机观察，不能把100窗变慢全部归因新代码、另一进程或多线程竞争。资源原始输出见 `source_repair_021/shared_host_snapshot_021.read001.json`。

可选 `entropy_coef` 接线候选002已完成本地验证：缺字段保留V40原.005，显式0有效，非法值拒绝；原生cfg／实际PPO／恢复／进度／checkpoint infos和seal记录对齐，不改变采样、log-prob、masked GAE、网络或Adam。原生PPO load本身不恢复entropy系数，所以必须显式物化并在恢复后实读。168项相关CPU通过、5个旧fixture在冻结021独立复现并明确排除；独立39新例和6个真实CPU PPO/Adam故障路径核查通过，测试集合重叠，不相加报总数。

独立审查发现的初稿两点已修：真实optimizer返回后先记录successful／actor及课程时钟再做后验系数断言；NaN／Inf失败回执写null、明确invalid状态和原值repr，保留原error。真实已执行或预留消费不退款，失败状态不封存。旧patch、测试与拒绝回执保留；当前代码SHA `c8fbbf3d64a6230ee7ba76c94e0261771a42e350b0e3b27f42fcf002b74cc6bb`，独立review SHA `9eb36bea9717788d1e1d5e8b748080f45edc809023868ee4ba7f5fa616f7e5a0`。

候选仍未部署，正式entropy保持.005。这只解决原生参数入口，不能宣称A/B准备完成：当前campaign全程持有唯一账本锁，strict v19设计固定训练seed，resume会恢复父CPU／CUDA／env RNG；仅改seed标签或开另一个supervisor都不是合法独立研究。下一试验必须先定义严格合同继承、串行收费及真实RNG分叉，不能绕过20k各域／121000总预算、均值伪装PPO或热改运行合同。两臂机械guard的出生／脚本原因仍单独核查，entropy消融不掩盖该缺项，也不放宽原3mm门。

相邻窗比较见 `learning_audit/recovery_window_candidate_021/adjacent100_comparison_021.json`（SHA73260196…f0fe9b）；新完整原报告SHAf10b04cd…fd8e9e，精确压缩保全在同目录 `preserved_adjacent100_021/catalog.json`。入口候选及完整CPU／独立故障回执在 `learning_audit/exploration_gate_review_021/entropy_entry_candidate_021/`。当前仍无新的行为或实车资格，腿侧bag未qualified。

相邻窗口的独立审查重算590项差值、95组人口与结束分账、54恢复池frame加权均值，全部一致；SHA `24aa8cb1e2103ed9b74aff3a225ca57608a07e41b9388d070c8d27f5baca0df7`。表中静止速度是world水平root-link速度，驻停是body-XY范数及请求时间mask；`settled_stop`只表示驻停观察时段，不证明已经低于3cm/s。

机械只读诊断也独立复核：lane1708@57／2216@59两臂首guard记录逐值相同，均SIDE脚本、L2、handover0且actor mask=false；不能归因探索动作。静态出生闭链约1.23e−16m合法；固定root出生→prepare／fold的5%插值保守边界margin为−3.923／−1.981mm。这是几何风险，**不是实际SIDE轨迹、PhysX穿透或必然非法接触**，自起可以存在合法壳体支撑；实际30ms释放后的初始路线、接触／力矩／pin gap先后仍需逐5ms复验。两个profile原本标记dynamic_validated=false；完整双级、自碰撞和动力学不授资格。全窗730.632mm没有argmax lane/tick仍不可定位。诊断和待议修复见 `learning_audit/exploration_gate_review_021/mechanical_first_guard_diagnosis_021.md`；独立18输入及CPU几何复核SHA `d11be38a44d46c45be730f8e73483fda50e258eafa33b1e21cbccb44a6984c4a`。暂排两个profile仅是未实施提议，当前合同覆盖与3mmguard均未改。

严格entropy研究协议已写为 `learning_audit/exploration_gate_review_021/entropy_trial_protocol_021.md`，`launch_allowed=false`。推荐两真实训练RNGfork×A/B×100actor，400actor；未实现严格零critic例外前必须预留440optimizer并全部收费。协议中98–108分钟使用前期14–16s读数，当前第二100窗21.24s意味着400次仅PPO约142分钟，另有启动及离线诊断；两者均不是工期承诺。研究父seal、四arm合同、预算预留和队列仍未物化。正式021保持连续，结果提交不触发重启。

2026-10-06 12:43（UTC+8）只读回执 `source_repair_021/current_std_live_021.read008.json`：原021三owner和源码身份不变，REC actor3331／本段256actor＋10critic，完成与消费账本30262、actor消费30112、REC消费3371。新熵候选仍只在本机、正式β0.005；结果提交不触发部署或重启。下一步隔离复验脚本阶段的台阶闭链终止，捕获逐5ms实际状态和接触，保持3mm门。


## 021第三个完整窗口与脚本冲击捕获

2026-10-06 13:17:58（UTC+8）原source021／v19、tmux r21、recovery attempt_005仍连续PPO。实际REC actor3434，本段359actor＋10critic；完成与账本30365、actor消费30215、REC消费3474，原三owner3506636→3506645→3506652未变。记录结果没有重新启动或改变合同，腿bag仍未qualified。

新actor211–310／task3286–3385／TB30216–30315完整100窗，与前111–210不重叠，CRC、95组人口及累计差分核对通过。230／305／430mm静止平面均速12.38／10.95／8.42cm/s；前后0.5速度MAE28.70／28.56cm/s，实际+.2095／−.2111m/s；正反yaw角速度MAE.6363／.5522rad/s。六个真实驻停组184504条件帧的加权均速11.75cm/s，92.693%高于3cm/s。混合L0/L1/L2恢复组0/7640结束回合成功，不是最终L2整链率；未结束回合保留删失。趋势仍混合，不能称整体质量恢复。wall均20.673s，只作共享机器观测，不作为源码提速证据。完整对照 `learning_audit/recovery_window_candidate_021/third100_comparison_021.md`，JSON SHA7c0ce4bb…234cd，新原报告SHA4717690e…c17e5。

独立本机机械捕获候选003完成4096×600策略步／2400物理步；仍用同own-REC3076完整seal、原出生/RNG/std、50Hz/200Hz、PD160/2.5、solver64/32与3mm门。第一次002尝试因工厂过早导入PhysX，场景启动前退出1；原失败字节保留。003只延后模块解析，30CPU及独立冷factory审查通过；实际native PID185242正常exit0，257.477s、GPU释放。原private采样流、出生与两条首guard57/59签名精确复现，外部冻结collector SHA74d4f473…65d244。capture hooks已在Kit关闭前恢复；App.close是否返回未知，不能由exit0补造。诊断0PPO、0optimizer、0正式消费，不改变Kaiser训练。

| 原lane首出生 | 前一5ms子步 | 首3mm越界子步 | 同原policy终止 |
|---|---|---|---|
|1708 step200_wheels_split|frame226，RR_link1与ground法向合力8505.53N|frame227／1.135s，R_rocker_P 49.484mm|tick57／frame228，31.374mm|
|2216 double_step_first_riser_split|frame234，RR_link1与ground法向合力2883.82N|frame235／1.175s，R_rocker_P 23.048mm|tick59／frame236，3.273mm|

两个越界都发生于SIDE脚本、route4、blend0、actor mask=false。原生接触冲击先于pin越界一个5ms子步，实发电机力矩仍在40/4.5N·m帽内；越界当帧大的preclip请求不能叫实际轴力矩。1708参考连续每5ms+.03rad，2216越界前参考已恒定约30ms，不能共享归因为目标跳变。ground接触不是已证明非法台阶面接触，也不是唯一根因证明；当前证据指向脚本与接触／闭链动力学的配合，下一次只复验SIDE速度6→1.5rad/s，原生速度相关相位时长公式保留，出生／模型／奖励／熵／solver／3mm均保持。此单变量复验仍为待实施，不能保证修好，也未移除profile或授机械/实车资格。

新的全窗5ms峰可定位为1708首出生frame227、pin3=49.484mm；旧mean31.374mm是policy-end最大值，测量频率不同。旧stochastic730.632mm仍未被此mean复验定位。所有pin都使用实际body pose；实际terrain接触法向/friction/点/分离量已保存。自接触pair、真实mass/COM/friction随机draw与原生overflow标志尚未捕获；force×dt的冲量是派生量，不构成完整机械辨识。实际readback与SHA见 `learning_audit/exploration_gate_review_021/mechanical_capture_candidate_021/actual_analysis_001/`；大型原始JSON有逐字可还原gzip保全，原文件未改。

熵研究实现设计补明两个阻塞：trial容量从claim内真实attempt重放派生，不能被旧campaign_state覆盖；研究publish必须隔离普通main review token/dispatch，防止误SIGTERM。原稿保留，新设计SHA33336097…52f31／MD f41448e4…42ff；440消费试验尚未实现或启动，正式β仍.005。先完成机械单变量复验，再独立处理mean与stochastic连续接稳差距，不混改奖励、探索和物理。

实际捕获独立审查已完成：全部4096首出生记录、3719事件、600次原生付款与原mean复现；466真实子步的19body／6pin重建一致，机械资格仍为空。回执SHA `eb4023f17d6ab9496ffadab400795c2ed7a3dcb52ff2b4aa14bf982dc6cc9b73`。第三100窗全部95组算术独立重算通过，SHA `6fdc483fefd6f931b60ebb06b12386a821c5612df98de1cd918f66dd53808380`。最新13:24实读原三owner仍连续PPO，REC actor3452／本段377actor＋10critic，完成与账本30383、actor消费30233、REC消费3492；`source_repair_021/current_std_live_021.read011.json`。单变量SIDE候选开始本地实现，尚未原生执行或正式部署。

## 021侧向脚本降速的实际对照

2026-10-06 13:48（UTC+8）实读：原source021／v19、r21／REC attempt_005继续PPO，三owner仍3506636→3506645→3506652。REC actor3524、本段449actor＋10critic，完成与账本30455、actor消费30305、REC消费3564；回执`source_repair_021/current_std_live_021.read013.json`。下面的本机0PPO试验没有改正式源码、合同、std／Adam、消费或课程接续。

候选只将已构造原生RecoveryFSM实例的SIDE速度6→1.5rad/s，使用原速度相关时长公式、原8s脚本期限、1s稳定门、64/32求解器和3mm终止判据。侧向往返前进段.262→1.047s；局部失败期限.924→2.494s，由原公式产生，不能称四倍放宽期限。四根驱动轴的共同进度和闭链相对参考保持。捕获范围扩为两条首出生完整12s，观测器不增加物理步、不调用第二次FSM更新、不写状态或RNG。

37项聚焦CPU测试及独立同集合复核通过，集合不相加。实际本机PID194491完成600策略／2400物理步，350.166s、exit0，GPU已释放；配置与全部钩子在Kit关闭前恢复。App.close是否返回未知；进程exit0是外部实际观察。同own-REC3076完整seal、初始出生、私有采样流与保存std精确一致；SIDE前180／177帧逐位一致，600次原生付款逐lane残差0。候选manifest SHA2d04ffc4…487c1，根实际review SHA00d5ea58…c3c8d94，独立实际review SHA1370b47e…300352。

| 实际结果 | 原6rad/s | 候选1.5rad/s |
|---|---:|---:|
|1708首出生最大闭链误差|49.484mm|0.762mm，无3mm越界|
|2216首出生最大闭链误差|23.048mm|0.441mm，无3mm越界|
|1708 RR轴速度绝对峰|192.165rad/s|21.890rad/s|
|2216 RR轴速度绝对峰|115.526rad/s|11.541rad/s，峰在SIDE前|
|1708 RR_link1法向patch标量力峰|8505.526N|1395.271N|
|2216 RR_link1法向patch标量力峰|2883.823N|859.721N|
|全4096／2400子步闭链峰|49.484mm|27.620mm|
|L2原完整链首出生成功|2/89|2/89|

两条指定轨迹的机械表现改善，但功能没有完成。1708经历SIDE→ORBIT→THRUST→CAPTURE→PREPARE，到8s仍由脚本控制，触发全局脚本预算failure_code6／recovery_script；它不是SIDE局部超时，也不是RL造成的失败。2216在4.92s进入pure RL，policy247首个完整区间具有actor学习资格；12s仍holding，连续稳定峰80ms、82次打断，未释放后续运动或成功。该条记为右删失未完成，不记失败或成功，也不能描述成没有接管。

全域仍有三个首出生closure guard（881／1877／2121）及四次全出生guard。27.620mm峰已定位lane881／birth1／physical888／pin2 L_spring_upper_mount，script-owned；这条不在本次两lane详细接触捕获范围，不能从峰定位推断接触原因。原stochastic730mm也未由此mean试验定位。后续reset的随机流允许随行为分叉，全域281→283成功数不证明降速普遍因果收益；原始L2分母89与结果2不变。

下一步分别核查1708的CAPTURE→PREPARE目标／支撑丢失、2216的连续稳定打断门，以及其余首出生越界。保持脚本预算、1s门和3mm判据；候选尚未正式部署，不授机械／自起／实车资格。完整实际对照在`learning_audit/exploration_gate_review_021/mechanical_side_speed_candidate_021/analysis_candidate_001/actual_comparison_001/`，comparison SHA9befac4a…4e315a；大型文字回执有可逐字还原的gzip catalog，冻结原文件未改。

熵研究的调度隔离候选也完成24项CPU检查和独立窄源码审查：claim→内存→报告保留trial_state，research发布隔离main评审入口，旧普通路径逐字兼容。生产调度器未应用补丁，正式β仍.005。严格合同、容量重放、真实RNG分叉、四arm串行队列与440收费准入仍未实现，不能据此启动试验。补丁SHA4105945a…be62，独立review SHA45325df9…caba55；详见`entropy_scheduler_isolation_candidate_021/README.md`。

## 021准备阶段与RL接稳的分开归因

1708原生子步只读重建说明：CAPTURE→PREPARE前一输入的倾角18.79°、gyro约6.97rad/s，原转移只检查倾角／条件支撑／高度。随后代理支撑与真实轮接触分离，PREPARE目标最终已跟上却仍倒置；后600子步最大腿跟踪误差仅.0865rad，最终车体真实接触约135.9N、两轮无接触、条件FK高度−.196m。代理support=false不能等同真实无接触；也不能用延長脚本时间解释修好。下一源码候选只在CAPTURE→PREPARE复用现有gyro门，暂未原生验证或部署。实际只读重建1593个PD request逐值一致，height差约1.5e−7m；新分析`mechanical_side_speed_candidate_021/prepare_stall_analysis_001/actual_read001/report.json`，SHAd716667a…f48502b。

2216在mean策略下已进入RL，随后1416个参考输入八门按原float32时钟重建，与82次稳定打断、80ms连续峰及末0逐值一致。非互斥失败帧为轮787／gyro596／腿332／tilt39，valid／support／body_clear／height全部通过；真实两轮terrain接触始终大于42／44N。因此本例主要阻断不是支撑不足，**mean本身也没接稳，探索不是失败的必要条件**。这不推翻采样对L0/L1的对照差异，但不能以熵消融承诺解决全部恢复问题。

根据未限幅纯RL PD请求与实际PD参数反演，75%参考输入的有效轮目标超过.25m/s轮缘稳定门；这指已经过当前链路的PD目标，不是raw Gaussian mean。轮缘实测均.292m/s，gyro范数均.680／P95 1.359rad/s、pitch主导，腿请求峰18.878N·m没有碰40N·m帽。下一观测要直接记录解码目标、延迟后目标、原八门及hold付款，先区分目标波动、延迟与奖励支付，不能立刻叠权重、改PD或硬夹std。独立报告`independent_hold_gate_2216_SIDE_001/hold_gate_analysis_2216_001.json`，SHAe6c92cf6…d5fe13；这是own-REC3076的诊断，不冒称当前最新模型确定性结果。

14:00（UTC+8）原正式训练实读actor3560／本段485actor＋10critic／完成与账本30491／actor消费30341／REC消费3600，原三owner不变。第四完整100窗actor311–410／task3386–3485与前窗不重叠：230／305／430静止12.78／11.33／7.80cm/s，前后0.5 MAE29.37／29.56cm/s，正反yaw角速度MAE.604／.541rad/s。六驻停组190240条件帧加权11.63cm/s、92.601%高于3cm/s；混合恢复0/8683不是L2最终率。Wall均20.335s，质量仍未恢复。读数通过CRC／来源身份和95组人口／累计分账检查；远端只读reader的本地SSH等待55s超时后，原结果已写完并进程退出，直接取回核SHA，没有重启reader或训练，远端退出码未知。详见`recovery_window_candidate_021/fourth100_comparison_021.md`与`fourth100_read001/adoption_receipt.json`。


## 021剩余三个首出生闭链越界的实际时序

新增本机原生测量仍只观察新三lane，控制、SIDE1.5、REC3076、全部出生/RNG与600/2400时钟保持；实际PID205097于14:17正常exit0，285.245s，原hooks预close恢复，无接触容量告警。三条首birth轨迹完整889/417/1005帧，所有4096首出生、3756事件和9条全域峰记录的解压字节及floatbits与SIDE001精确一致，identity／teaching／配置回执一致。

原整体严格readback仍拒绝：付款统计有2780个float-bit差异，包括21个正负零极值及2759个浮点归约尾差，最大绝对差约1.8e−9；对应GPU float64 scatter_add统计不保证归约顺序。尚不能用统计接近冒称逐样本reward字节相同。实际report中的时间与各自真实artifact SHA分别核对，没有改写旧拒绝、容差或机械判据。新mechanical-only只读admission明确strict_whole_reproduction=false／qualification=null；完整分析与原始SHA在`mechanical_remaining_capture_candidate_021/analysis_candidate_001/`。

| 首birth实际lane | 首3mm越界 | 原policy终止 | 前一时序的实际证据 |
|---|---|---|---|
|881／L2 front_down90|phys887，pin2 15.331mm|222／phys888，27.620mm|886 L_link1 terrain法向4109.22N；887 setter前L髋dq−90.30rad/s，preclip250.84Nm，实际帽40、setter38.89Nm；PREPARE，目标连续约4rad/s|
|1877／L2 step200_lip_near_shell|phys411，R_rocker_P 9.903mm|104／phys416，9.036mm|410 RR_link1 terrain法向31.47N；411 RR dq73.48rad/s，preclip−148.87Nm、帽40、setter−32.80Nm；ORBIT，目标约4rad/s|
|2121／L1实际step200_upright_top|phys998，R_upper_E 3.739mm|251／phys1004，3.448mm|前20ms内腿速约80rad/s、gyro46.32rad/s与L/R_link3 terrain接触；数百Nm是preclip，真实40Nm帽生效；PREPARE|

三条均script-owned、learning_mask=false、未RL接管。881／2121的CAPTURE→PREPARE分别发生于phys861／850；1877仍ORBIT。先后关系可核查，不证明接触或目标速度的唯一因果。新摘要SHAf95b7fd5…0fb6、完整analysis SHA15e363a0…847f；独立mechanical admission SHA42dba45e…b34b，原strict拒绝保留。

CAPTURE转移的hook001独立审查发现真实原55观察器已安装update closure及共享RNG分叉边界问题，未启动native，原冻结候选与blocked回执保留。hook002只包装真实原调用链、硬逐位范围截止全域首次介入前，新增19针对性CPU及App-free预检通过，等待独立窄审／原生验证，未改生产。奖励终局假设另核实实际v19成功事件为30、不是函数默认5；仅靠五项基础正峰上限5/s不能证明延期更优，不据缺省值改权重。

14:14（UTC+8）原正式021／v19三owner继续：REC actor3602／本段527actor＋10critic，完成与账本30533／actor消费30383／REC消费3642，回执`source_repair_021/current_std_live_021.read015.json`。以上全部本机诊断0PPO／0正式消费，未部署、未授功能或实车资格，既定自动域接续与各域20k／总121000保持。


## 021终局付款与CAPTURE转移原生复验

14:37（UTC+8）原source021/v19/r21持续PPO，REC actor3671／本段596actor＋10critic，完成账本30602、actor消费30452、REC消费3711、剩余90398；三owner3506636→3506645→3506652不变。回执`source_repair_021/current_std_live_021.read016.json`；本机诊断没有optimizer或正式消费，结果提交不重启。

终局奖励窄审计核对实际worker显式success_event_bonus=30、policy_dt=.02、配置gamma=.99。五个正峰至多5 reward/s，对应理想无限折扣和10；在完成成本、成功状态相同的反事实中额外延迟成功收益下降，不能沿函数默认5推导本合同的拖延漏洞。既有SIDE1.5 mean原生hold首出生实际净付款加权2.96561/s，成功事件实际30、回报对账残差0；这不是当前Kaiser探索付款分布，也不是PPO优势或学习保证。完成成本、失败风险及timeout对critic当前V的原生bootstrap仍需逐lane证据。旧takeover去峰函数当前未激活；不基于函数存在或常量正峰立即重写奖励。审计`learning_audit/exploration_gate_review_021/reward_terminal_incentive_audit_021.md`，JSON SHA641b6197…bb4c4。

CAPTURE→PREPARE hook002独立窄审及新增19CPU／原App-free预检通过后，root唯一启动实际PID211871，600策略／2400物理步，365.826s、exit0、进程消失。只在原转移复用既有gyro<.75rad/s，原update链每物理子步一次，源码、config、1s稳定门、8s脚本与3mm判据保持；hooks在Kit.close前恢复。原完整readback因介入前contact patch槽位记录逐位不一致拒绝exit2；原失败保留，尚不能以原strict整体通过标注单变量成功。

全域首次实际阻止转移在update531，两条选中轨迹首次667／710。实际12s原始终局：1708进入pureRL，脚本约4.88s、稳定峰180ms、140次打断；2216进入pureRL，脚本约4.915s、峰155ms、107次打断；均未motion-release／followup／success、未done，记右删失。全域闭链峰仍27.620mm，不能称完整机械修好。状态／控制与接触联合记录的独立只读分析另列新回执，不覆盖原拒绝。候选未正式部署、qualification仍null。

直接PD／目标延迟／原八门及三项hold付款观察器已通过22CPU、原App-free预检与独立窄源码审查，root在上个native进程消失后串行启动PID214432。它仍用旧ownREC3076 mean＋SIDE1.5，不叠加gyro候选、不重新采样policy或计算奖励；保留rawdecode与延迟后真实目标、PD请求覆盖前副本，5ms输入k−1与20ms付款4n末分开。本轮原生执行及readback尚待完成，不能用CPU通过冒称已观察到新机制。


hook002的独立拒绝诊断已完成：已记录1708/2216在phys0..530的非接触布局字段、state和applied_control逐位一致，完整力／点／法向／分离量／冲量联合record逐floatbit multiset一致；仅1708的6帧／210字段及2216的11帧／215字段跨contact槽位置换。不能独立排序各字段制造等价。2682条前介入事件与7条全域峰记录逐位一致，但没有全4096逐步动态历史；新admission仅descriptive、strictwhole及counterfactual=false，原拒绝SHA3a8da3ef…bb3aa保持。诊断SHA7e67eb3f…e21a3c／新admission940d1f4f…6e3a3e，详见`capture_prepare_candidate_021/native_hook_candidate_002/rejection_diagnosis_001/`。没有重跑native或放宽原物理判据。


## 021接稳目标与真实付款：直接观测完成

PD/payment观察器原生PID214432实际exit0、353.944s，完整600策略／2400物理子步及回调预close恢复；App.close返回未知。唯一root readback通过SHAa0c36774…17419，独立实际时钟/归属核查SHA932fef02…aa285。与SIDE001全部4096首出生、3756事件、9全域峰的解码类型及floatbits相同，出生／private RNG／std保持；没有新optimizer、PPO、额外policy/decode/delay/FSM/reward调用。

2216首birth已pureRL的1416个5ms门输入中，每个输入先取双轮绝对轮缘速度的最大值，再作时间均值：原decode目标.43222m/s、真实延迟后PD目标.43047、fresh实测.29199；raw目标超过.25m/s门为1060/1416，delay为1062/1416。大目标已存在于延迟前，不能唯归因USB／目标delay。最大轴raw→delay轮目标差均值5.6466／最大22.7539rad/s、腿目标差均值.01609rad；gyro norm均值.67969rad/s、腿maxdq均值1.46898rad/s。这里的轮缘目标/速度不是车体平动速度，mean策略结果也不是当前Kaiser探索动作的期望行为。

原八门直接观测联合通过295输入，连续峰16tick=80ms，82次break（分四段12/27/20/23）；轮787、gyro596、腿332、tilt39失败，非互斥，不可相乘作成功概率。354完整actor-owned的20ms hold付款共腿−1.209551、轮−1.552033、gyro−5.681967；script/blend不付该专用hold成本。因此“成本未接入”在本native例中被排除，不能用增税取代目标／响应／时序归因。12s无release／success／done，保留右删失；未授功能、机械或实车资格。

完整只读分析及最小后续提议见`pd_target_payment_candidate_021/analysis_candidate_001/actual_analysis_001/`，report SHA3f41a7af…78f61／independent_actual_review_001 SHA b723ac04…168662。后续目标延迟单变量只作辨识候选，仍维持观测delay、std、奖励、1s/3mm，不把raw请求本身的超线归给delay，也不据本例直接改PD或权重。

14:47（UTC+8）原source021/v19/r21三个owner仍持续PPO，REC actor3703／本段628actor＋10critic、完成账本30634、actor消费30484、REC消费3743、剩余90366；`source_repair_021/current_std_live_021.read017.json`。本轮观测和结果归档没有停止、重启或变更正式合同，现有域自动接续保持。gyro+SIDE的描述性全域L2首出生计数是3/89（SIDE-only 2/89），但原counterfactual admission未通过，不能把单次计数变化认证为功能收益；两条选定轨迹均未完成。


已冻结最小脚本修复提案`learning_audit/exploration_gate_review_021/script_repair_proposal_021/proposal_001.md`（SHA85a2fc0d…164a7）。SIDE拟通过已有recovery_training字段显式设1.5，保留native DEFAULT6；gyro只增加原转移谓词。v19场景选择只有恢复域54组/65%会直接执行该FSM，其他五域所选recovery_pose=0。提案要求新源码及合同身份、运行时实际断言和最小CPU/native验证；从届时最新完整学习seal续跑，任何critic适应也计消费。ORBIT1877、其余闭链峰及RL接稳分别仍待解决，未标ready_deploy，也不据本轮报告重启正式任务。


## 021第五个完整学习窗与源码脚本修复实施

15:02（UTC+8）实读原r21／source021仍三个owner3506636→3506645→3506652、REC actor3747／本段672actor＋10critic，完成和账本30678／actor消费30528／REC消费3787，剩90322；read018与主launch更新。没有信号、热改或退款。

第五完整100窗口411–510／task3486–3585／event30416–30515与第四窗相邻，95行为组、41基础组和条件化真实驻停口径沿用原生CRC事件计数。静止230/305/430的平面速度分别13.12/10.87/7.99cm/s，前后0.5指令MAE29.86/30.33cm/s；正yaw角速度MAE .68661rad/s（不同于平动误差）。187575个真实驻停frames加权速度11.937cm/s、93.251%超过3cm/s。混合恢复0/8018、2497failure/5521timeout，是混课结束分布，不作为L2固定整链验收。wall20.2676s/env17.0136s，共享主机描述数据，不授源码因果提速。整体质量未恢复，低姿态静止和速度误差未随继续PPO收敛。

原只读客户端55s超时exit1保留；远端原读已产出CRC完整report，root只取回及核对六文件SHA、未重跑reader，原remote exit code未知。精确原始字节压缩档`learning_audit/recovery_window_candidate_021/preserved_fifth100_021/`；比较SHA44b31160…23e1b、adoption回执独列，不伪造成功进程返回。

前段源码候选已实现：在旧CAPTURE原条件追加existing gyro<.75、DEFAULTS6不改，新REC设置片段1.5；focusedCPU48项通过，其中29复用和19新增，不与既有测试相加。实际源SHA04ff676a…c1ae6。新的合同及所有启动入口身份接线仍在候选阶段，正式源码/合同未变。后续原生源码版验证、冻结与最新fullseal接续必须实做后才更新部署状态；脚本两项修复并不宣称漂移、RL接稳或剩余ORBIT闭链故障已解决。


## 022源码修复的冻结与原生验证边界

冻结源码723732adae37是正式source021 d0dce618的精确子提交，15个文件仅包含两项脚本变更及v20合同/身份接线。REC train角色SIDE速度1.5rad/s、原CAPTURE条件追加gyro范数小于现有.75rad/s；native缺省6和固定评测SIDE6保持。原π/2回转／out-return／80ms驻留／8s脚本／100ms ready／200ms混合／1s连续稳定／1s后续运动、3mm机械护栏均不变。ORBIT与三条剩余闭链故障不是本修复已解决项。

实际归档含2632文件，旧223份合同逐字节保持、新增4份v20合同；独立完整clean树CPU恢复真实REC3076 actor/critic/17 Adam状态/std通过，实际远端30roles通过。模板21CPU和actual metadata检查只授工程准备，不作为原生或PPO完成。归档SHA7ee249d2…1843，清单17881809…0fbe；metadata与部署回执在`source_repair_022/`。

15:28（UTC+8）root唯一启动本机原生PID230688，以旧ownREC3076 mean执行实际新源码600策略／2400物理步、4096环境；同时捕获1708/2216/881/1877/2121首birth、完整联合contact record、全域argmax及原控制付款链。无SIDE/gyro控制hook、无优化器/PPO、不消费正式预算。联合接触multiset的逐位比较仅在同body/filter/tick内保留关联与重复数，旧ordered严格拒绝保留；全4096只记录全域首次源码敏感事件和argmax，不能伪称逐state动态反事实齐全。实际退出与readback尚待完成，功能/机械/实车资格仍null。

15:27正式source021/r21实际REC actor3826、完成账本30757／actor消费30607／REC3866，原三个owner不变。准备在原生工程核验完成后，合法封存**届时最新**完整学习状态并v19→v20接续，10critic适应照消费、actor时钟不推进，原std/Adam/RNG不重置。旧REC3076只是诊断输入，不是回滚或新的正式学习父模型。其他五域worker／父树／20k域上限／总121000／4096／35D81D／50Hz策略200Hz仿真PD／轮响应与毫秒延迟DR保持，未qualified腿bag不接入。


## 022实际源码原生通过与最新全状态接续

源码原生复验PID230688实际exit0／410.1745s，600策略／2400物理完整、原producer回调预close恢复、0PPO。独立实际审查4d290184…b100核对源码/归档/manifest/进程/报告/identity全SHA，三组完整state/control与关联contact record multiset无容差精确一致：source021全域首次源码敏感事件之前298/447行、与先前SIDE+gyro两选定轨迹全窗4802行。全域首次源码敏感SIDE参考在reference149/inputphys148/lane2567；以后共享reset/RNG可分叉，仍不声称完整4096动态反事实齐全。原ordered严格拒绝保留。实际新runtime四字段04ff／1.5／gyro.75与冻结v20设计一致，无控制override。

1708/2216现在实际在4.885/4.92s进入pureRL，稳定峰.180/.155s，均12s仍holding、未motionrelease/success，保留右删失。1708/2216闭链峰.762/.441mm；881/1877/2121仍firstbirth closure guard、全域峰27.620118mm。新工程准入74ba79cd…0817只允许受原护栏约束的研究训练，不授功能、全域机械或实车资格，不以正常退出替代行为达标。

15:42root只向已认证旧supervisor3506645发送一次SIGTERM；source021 worker实际以REC3871／本段806=796actor+10critic封存，消费30802、actor消费30652、REC3911、stage17239、learning lineage17309，model_final SHA47bc2845…78bd。第一次50s只读等待时worker/supervisor已退出但wrapper还在归档，原snapshot保留未冒称所有owner退出；旧wrapper完成5.09GB最终归档后才prepare，未重复stop。15:45冻结v20 latest-own-seal/全账本/所有五已运行域最新parent真实认证后启动唯一r22，10critic适应计费，旧REC3076仅诊断、不作为正式rollback。其余五域worker/父树和所有预算/频率/DR不变。

15:48实际新source022／v20／REC attempt_006唯一3543205→3543210→3543217，已完成4critic-only／总30806、actor时钟3871和消费30652保持、0新增actor。原生读取同source与唯一invocation，不建立第二训练器；本机不跑TensorBoard。首actor及全state恢复的完整核验仍待适应完成，实际身份查主launch与`source_repair_022/current_live_022.read002.json`。本机main已合入精确15file差异，86定向CPU通过；train既有optional entropy与preflight未评分回放hunk保留，不把本机额外字节冒用远端723732完整inventory。

15:54只读核验已完成：`source_repair_022/first_PPO_verification_022.readonly002.json`实际10项通过、0 pending／failed，SHA3c27939e…d9e7。actor／critic／全部17 Adam状态初态精确继承；std包含在actor中，未重置。首critic封存证明actor/std不变，critic/Adam改变；1..11实际progress副本证明10critic消费30803–30812、策略与课程时钟保持，30813首actor的9个张量（含std）及critic/Adam改变。checkpoint500未保存中间每个critic的tensor，不能声称逐次权重逐位复核；RNG状态存在并经过冻结恢复路径，reset会消费RNG，不声称reset后与父seal逐位相等。

首次inspector原实际结果9通过／1失败完整保留：验证工具误从新的script recipe导入旧quality模块定义的HOLD_GATE_BINDING，runtime核验提前中断。第二只读工具仅修正该一行import，4.270s实际exit0，其余判据保持；冻结源、admission与原失败证明未改，未重启PPO。新工具随后实际核完原生模块、SIDE/CAPTURE四字段、人口、旧课程origin、轮侧响应与腿bag未接入边界。

15:54:45实读r22／attempt_006已完成23更新=10critic＋13actor，REC actor3884、stage17252、学习谱系17332；完成消费与账本30825、actor消费30665、REC消费3934、剩90175。后续读取账本可能含在途预留，要与progress完成数分列。唯一三owner保持，无旁路评测或第二训练器。原生六域自动接续、各域20k消费/总121000、REC actor15000节点与全部频率/DR维持；TensorBoard只运行于Kaiser6006，本机6007不动。

当前完成的是源码部署与实际学习链核验。source022尚无完整100更新行为窗，不能宣布漂移、欠速、yaw或RL接稳已改善；最近完整读数仍是上一源码第五窗。三条881/1877/2121首birth闭链护栏及27.620mm全域峰未解决，两条pureRL轨迹12s未完成；功能/完整机械/实车资格仍为空。继续用原生TensorBoard事件观察常驻41基础行为与恢复链，避免把脚本修复成功等同整链功能成功。

## 022剩余闭链故障的实际时序

独立只读核查已完成：`learning_audit/exploration_gate_review_021/remaining_closure_readonly_022/`保留原输出与新的scope erratum；独立review SHA41bef6f9…f11ed。2383原始机械行及2380份k−1反馈时序核准，setter/engine实际腿40Nm、轮4.5Nm帽保持。三条在故障时均script-owned、非pureRL，不能归到此诊断窗的actor探索。

| 首birth lane | source022当前阶段 | 首个5ms超3mm | 50Hz终止时刻 | 已证实先后关系 |
|---|---|---|---|---|
| 881 | CAPTURE | phys887，15.331mm | phys888，27.620mm | phys886 L_link1地形法向记录合计4109N，随后髋反馈−90.30rad/s；下一输入请求250.84Nm被限到40Nm，再出现闭链超线。 |
| 1877 | ORBIT | phys411，9.903mm | phys416，9.036mm | phys410 RR_link1接触后反馈73.48rad/s；phys411请求−148.87Nm被限到40Nm。首超线后四帧回线内，随后才在policy边界再次超线终止。 |
| 2121 | CAPTURE | phys1003，3.315mm | phys1076，3.760mm | 更早phys993车体接触合计15082N、gyro51.10rad/s／腿速79.51rad/s，当帧gap仅.134mm；50ms后首次超线。不能因超线前20ms无接触就排除更早冲击。 |

contact法向记录合计不是独立净冲量分解，反馈、请求、帽、下一gap的时间关系也不能证明某因素是唯一根因。旧source021的881 phase7／2121旧触线时刻不适用于本source022。原anchors对1877使用ORBIT局部峰（1287N／5.55rad/s），更早FOLD全首birth峰2237N／7.84rad/s已在scope erratum另记；原SHA不改。

下一隔离诊断仅检验native既有`capture_speed_rad_s`从4降至1，保留当前SIDE1.5与gyro.75、ORBIT4、几何目标／耦合比例／PD／8s预算／1s稳定／3mm护栏。需独立记录机械峰、脚本超时、接管与原链成功／右删失；若只推迟失败或造成超时，按负结果保留。该候选尚未正式部署，不据此再重启source022训练。

## source022首个完整100策略更新窗口

16:23（UTC+8）唯一远端CPU reader实际exit0，原生event窗口30812–30911、REC actor3872–3971已读取完整；10次critic适应消费保持，未混入此100 actor窗口。实际报告SHA4b99be7b…0631、独立JSON算术对照SHA0cd0e29b…4601，见`learning_audit/recovery_window_candidate_022/first100_read001/`和`first100_independent_comparison_001/`。95组帧数、原生episode分割、方向bias与真实停车样本权重核准；独立对照不声称另行复算TFRecord端点，此层由原CRC/metric读取器承担。

| 指标 | source021第五100窗口 | source022首100窗口 |
|---|---:|---:|
| 230mm静止平面速度 | 13.12cm/s | 12.49cm/s |
| 305mm静止平面速度 | 10.87cm/s | 10.92cm/s |
| 430mm静止平面速度 | 7.99cm/s | 8.85cm/s |
| 前进0.5速度MAE | 29.86cm/s | 30.40cm/s |
| 后退0.5速度MAE | 30.33cm/s | 31.59cm/s |
| 正向yaw角速度MAE | .6866rad/s | .6368rad/s |
| 反向yaw角速度MAE | .5767rad/s | .6051rad/s |
| 真实驻停条件下速度 | 11.94cm/s | 11.63cm/s |
| 驻停样本超3cm/s比例 | 93.251% | 92.995% |

新前后实际均速+.19357/−.18089m/s；六驻停组179052真实条件样本按数量加权，不能把整段运动位移当驻停漂移。混合恢复ended计数0/6944、failure2076、timeout4868（含63 boundary），三教学级及原链成功日志仍零；它不是L2原始完整链的最终验收率。轮／腿／gyro门边际或归一均值小幅变化不能推出连续1s联合稳定。新旧窗口未配对、场景重置和课程分布不同，原链人口份额7.405%→11.265%，不把全部变化归因于脚本修改。

质量仍未恢复，指标变化混合。当前100窗口墙钟均19.0675s／env15.9551s／optimizer.1291s／diagnostics.4765s／logger.3546s，属共享Kaiser描述，不授配对提速结论。16:25实读原三owner继续PPO，REC actor3979、本段118=10critic+108actor、完成与账本30920／actor消费30760；未因读取结果重启、热改或退账。

CAPTURE4→1本机隔离候选artifact002的17 CPU与实际App-free通过，独立窄审SHA38cee1ce…59a50仅授单次0PPO工程诊断准入。16:22:55 root在本机空GPU启动唯一PID247148，冻结source022、历史REC3076 mean／4096×600策略2400物理；Kaiser正式CAPTURE仍4。实例配置结束前恢复，原8s/1s/力矩帽/3mm/64/32保持；是否减少故障待实际回执。all4096全局首敏感参考不等于all4096全state逐位核验，完整关联state/contact前缀限于五条保存lane。旧CPU数值运算顺序伪拒及跨run identity绝对路径伪拒保留，不改原失败字节；admission路径按各run独立SHA核准后只排除该单一path，其他identity保持逐值相同。

## 022最新曲线、课程风险与即时接管诊断

20:53（UTC+8）只读实查：source022/v20仍在REC attempt_006，唯一3543205→3543210→3543217持续PPO；REC actor4894，本次1023actor＋10critic，stage18262，actor消费31675。实际回执为`source_repair_022/current_live_022.read016.json`，未重启、退款或热改合同。TensorBoard仍在Kaiser6006；本机6007的旧flat页面不是当前REC入口。

最新已完整读取的100 actor窗口801–900对应REC4672–4771。与本源码首100窗相比，三高度静站平面速度12.49/10.92/8.85→12.82/11.95/8.90cm/s，前后0.5速度MAE30.40/31.59→29.60/32.07cm/s，真实驻停均速11.63→11.98cm/s。混合恢复结束回合0/7488，三个教学级的成功日志均为0；它不是独立L2固定整链验收。完整窗wall均值14.9426s是共享主机实际观察，不是配对提速证明。客户端原60s超时exit1保留；原远端完整report随后只读取回，原remote exit未知，没有第二reader。原报告与独立算术比较见`recovery_window_candidate_022/latest900_read001/`及`latest900_independent_comparison_001/`。

课程是按`task_actor_update−2294`推进，在reset选择教学级，不读取掌握率。首100至最新窗L0/L1/L2均值55.60/33.13/11.27%→43.60/37.13/19.27%，零成功时仍在减少容易课、增加完整链。当前actor4894的下一reset概率由源码推算为41/38/21%，不是新增TB测量。54恢复组最新窗vx/yaw记录全部为0：脚本和RL接稳期间运动命令被hold门保持，名字带forward/turn不等于已经获得后续运动曝光。基础41组实际1449/4096常驻，不能称完全没有复习；比例是否足够仍待同条件诊断。20s episode和跨rollout持久稳定计时没有“480ms rollout短于1s门所以无法完成”的硬矛盾。只读审计见`learning_audit/course_readonly_review_022/audit_read001.json`。

驻停与腿/轮/gyro hold成本已启用。最新窗三项精细成本接近饱和，但广域pseudo-Huber仍未裁剪，不能说所有误差已没有支付区分。轮/腿/gyro的边际门通过率14.78/21.25/23.41%，倾角门79.38%，不能相乘成为联合连续稳定率。reward密度不能倒推出pitch/roll角，历史出生后最大倾角也不能当作最新100窗最大值。探索std均值.4638→.4837、课程加难和奖励竞争均是候选机制，非配对曲线不能认证唯一因果。

用户最新明确要求“执行完动作马上RL接管”。新增隔离仿真候选仅研究：PREPARE参考轨迹到终点后，在下一原生5ms控制tick立即100%交给RL，不等待此前的八门、100ms ready或200ms blend。脚本前段、有限值/机械护栏、PD/力矩帽保持；接管后的真实连续1s稳定继续单独记录，本轮不混入“未接稳立即开放运动命令”的第二干预。保存轨迹终点、实际轴误差、倾角/gyro/轮腿速度和前后实发力矩，实际运行结果出来后才评定。候选尚未部署，不把参考到终点说成机械已跟到。

华南虎固定b8ff79f3的实际flat/rough配置使用非标准腿出生、0.2s零力矩后直接恢复policy控制，1.5s有限命令覆盖到时解除；并非我方完整自起脚本和连续1s运动放行门。其奖励包含腾空相位，`flat_orientation_x/y_v`是姿态平方罚而非角速度罚；原“完全无相位奖励”说法撤回。延迟buffer在5ms物理子步推进，不能把配置帧数乘50Hz策略周期。采用这些机制必须区分配置上限、实际更新消费和控制/验收语义。

先前CAPTURE4→1本机诊断已经因Errno28写盘失败中断，PID消失，退出码未知、恢复未获确认；没有有效结果，不重跑或覆盖001，也未作用于Kaiser正式CAPTURE4。实际失败回读为`capture_speed_candidate_022/native_CAPTURE_SPEED_001.interrupted_readback.json`。用户授权的本机清理已完成：六对SHA相同旧测试输出以hardlink去重，路径及字节保留，/home实际释放1138565120字节（约1.06GiB）；原始模型、bag和唯一证据保留。批准的旧pytest目录执行时已不存在，不声称额外回收。回执在`maintenance/test_residue_cleanup_20261006/cleanup_receipt.json`。


## 即时接管001：工程运行完成，完整动作终点语义未通过

21:20:30（UTC+8）本机单次原生仿真实际exit0／412.8373s，600策略／2400物理tick、4096环境、historical ownREC3076 mean，与旧source022基线同封存模型；0 PPO，不消费正式更新。完整hook预close恢复／源码身份通过。实际回执`learning_audit/immediate_handoff_candidate_022/native001.process.json`和独立终审`native001_independent_review.json`，原27文件37197215字节已逐SHA保存在`native001_preserved/`，原失败及001冻结core不改。

候选把PREPARE的每tick动态goal剩余量误当完整自起动作结束。PREPARE既能由SELECT近直立直接进入，也能由PLANT／CAPTURE／SIDE末段进入。1708／2216本次首次真实接管均在phys53（.265s），来自初始UPRIGHT分支，从未执行FOLD／SIDE／ORBIT／THRUST／CAPTURE；原baseline则完成恢复路径后在4.885／4.920s接管。phase9的被动前置占位、release_finished=false以及script_age=.03s不是首次真实接管，不能混淆。

| 首出生案例 | baseline接管后实际倾角 | 001接管后实际倾角 | baseline最大单轴实发力矩跳变 | 001跳变 | 001结果 |
|---|---:|---:|---:|---:|---|
| 1708 | 5.79° | 38.54° | .0946N·m | 50.6378N·m | .30s终止，倾角53.94° |
| 2216 | 9.43° | 38.39° | 2.2849N·m | 51.3193N·m | .30s终止，倾角52.62° |

力矩跳变是相邻帧差，并非单帧超过40N·m力矩帽。两baseline稳定峰仅180／155ms，12s尚未放行；001峰0。姿态来自物理gravity，不是reward反演，诊断不是当前REC5017模型资格。全局首干预ref4以前五lane physical0..3共20行state及contact multiset严格一致；之后共享reset／RNG可分叉，不能用先前source修复的148帧cutoff或宣称全4096轨迹逐位配对。原始链成功3/89不变、失败34→51，放行18→9；总体成功287→255、失败282→432是12s描述，保留右删失。

因此本次只证明“当前参考到点时直接pureRL”的工程执行，不证明用户要求的“完整自起动作做完马上RL接管”。完整动作完成意图明确false，候选未部署、qualification仍null。下一候选须按每birth真实末段转移provenance定义完成事件，不能靠初始SELECT→PREPARE，不混入稳定门或改目标形状。

21:24:59正式训练只读REC actor5017／新增1146actor＋10critic，完成消费31958、actor消费31798、stage18385；原三owner持续，正式source022／v20保持。完整100窗姿态审计仅提供tilt门79.38%、gyro23.41%、轮14.78%、腿21.25%的边际均值。95组保存的倾角峰均为出生以来累计值，无最新100窗pitch／roll均值，不能把历史180°恢复出生说成当前持续倒置，亦不能从奖励反推车体水平角。

## 2026-10-07：实际故障恢复与最新完整曲线

attempt_006于10月7日02:51:26（UTC+8）在原生环境步进报告`Nonfinite physical transition`后失败。实际完成2452次更新，其中10次critic适应、2442次actor；own REC6313、总消费33254、actor消费33094。原日志没有非有限张量名称和lane，不能认定为特定关节、气簧或奖励的故障。最新完整封存是update33000／own REC6059／SHA709e2844…80fe0ff；之后254次未封存actor更新的权重丢失，消费照计。

根代理的第一次同合同恢复attempt_007漏掉原部署外层`source sim60-runtime.sh`，因此libXt加载失败、CPU后备路径出现Warp ProxyArray类型错误，0次PPO。原失败回执保留，不改IsaacLab或安装依赖。只读审计核准原runtime SHA c76cc88f…94dcff7及隔离sysroot/WSL库路径。007封存、确认无原训练owner后，于04:57按原runtime和custom tmux socket修正入口启动attempt_008，同一source022/v20、同一33000封存、同一预算，0新critic预热。

实际首33255检查点：actor/critic/Adam恢复前逐值一致，9个actor张量（含std）、critic及Adam随后确实改变；REC6059→6060、stage19427→19428、学习谱系19507→19508、消费33254→33255分别验证。旧只读核验器误读seal中不存在的`actor_updates_in_block`，KeyError已保存；只修只读检查为seal的stage/actor消费字段，未再次重启。04:59之后核验已完成16次实际PPO／REC6075／消费33270，唯一3644259→3644260→3644267。回执在`failure_audit_20261007/`与主launch；恢复环境为新回合，未凭这些字段宣称环境物理状态或RNG逐值复现。

停机后完整100窗口仍由原生TB reader读取，REC6212–6311、事件33152–33251、100次actor／48s每环境，95组计数和七pool独立算术核准。以下为随机训练窗口，非固定验收；新恢复attempt_008尚未获得新的完整100窗口。

| 指标 | REC4672–4771 | REC6212–6311 |
|---|---:|---:|
| 静站230/305/430平面速度 | 12.82/11.95/8.90cm/s | 13.23/11.90/9.35cm/s |
| 三档高度MAE | 7.75/6.45/6.02mm | 7.72/6.16/5.83mm |
| 前/后0.5m/s实际均速 | +.201/−.176m/s | +.236/−.220m/s |
| 前/后vx MAE | 29.60/32.07cm/s | 26.12/27.60cm/s |
| 正/反yaw角速度MAE | .623/.651rad/s | .549/.598rad/s |
| 真实驻停样本加权速度 | 11.98cm/s | 12.58cm/s |
| 驻停超3cm/s比例 | 92.98% | 93.80% |
| 混合REC成功/失败/超时 | 0/2120/5368 | 0/1968/5636 |
| 单PPO更新墙钟均值 | 14.943s | 15.154s |

速度与yaw有所改善但仍严重欠跟踪，静止及停车未改善，恢复后1s接稳尚未形成。0/7604是54混课结束回合的严格成功记账，不是独立L2固定整链验收，也不据此否定已封存的其他域能力。最新瞬时门边际tilt79.16%、gyro23.03%、轮14.22%、腿20.27%，不能当联合连续稳定率；姿态max仍是出生累计。环境步进均12.416s、optimizer.1117s、diagnostics.4150s、logger.2399s，是共享主机观察，非配对提速证明。

下一隔离即时接管002已经按每birth实际PLANT／SIDE／CAPTURE→末段PREPARE provenance修复初始UPRIGHT误接管。原25CPU验证、额外12关闭故障注入、独立二审及CPU预检通过；002 driver关闭失败不会跳过App.close或误授成功，001与冻结core保持。05:01左右仅本机启动单次原生诊断，0 PPO，使用历史REC3076 mean以隔离端点机制；实际结果待完成，未部署002、未改变正式合同或降低1s稳定/3mm门。不能把CPU准入或进程存在当作自起改善。

## 2026-10-07：即时接管002完成，连续接稳仍未形成

上述05:01第一次002运行随后被外层工具生命周期中断；外层实际143，子进程退出码未知，缺完整末态。原24,851,676字节逐SHA保存在`native002_interrupted001_preserved/`，只认可各流持久写入的下界，不把不同流的末行混成完整2400tick证据。固定driver/core不变，05:07:50在独立本机tmux监控中重试，05:17:15实际子退出0／564.4548秒；29文件61,616,370字节保存在`native002_completed_retry002_preserved/`。独立review SHA a5fee54f…9fe3和verification SHA70e5ef…919c7核准所有产物SHA、12个gzip footer／CRC、600策略／2400物理／2400次原生update及预close hook恢复。缺`process_completion.json`，App.close返回仍unknown；exit0不替代该证明，也不授行为资格。

| 首birth lane | 末段来源与立即接管 | 接管倾角／gyro norm | 六电机相邻最大力矩差 | 12秒内连续稳定峰／结果 |
|---|---|---:|---:|---|
| 1708 | CAPTURE→PREPARE call907→pureRL908，4.540s | 9.6901°／.72506rad/s | 11.5292N·m | 225ms，未放行、右删失 |
| 2216 | CAPTURE→PREPARE824→pureRL825，4.125s | 15.5265°／.72397rad/s | 10.6828N·m | 80ms，未放行、右删失 |
| 881 | CAPTURE→PREPARE1019→pureRL1020，5.100s | 见完整独立review | 见完整独立review | 15ms，未放行、右删失 |
| 1877／2121 | 首birth未接管 | 脚本阶段 | 非actor探索归因 | 2.08／5.42s闭链失败 |

完整流核准：初始SELECT→PREPARE而无末段provenance的案例不请求立即接管；有真实末段且当前生成参考剩余0的案例，下一call直接phase9／blend1，六轴actor raw PD=request，不等待ready或BLEND。这里的“动作结束”仅指末段动态生成参考到点，尚未把最后参考施加满一个物理子步；它不是机械跟踪已收敛的证明。原生连续1s八门、未放行零运动命令均逐5ms/策略步复算，保留原判据。

全4096闭链最大间隙11.4725mm，仍超过3mm；固定初生L2为4成功／47失败／38右删失（共89），baseline为3／34／52。首全局干预call299以前五条选定物理state、torque及完整contact multiset严格一致，之后episode/reset RNG可能分叉，因此全窗4/89对3/89仅描述，不能授改进、机械或实车资格。候选不部署，正式source022/v20保持。

最新MLP课程归因核查见`failure_audit_20261007/source022_five_point_scope_review.json`：36.7%为单个24×4096 rollout，新attempt前16次累计有效actor样本72.17%，不是全训练比例；54组仅在未release的lane保持零运动，release后恢复原目标。实际驻停成本为10cm/s广域、4cm/s精核及低前向指令速度绝对值成本−2；旧`.03`项已被替换，不得用旧合同推断当前饱和。actor35维有IMU/编码器/上一动作，无真实body速度，history1，不能只因漂移宣判MLP容量不足。

05:27实读唯一三owner仍3644259→3644260→3644267。当前可供配对诊断的最新不可变seal是update33255／ownREC6060／stage19428／lineage19508／消费33255／actor消费33095，PT SHA27a74ce4…00afd5；已取回PT/合同/封存并逐SHA核准。六轴saved std为腿.2673/.2839/.2659/.2704、轮.9497/.9365，是raw动作分布尺度，不是实际运动速度。下一同seal mean/stochastic观察只复用原生推理／付款／reset，不装即时接管hook、不PPO、不改变课程时钟；实际结果尚未获得，不能预先归因探索。

## 2026-10-07 恢复后首个实际100窗与当前封存模型对照

实际attempt008完整可观察窗口为actor11–110／ownREC6070–6169；actor1–100没有实际actor0 Behavior基线，原reader exit1与stderr保留。95组和7池独立加权复核通过。静止230/305/430均速12.81/11.63/8.24cm/s，高度MAE6.92/6.07/5.68mm；前后0.5实际+.238/−.222m/s，速度MAE26.06/27.58cm/s；正反yaw角速度MAE.538/.560rad/s；驻停12.39cm/s且93.685%超3cm/s。REC54为0成功／1999失败／4878超时的结束回合，不能当L2独立整链率。

有效actor样本8,679,141／9,830,400=88.2888%，不是此前其他窗口的36.7%。更新wall/env/optimizer均值14.148/11.500/.110秒。旧停前100窗REC6212–6311比当前早复原后窗口晚142actor，L2抽样33.0542→31.8708%，出生/相位不同；静止与yaw均值略低不证明恢复收益或没有遗忘。独立比较在learning_audit/recovery_window_candidate_022/resume002_first100_read001/independent_comparison。

同一实际封存update33255／REC6060模型的均值原生回放已真实exit0：600策略／2400物理、12秒、256.056秒墙钟。六轴保存std为.267/.284/.266/.270/.950/.936，动作均值臂仍完整消费原生采样RNG，未改奖励/控制/课程。原始6文件按SHA逐字节压缩封存并回读验证；有闭链峰13.658mm和4条护栏终止，不授机械/功能或实车资格。12秒未结束按右删失保留。探索臂已在独立本机tmux串行启动，实际PID78293；正式Kaiser PPO未停。完整对照未结束前不认定探索为唯一原因，也不热改entropy/.std/Adam。

任务判定与旧terrain报告的mode口径校正在reports/v6_jump_progress_20261005/task_completion_interpretation_20261007.md：八种terrain都有完成定义，普通climb每级.35s锁存后外层2s不重新检测末平台姿态；原地/行进跳离地事件不是逐请求成功率。此次没有新增训练gate，腿侧bag仍未qualified。

## 2026-10-07 同封存模型探索对照与最小正式修复

update33255／REC6060／PT SHA27a74ce4…00afd5 的两臂都实际完成600策略／2400物理。mean/sample子进程均exit0，墙钟256.056／253.105秒；每臂6份原始输出逐SHA压缩回读，未PPO。初始4096lane的状态、教学类别、actor/std、原生私有采样RNG起末状态均相同；动作导致后续reset和环境RNG分叉，全轨迹不能冒称严格逐位配对。App.close返回未知，preclose完整证据与进程退出分别记录。独立复核在learning_audit/latest_seal_mean_stochastic_022/native_mean_stochastic_pair001_independent_review.json。

| 首出生类别 | 数量 | 均值动作接管／放行／完整成功 | 随机采样接管／放行／完整成功 |
|---|---:|---:|---:|
| L0容易前段 | 846 | 846／99／51 | 846／0／0 |
| L1释放扰动 | 980 | 912／80／29 | 902／0／0 |
| L2原始完整链 | 821 | 684／61／21 | 681／0／0 |

均值全4096首出生101成功／446失败／54边界／3495右删失，采样0／562／30／3504。均值L2成功为20平地＋1个200mm台面upright，不能称已覆盖倒地/楼梯完整链；585未结束不是失败。L2瞬时支撑／gyro／轮速／腿速通过比例均值96.36／76.62／61.48／90.89%，采样53.81／22.80／14.61／21.73%，不能当连续1秒通过率。两臂闭链峰13.658／15.495mm、guard终止4／9，仍不授机械、功能或实车资格。

这些同源观察支持去掉正熵激励这一小修复；它不证明reward唯一病因，均值动作本身也未完成多数L2。v21/source023只为恢复域显式entropy_coef=0，原native Gaussian采样、logprob、std和Adam恢复，不强行把std清零、不改变网络或物理/奖励/脚本/连续时间条件。保留首次合同迁移10次critic适应并全部计费，actor课程时钟在critic期间不推进，因此不宣称正式接续是严格单变量PPO因果实验。其余五worker与227旧合同保持字节，REC终点15000、每域20000与总121000消费上限不改，不退款或重开课程。

已完成隔离215CPU、独立17迁移/账本CPU与main选择合入118CPU；集合可能重叠，不能相加当独立测试总量。旧7fixture失败在冻结022复现并保留，补上下文后原断言通过；另一个旧test硬编码已删除的冻结021目录，在冻结022同样失败，保留负回执，以实际227旧合同独立核验替代该过时范围。source023从022真实冻结e58d66aa5933、archive SHA a89135fe…a1c08ee，新源上传后Kaiser30角色静态检查实际通过。此处记录截至06:14（UTC+8）的工程准入，尚未停旧PPO或实际启动v21；是否部署以source_repair_023及主launch最终回执为准。


## 2026-10-07 source023 冻结修正与真实优雅停训

首次e58d66aa冻结的包装器缺v21 schema注册，实际预停入口拒绝，旧PPO未停；该源/归档/静态通过与负回执均保留。只在原包装器三处schema集合增加v21，重新独立冻结6f07ec298e01（父723732adae37），archive SHA ee2288079e27d8e0f03fe5cd4dbfb675c4b414c1eb1e222559fa84be353c445a。683 runtime项、2642 Git文件与227旧合同核准；新源Kaiser真实wrapper预停认证与30角色CPU通过。原PPO配置/六任务父树/奖励/机械/连续稳定判据保持，REC仅去掉正熵激励，不重置std或Adam。

第一次实际stop工具因same-contract启动回执不存在tmux_exit_code在信号前KeyError，未生成stop请求、旧PPO仍继续。旧模板/51CPU/根admission与真实负输出保留；二版按真实启动schema/source/plan/runtime/wrapper/session和attempt008 invocation核验，62定向CPU及独立74实际shape边界检查通过，不增加功能gate。

06:27:14（UTC+8）只向当时supervisor3644260发送一次SIGTERM，原3644259→3644260→3644267及starttime链再次核准；实际attempt008完成366次actor、0新critic。最新model_final完整封存为REC6425／stage19793／学习谱系19873／消费33620／actor消费33460，PT SHA 8dd37ba76092fa634e174facc9665d65b1df1fd9bbd1dfb42e852c19be9883b0。未回滚到诊断33255，未退还旧254未封存消费。06:28只剩原wrapper真实运行归档；旧delivery仍是attempt007 failed归档，不能据它启动。新v21尚未PPO、qualification仍null；待无原owner且本次delivery/全状态/账本通过后接续，不重复启动。


## 2026-10-07 source023 实际首PPO与连续接续

旧wrapper在06:30（UTC+8）实际结束归档；prepare023认证本次delivery、最新attempt008 final、全9actor/8critic/17Adam及所有消费历史后，06:33:11从冻结6f07ec298e01／v21启动r23，实际worker为attempt009。第一次launch没有启动PPO：root自己创建的r23-progress只读监控被tmux前缀查询误认为r23，拒绝日志保留。只将该监控更名为v6-progress-entropy-023-20261007，再执行同一审查过的启动模板；未热改运行合同或另启训练。

实际首PPO只读检查exit0、10项通过，独立短复核确认同一证据：从消费33620／REC6425完整恢复，初始actor、critic、Adam逐值相同；十critic适应全部消费，actor/std/课程时钟保持；消费33631的首actor检查点九个actor张量（含distribution.std_param）、critic与Adam确实改变，REC6426／stage19794／学习谱系19884／actor消费33461。native合同、agent、resume、progress及seal均确认entropy_coef=0和fixed LR1e-4。中间九critic没有各自全张量封存，采用原生progress1..11与CRC event证明执行及计账，不能夸为每次张量逐值核验。恢复路径保留RNG；新episode reset消耗RNG，不声明reset后逐位相同。父seal未记录旧entropy，按未知处理。

06:41:37实读唯一3662137→3662142→3662149持续PPO，无并行评测：30次完成=10critic+20actor，REC6445／stage19813／学习谱系19893；实际progress与账本均消费33650、actor消费33480、剩87350，REC消费6759。资格仍null。熵系数0只移除鼓励噪声的正激励，原生Gaussian采样和已学std继续存在，不等于均值动作训练或静止已修好。

后续只读首个完整100actor窗口用actor11–110（REC6436–6535，TB33640–33739；actor10基线TB33639），排除十critic与累计计数的无基线首段。重点看静站三高度速度、高度误差、双向速度/yaw角速度、停车、L0/L1/L2接管与真实整链；不以总reward或瞬时门比例宣布成功，不加PPO阻断gate。保留原95组出生以来姿态峰的限制，不能将其当当前pitch/roll均值。现在该窗口尚未齐，不能声称新的效果改善。

实际证据在source_repair_023/first_PPO_verification_023.json、independent_actual_first_PPO_review023/review.json及live006.json。所有源/合同、旧失败和父模型保留；结果文档提交不影响正在运行的训练。

06:46:22只读023固定窗适配器实际exit0，认证原3662149 owner连续，完成50次=10critic+40actor／REC6465。请求actor11–110仍差70次，明确返回pending、未解析未齐event，完整行为判读路径尚待窗口齐后执行。只复用既有CRC/metric，不写远端文件、不启动评测/TB/PPO，不新增判据。适配器及本地语法/前三非重叠边界自检在source_repair_023/read_recovery_window_023.py与.selfcheck001.json；actual pending在first100_window_023.pending001.json。

## 2026-10-07 source023 首完整窗口与连续运行

07:05只读适配器实际完成actor11–110／REC6436–6535／TB33640–33739的固定100窗，排除十critic适配与前十actor累计基线。原始event读141,160,790字节、1,285,337个完整CRC记录，选窗必需tag缺失0；source、metadata及原worker owner保持。独立本地重算95组／7池／9,830,400物理观察帧、结束回合式和按帧加权均值一致；这是采样训练统计，不是确定性功能验收。

| 指标 | 旧REC6070–6169 | 新REC6436–6535 |
|---|---:|---:|
| 静站230／305／430mm平面速度 | 12.81／11.63／8.24cm/s | 11.68／10.10／8.19cm/s |
| 对应高度MAE | 6.92／6.07／5.68mm | 5.78／4.80／5.92mm |
| 前进／后退0.5实际速度 | +.238／−.222m/s | +.230／−.215m/s |
| 前进／后退0.5速度MAE | 26.06／27.58cm/s | 26.64／28.19cm/s |
| 正向／反向rotate_1 yaw角速度MAE | .538／.560rad/s | .497／.515rad/s |
| 真实驻停段加权平面速度 | 12.392cm/s | 10.650cm/s |
| 驻停样本高于3cm/s | 93.685% | 91.839% |
| gyro／轮速／腿速门边际通过率 | 23.84／14.45／21.34% | 32.36／16.74／32.41% |
| 有效actor学习样本比例 | 88.289% | 88.557% |
| 每actor迭代墙钟均值 | 14.1477s | 14.1444s |

低中姿态驻停、yaw误差及部分hold门均值有所改善，欠速没有改善，不能宣告接管学会。新窗54个静态恢复组结束回合为0成功＋1681失败＋5034超时＝6715，L2瞬时整链成功均值也为0；混课分母不能替代L2的独立最终完整链分母。54组实际vx/yaw命令记录均为0，后续行驶曝光仍未形成。边界截断109已含在5034超时中，不重复加进分母；普通41组没有相同完成任务，不能据零成功计数说不会站立/行走。

entropy_coef=0后仍使用原生Gaussian采样，mean_std均值.439421，没有强制清零；10critic适配、366次REC学习年龄差、L2抽样31.871→34.921%及reset/DR差异均保留，因此上述比较不是同难度单变量因果实验。95组tilt峰仍是出生以来累计峰，缺当前pitch/roll均值，不据它或reward反演姿态。旧主对照必须读resume002_first100_read001/retry001/report.json；latest2440_stopped_read001/report.json实际是旧attempt006 REC6212–6311，独立作为次对照，不能混成REC6070–6169。

07:19:37实际live009原三owner3662137→3662142→3662149持续PPO，无并行评测、completion为空；完成189＝10critic＋179actor，REC6604／stage19972／学习谱系20052。progress与account消费33809／actor消费33639完全一致，REC消费6918／总剩87191；这些是活进度，不冒称不可变封存checkpoint。live008的actor账本比progress落后1次，是分开读取的在途快照差，负原始值保留，live009重新读到一致。下一固定窗actor111–210／REC6536–6635／TB33740–33839尚待齐，不滑窗或补缺tag。主launch与原始first100_window_023.read003.json、独立first100_independent_comparison001、live007–009分别承担结果及进度证据。

## 2026-10-07 source023 第二个不重叠完整窗口

第二窗actor111–210／REC6536–6635／TB33740–33839的原生CRC读取实际exit0，必需tag缺失0，仍原3662149 owner。独立保存报告重算95组／7池及9,830,400观察帧一致，与首窗无actor重叠；source、合同、metadata与进程身份不变。

| 指标 | 首窗REC6436–6535 | 第二窗REC6536–6635 |
|---|---:|---:|
| 静站230／305／430mm平面速度 | 11.68／10.10／8.19cm/s | 9.23／8.10／7.02cm/s |
| 对应高度MAE | 5.78／4.80／5.92mm | 5.11／4.15／5.69mm |
| 前进／后退0.5实际速度 | +.230／−.215m/s | +.241／−.226m/s |
| 前进／后退速度MAE | 26.64／28.19cm/s | 25.45／26.90cm/s |
| 正向／反向yaw角速度MAE | .497／.515rad/s | .390／.421rad/s |
| 条件驻停段加权平面速度 | 10.650cm/s | 8.297cm/s |
| 驻停样本高于3cm/s | 91.839% | 87.462% |
| 原生Gaussian mean_std | .439421 | .362737 |
| actor有效学习样本比例 | 88.557% | 86.341% |
| 每actor迭代墙钟均值 | 14.1444s | 14.5865s |

驻停与yaw角速度误差在这两个窗口内改善，但欠速仍明显，不能称接稳或各能力通过。第二窗54个静态恢复混课组实际0成功＋1609失败＋6639超时＝8248结束回合；边界59已含在超时。L2瞬时整链成功均值仍0，实际vx/yaw命令记录仍全0，L0独立放行/goal曝光tag缺失；不能据这些记录单独确定命令桥接的错误，也不能将混课分母替代L2结束分母。L2采样概率34.921→35.754%、人口859.336→893.598与episode/reset/DR改变仍存在，此相邻窗比较仅描述训练趋势，不证明熵单变量因果收益或确定性部署质量。

07:41:15（UTC+8）实际live011核对同一wrapper3662137／supervisor3662142／worker3662149及starttime，PPO未重启、无评测owner、completion为空。完成278＝10critic＋268actor，REC6693／stage20061／学习谱系20141；消费33898／actor消费33728／REC消费7007／总余87102，progress与account一致。当前固定下一窗为actor211–310／REC6636–6735／TB33840–33939，读时尚差42actor，不替换为滑窗。证据在second100_window_023.read001.json、second100_independent_comparison001及live010–011；主launch更新不改变生产进程或合同。

## 2026-10-07 后续地形跑道的可复现缺陷与本地候选

冻结source023的原生completion、episode_outcome与env AST实际CPU复现：goal=1.8m、正边界3.6m、双轮位置与root对齐、恒速1.2／1.5m/s且姿态/高度全程理想时，边界截断早于连续2秒完成，分别只累计1.52／1.24秒。浮点真实保持需101个20ms样本；同tick成功优先和.65m终点控制例也实际核准。该证据解释部分高速地形的结构性不可完成余程，不证明所有零成功来自它，不替代物理策略评测。

本地可选terrain_route_runout候选追加旧x=4m之后至6.2m的真实短支撑段，正边界5.6m、负边界−3.6m及侧界±1.7m保持；原障碍段、旧Surface对象/box尺寸、终点、奖励、命令与2秒判据保持。只选terrain训练54／129个显式terrain_training_row且mode2/3的行，包含easy/medium/full各18，为完整课程统一保留过线后稳定余程；普通body_climb、固定322项评测、其它域和未启用配置不应用。rough延续末patch高度，material延续末段摩擦，纵/横坡按末支撑面继续，未假设所有出口都z=0。

append在contact DR及真实collider创建之前，碰撞和heightquery使用同一组DR Surface；lane mask随PhysX clone order排序。startup记录选组/人口和真实corridor间距。当前绑定资产的刚性轮心保守界<.510m，轮体外接球半径.0612883m，合计<.572m，因此至少.6m支撑缓冲；原.4m不足以覆盖合法姿态，FK反例与四项资产/源码SHA已写入测试。此几何界不授闭链动态或行为资格。

两次重叠CPU测试均113通过，root实际17.12s／delegated17.34s，不相加；候选已集成本地env，生产冻结source023及所有合同未变、REC未停。证据在terrain_runout_audit001和terrain_runout_candidate001/CPU_receipt.json。初版独立env接线审查已完成（actual_cpu_review.json，仅覆盖当时helper/env字节）；未来v22 recipe与五入口注册的本地候选也已实现，实际44 recipe／24入口CPU通过，四份新合同仅在临时测试目录生成。新增启动时碰撞/material/query/contact-view读回尚需本次独立CPU与原生核验，不能用旧113项测试覆盖新读回；随后再冻结terrain/design和新源码，并在封存的课程边界接续。当前v21 supervisor未自动采纳未来v22，边界采纳接线仍待完成；不能把本地CPU通过称为远端已修复、缩短成功时间或增加PPO gate。


## 2026-10-07 source023 第三个完整窗口与 terrain v22 冻结

第三不重叠100actor窗为actor211–310／REC6636–6735／TB33840–33939，原生CRC读数实际exit0、缺必需tag0；三个窗各9,830,400观察帧，95组／7池本地复算一致。表内均为含Gaussian采样与DR的训练统计。

| 指标 | 第二窗 | 第三窗 |
| --- | ---: | ---: |
| 230mm静站平面速度 | 9.23cm/s | 8.09cm/s |
| 305mm静站平面速度 | 8.10cm/s | 6.29cm/s |
| 430mm静站平面速度 | 7.02cm/s | 6.38cm/s |
| 三档高度MAE | 5.11/4.15/5.69mm | 4.36/3.50/5.40mm |
| 前向0.5实际vx | +.241m/s | +.256m/s |
| 反向0.5实际vx | −.226m/s | −.246m/s |
| 前/后0.5速度MAE | 25.45/26.90cm/s | 24.11/25.05cm/s |
| 正/反yaw角速度MAE | .390/.421rad/s | .336/.368rad/s |
| 真实驻停样本加权速度 | 8.297cm/s | 7.257cm/s |
| 驻停超过3cm/s样本 | 87.462% | 82.763% |
| 每actor更新wall均值 | 14.587s | 14.681s |

第三驻停192308实际相位样本、54混合恢复组0成功/6832结束（1509失败、5323超时，58边界包含于超时）；L2瞬时整链成功仍0，L0/L1不能授完整链信用。真实后续指令记录仍全0，L0独立release/非零goal曝光仍无可用tag。轮速门边际通过26.577%、gyro57.044%、腿64.759%，历史连续稳定峰均值21.42ms；这些不是联合连续1s通过率。有效学习8,661,834／9,830,400＝88.113%，Gaussian动作std均值.31005，熵0仍非均值策略训练。驻停/速度/yaw改善只描述这些窗口，不能据此证明因果熵收益、无遗忘或实车资格。

07:54 live012原三owner仍3662137→3662142→3662149，实际已完成329＝10critic+319actor／REC6744／stage20112／lineage20192；已完成消费33949，账户33950含1次在途预留、actor消费33779、REC消费7059、余87050。不是账本不一致，也不是不可变seal。下一完整窗actor311–410／REC6736–6835／TB33940–34039；继续原PPO，无新增评测gate或重启。

terrain v22四份实际新合同已独占创建并冻结至源码5ab8b54ea78f532a847ce71ca11dc7679d02cc01；只给54组/1724环境增加出口4→6.2m支撑与5.6m正界，旧任务判据/奖励/终点/其它域保持。独立新增runtime CPU117通过／2.34s，实际ROOT30role预检通过／3.845s，68recipe/entry组合与其他旧集合有重叠不得相加。真实生产source023含231份合同（父source022为227），本机冻结前233份旧合同全部字节保持；早期227的source023计数是父版本口径，现纠正。

本机原生4096 USD/PhysX首次读回在隔离冻结源启动，session73827、/tmp/terrain-runout-native-v22-001；实际启动读回失败，报collision top未覆盖support interval，96.113s后关闭，未执行env.step或PPO。工具session退出0不能覆盖明确failed报告，原生退出码未独立捕获；负结果及原日志保留在terrain_runout_native001。新source不是当前Kaiser源码，v21 supervisor不会重读本机v22；自然课程边界接续与实际own-terrain全状态恢复仍待实现/验证，不提前中断REC。

证据：source_repair_023/third100_window_023.read001.json、third100_independent_comparison001、live012、terrain_runout_freeze001、terrain_runout_runtime_cpu001、terrain_runout_final_review001和terrain_runout_native001。

## 2026-10-07 原生地形四元数边界修复与连续训练

首次负结果定位到实际API顺序错配：Surface.box给出USD使用的wxyz，而安装IsaacLab 3.0的spawn_cuboid/create_prim/standardize_xform_ops按xyzw解析。原CPU夹具直接写Gf.Quatd(wxyz)，绕过该边界而掩盖问题；Viewer直接使用Gf，公共Surface格式不改。只在env原生Cuboid调用转换为(*quat[1:],quat[0])，并让USD夹具消费当前env实际orientation表达式，保留旧顺序失败的负对照。碰撞覆盖1e-4容差、64/32求解器、奖励、终点与成功判据均未放宽。

实际186项runtime/recipe/entry定向CPU通过，20.55s；三文件小修复已单独冻结b0d2800a338fa11099b355ece48c0baaccbb1b02，235合同与父5ab8字节保持，源码仍仅本地候选。独立安装API的完整CPU枚举16720个Cuboid保留旧序232个横坡失败、转换后误差<9e-16m；首lane只是CPU重建预测，初次遗漏驻停重复地形的范围失败证据也保留。

native002实际exit0、100.300s完整4096配置读回通过：1724选定环境、18370碰撞面、8组真实contact view、99480地面查询；顶面最大误差1.058e-6m、query2.980e-8m、surface/friction4.768e-8，所有1e-4容差保持。实际再执行1策略区间/4物理步、4096转移，最大闭链间隙.610mm、终止/超时/成功均0，35D/81D与reward有限、审计源码SHA全部保持。没有policy或PPO更新；一个零动作区间不验证整回合边界、不授地形完成或实车资格，也未验证实际terrain学习状态恢复。正式REC仍原source023/v21连续训练；修复尚未远端部署，课程边界采纳仍待实现。

08:08:43（UTC+8）live013实读同一wrapper3662137／supervisor3662142／worker3662149连续PPO，无评测owner、completion为空；完成389=10critic+379actor，REC6804／stage20172，总已完成与账户消费34009、actor消费33839、余86991。完整学习谱系由认证恢复父19873+389计算为20262；此前live009–012报告用stage+80遗漏本attempt的10critic，现明确纠正。这是本地报告修正，REC/stage actor时钟和真实账本未改。下一非重叠窗actor311–410尚差31，保持原生产source023/v21，不新增gate或提前停止。

## 2026-10-07 第四个完整窗口与当前 REC 的几何影响核查

第四非重叠100actor窗为actor311–410／REC6736–6835／TB33940–34039；原生CRC读取实际exit0，必需tag缺失0，原owner和冻结源码不变。四窗分别包含9,830,400观察帧，95组／7池，复用已独立审查的算术作本次root重算，不冒称第四窗另有独立审查。报告SHA为42ea53450693bdfbf6caa7b150713b8a26bdee7df022323ce3462d133ddbb619。

| 指标 | 第三窗 | 第四窗 |
| --- | ---: | ---: |
| 230／305／430mm静站平面速度 | 8.09／6.29／6.38cm/s | 7.45／5.55／5.72cm/s |
| 三档高度MAE | 4.36／3.50／5.40mm | 5.07／3.32／5.55mm |
| 前／后0.5实际速度 | +.256／−.246m/s | +.262／−.245m/s |
| 前／后0.5速度MAE | 24.11／25.05cm/s | 23.32／25.05cm/s |
| 正／反yaw角速度MAE | .336／.368rad/s | .326／.330rad/s |
| 条件驻停样本加权速度 | 7.257cm/s | 6.222cm/s |
| 驻停样本超过3cm/s | 82.763% | 76.844% |
| Gaussian动作std均值 | .31005 | .26651 |
| actor有效学习样本比例 | 88.113% | 87.494% |
| 每更新wall均值／环境步进 | 14.681／11.941s | 14.842／12.129s |

驻停192201样本，54个静态恢复混课组实际0成功＋1430失败＋6481超时＝7911结束回合，47边界截断包含于超时；L2瞬时整链成功仍0。L0/L1不能记完整链成绩，endpoint计数差是24帧平均累计值的差，不冒称精确整数L2回合。恢复hold边际轮速通过32.789%、gyro65.825%、腿速77.546%，历史稳定峰均值29.67ms；这不是联合连续1秒通过率。静止和驻停继续改善，230mm高度与后退0.5未同比改善，不能笼统称全能力改善或无遗忘。含探索、DR与变化的课程分布仍不构成熵单变量因果或确定性部署资格。

针对当前REC是否也应提前换四元数修复，独立只读核查覆盖真实v21的95组／4096环境：3116个flat三角网格绕过Cuboid接口，490个step_up＋490个stairs共2940个水平实心Cuboid。旧wxyz被当xyzw后的180°X旋转在代数上只置换完整八顶点及六面，实体集合相同；实际USD由于sin(pi)舍入并非逐位相等，最大顶点残差2.78e−16m。材质、6056个接触路径和147456次地面查询保持，query逐位一致。local +z标签的最大.55m差异是面标签交换，不能当实体台面降低。初次严格bitwise负结果保留，最终实际exit0／3.29s；不授PhysX同轨迹或硬件等价信用。此证据不支持为了该问题提前中断REC，横坡/斜坡修复仍需后续正常边界采纳。

已将验证过的b0d2800冻结源真实暂存Kaiser的source_b0d2800a338f，未启动它：归档SHA d5786ade35a056583380acdd9391409a57ca53caddeefbb0c04792f92ab45546、202615412字节；3569个文件逐一与归档一致，231份旧合同保持、235份总合同、v22计划d3e996…32dd、其它五域及迭代调度保持。暂存不等于正式运行已采纳；旧v21 supervisor不热读v22，正常REC结束后的一次原生接续队列仍待实现和验证，不能开第二个PPO。

08:23:22（UTC+8）live015再次实读原三owner及starttime不变，worker R，无评测owner、completion为空；完成448＝10critic＋438actor，REC6863／stage20231，完整学习谱系19873＋448＝20321，已完成与账户消费34068、actor消费33898、余86932。保持原source023/v21连续训练。下一固定窗actor411–510／REC6836–6935／TB34040–34139，尚差72actor；正常读窗不拦停训练。

证据：source_repair_023/fourth100_window_023.read001.json、fourth100_comparison001、recovery_orientation_scope001、terrain_boundary_staging001和live014–015。


## 2026-10-07 正常课程边界接续工具已部署为等待状态

外部边界observer两文件单独冻结617dbe6b5b98808bd54533dc40a7471e05e70dab；实际root CPU 29通过／exit0／pytest5.31s，总5.798s，同SHA独立只读审查无剩余具体缺陷。作者另一个完整CPU轮29通过／6.13s，总6.616s；两轮不相加为58项。早期缺Python pidfd API和fixture缺stopped状态的负结果均保留；实现支持Linux pidfd syscall，禁止numeric PID信号回退。CPU集成夹具替换绝对生产计划注册表，不冒称生产准入。

实际远端check-only exit0／10.437s：3569冻结源归档字节、v21→v22原生迁移、六域30role原生合同入口预检、原job/receipt/runtime/owner均通过，返回waiting/rec_normal_endpoint_not_complete。只写外部observer状态与锁，不产生signal、migration receipt或PPO。随后仅启动terrain-boundary-observer-024，实读PID3681855／starttime97387444、单一literal argv、staged=true和三份既有speed目录基线；当前只等待原REC真实15000 actor端点。

该工具不是新的训练gate：REC端点/全状态正常完成后，等新speed attempt首个原生完整seal（可仍在原critic预热，必须计数超过认证父seal）；再一次pidfd SIGTERM原supervisor，等其自身封存、归档、全部owner退出、account锁释放。原生迁移保留所有成功/attempted消费，选择最新own-speed完整actor/critic/std/17Adam；same-contract只恢复尚余warmup，不再额外安排10次。后续terrain从真实own v15 actor2186/消费2206恢复至v22，一次10critic适应。计划、成功判据、独立任务模型与所有预算上限不改，不回滚退款、不早停REC、不启动并行PPO。真实边界尚未发生，不能把等待进程或冻结暂存称为v22已经训练。

08:40:22（UTC+8）live017独立读回原三owner及starttime均保持，完成517＝10critic＋507actor、attempted517、REC6932／stage20300／完整谱系19873＋517＝20390，账户34137／actor消费33967／余86863一致，无评测owner、completion为空。observer单线程RSS45208kB、总CPU260ticks（含启动，不是每20s消耗），四份边界signal/launch/migration/receipt均不存在，生产仍source023/v21。更早live016的完成493／attempted494／消费34114含一个在途预留，不能把该预留补成actor学习或退款。下一固定100窗actor411–510尚差3，正常读窗不阻断PPO。

证据：source_repair_023/boundary_continuation_queue001、live016–017与主launch；target源仍b0d2800a338fa11099b355ece48c0baaccbb1b02，仅外部observer代码是617dbe6。

## 完整任务统计的实现口径

恢复任务只给L2 original_chain完整链信用：保留原场景出生（包含倒地与upright场景，并非全部从倒地开始）→原生8s预算脚本（已开始blend可完成200ms）→纯RL连续1s传感器八项稳定，并在首次放行前积累同样≥1s物理稳定证据→后续命令连续正确1s。传感器稳定与物理稳定证据可以重叠，不是额外串行等待两秒。后续vx误差≤.12m/s、yaw角速度误差≤.2rad/s、静止平面速度<.10m/s；平台条件与无失败保持。L0/L1课堂成功另计，阶段人口瞬时full_chain_success不能当已结束回合成功率。成功率应以本口径成功/(成功+失败+超时)计算，边界截断不记成功。

平地原地/行进跳需要真实双轮离地≥1cm、连续air峰≥60ms、相对真实离地COM顶点在目标−1.5/+2cm范围，并满足落点/速度和双轮接触、倾角约9.94°以内/高度误差<1cm连续2s；行进和原地目标范围分别按现有任务设置。奖励的吸收/落地计费时钟不等于成功保持时钟。

普通爬阶和台阶跳逐级真实台面证据连续.35s后锁存completed；双级必须按顺序两级完成。普通爬阶允许合法底盘支撑，但不强制“先底盘支撑再特定收腿”。台阶跳要求请求、真实腾空、正确越沿和双轮落区、驻停平面≤.25m/s及稳定；成功逻辑覆盖generic jump，不能额外声称强制COM顶点区间。两类台阶的通用outer2s读取已完成锁存，没有再次逐帧合取落区/姿态；因此不是额外两秒连续接稳的证据。以上是当前统计语义核对，本次没有热改成功门或增加自动晋级要求。


## 2026-10-07 第五个完整训练窗

actor411–510／REC6836–6935／TB34040–34139实际原生CRC读取exit0／72.362s，报告8fe0ba47036a267fe1795d42aa816833e7f7b25d2850b2be19f61bf8f0ff233e；768MiB是event读取上界，不是并行环境数或训练参数。五个固定窗无重叠，各9,830,400帧、95组／7池、缺必需tag0；root复用已独立审查算术重算，不冒称第五窗新独立审查。

| 指标 | 第四窗 | 第五窗 |
| --- | ---: | ---: |
| 230／305／430mm静站平面速度 | 7.45／5.55／5.72cm/s | 6.43／5.22／4.95cm/s |
| 三档高度MAE | 5.07／3.32／5.55mm | 3.21／2.68／5.41mm |
| 前／后0.5实际速度 | +.262／−.245m/s | +.272／−.260m/s |
| 前／后0.5速度MAE | 23.32／25.05cm/s | 22.53／23.71cm/s |
| 正／反yaw角速度MAE | .326／.330rad/s | .282／.288rad/s |
| 条件驻停速度 | 6.222cm/s | 5.602cm/s |
| 驻停超过3cm/s样本 | 76.844% | 71.975% |
| Gaussian动作std均值 | .26651 | .23191 |
| actor有效学习比例 | 87.494% | 87.243% |
| 每更新wall／环境步进 | 14.842／12.129s | 14.825／12.074s |

54混课0成功／1395失败／5471超时＝6866结束，39边界包含于超时；L2瞬时完整链success仍全0。轮速门边际39.006%、gyro74.613%、腿速86.515%，历史稳定峰均值38.48ms、break均值73.006，不能把边际改善称联合连续1s已过。实际驻停196410样本，6组都曝光；前后0.5命令均值+.4964/−.4966，仍明显欠速。有效学习87.243%和wall14.825s是窗口实测，不将读包CPU时间算作PPO或用reward代理能力。

08:44:29（UTC+8）live018再次认证原三owner3662137→3662142→3662149与starttime保持；完成533＝10critic＋523actor、attempted533，REC6948／stage20316／谱系20406，账户34153／actor消费33983／余86847一致。observer PID3681855保持，RSS45208kB、单线程，08:40→08:44总CPU只增13ticks，status仍waiting，无四份边界action产物，不误报v22已采纳。正式source023/v21继续。下一窗actor511–610／REC6936–7035／TB34140–34239，仍待61actor，不用滑动窗口替代固定窗口。

证据：source_repair_023/fifth100_window_023.read001.json、fifth100_comparison001与live018；本次没有训练重启、合同热改、gate放宽或退款。

## 2026-10-07 第六窗与最新封存模型原生诊断

第六固定100 actor窗511–610／REC6936–7035／TB34140–34239实际只读CPU读取exit0、87.520s，报告SHA `2cf353b6138d1a80c5662d14f54f7627e42ff195570f975ec6cdf123228d7b5a`；第六比较SHA `d743223666968a62d3745e30782de0d6cc9e9781e4a0ced51de636e60ee0643b`。复用原生CRC、95组／7池、9,830,400帧的冻结reader及既有算术；新CPU重算exit0，未重读event或新增训练gate。

| 指标 | 第五窗 | 第六窗 |
| --- | ---: | ---: |
| 230／305／430mm静站平面速度 | 6.43／5.22／4.95cm/s | 5.53／4.43／4.14cm/s |
| 前／后0.5实际速度 | +.272／−.260m/s | +.288／−.267m/s |
| 正／反yaw角速度MAE | .282／.288rad/s | .253／.262rad/s |
| 条件驻停速度 | 5.602cm/s | 4.756cm/s |
| 驻停超过3cm/s样本 | 71.975% | 64.349% |
| 制动区间速度 | .29319m/s | .30179m/s |
| 54混课结束回合 | 0成功／6866 | 0成功／6710 |
| 轮／gyro／腿稳定门边缘比例 | 39.0／74.6／86.5% | 47.0／82.0／92.8% |

静站、驻停和yaw在改善，制动区间速度及最大位移峰没有同步改善，不能称全部停车问题改善。L0／L1／L2均有采样、成功累计均值差均为0；L2瞬时比例仍0。lesson端点差是24帧平均累计计数的差，不是整数回合；54混课分母含教学前缀，不是纯L2整链结束回合分母。历史stable peak全4096环境均值53.1ms也不代表连续1秒完成。

source023同封存mean／stochastic诊断取自真实不可变update34000，REC6795／学习谱系20253；不是读取时live REC6993或后续live REC7055。三文件实际下载与完整actor9／critic8／17Adam／RNG／std验证，768源资产文件认证，60相关CPU及独立静态review通过；root两个独立CPU准入和12 fake-child生命周期测试也通过。driver SHA `78b6645c806bc801ed53db1ff159f49d1e5c6116d5dbe4e2c6576c509527ff48`、admission SHA `d086c0d76097ec20eb89e543c9e620340924f08d9c84f29660a7ff2a1e36c3b0`、lifecycle manifest SHA `32893ae9761b730f65f121520e5b8370357d4ca1eb5cc0db5413726b3653e764`。原iter+1假设首CPU拒绝保留；真实update-hook保存时序经独立核查，380次更新时iter33998与消费34000相符，不改权重或计费。

本机mean实际原生exit0／267.739s，4096×600策略／2400物理／12s、实际env.cfg和源码匹配、std/权重/时钟保持、teaching/payment hooks在preclose恢复；App.close_returned仍null，不以exit0补造Python return证据，行为独立审查尚待完成。mean子进程187083实际退出后，stochastic唯一串行启动monitor190084／child190087，真实PID/starttime/argv回执已保存，结果待完成。0PPO、0正式消费、qualification仍null。900s观察超时只记录attention、继续等同一child，不自动重启或signal；初次出生相同不保证autoreset后全程RNG与轨迹配对。

live020在01:10:41UTC仍为原source023生产三个owner：完成640=10critic+630actor，REC7055／stage20423／完整谱系20513，消费34260／actor消费34090／余86740；外部正常边界observer3681855唯一等待，四action文件皆无，v22仍只暂存。下一固定窗actor611–710／REC7036–7135／TB34240–34339，按live020尚差80；未热改、重启或重复PPO。本节本地结果更新不触发远端部署。

## 2026-10-08 当前能力与保存证据校正

03:17:41（UTC+8）live026实际仍是source023/v21/r23 attempt009原三个owner，完成4303＝10critic＋4293actor，REC10718／stage24086／完整谱系24176；实读账户37923／actor消费37753／余83077，completion=null，读取exit0且PPO_actions=0。observer3681855仍等REC15000，四份边界action文件全部不存在，v22尚未采纳。既有read004的progress为REC10649／完成4234＝10critic＋4224actor／attempted4235，没有同期账户读取；最新账户由live026单独核验。latest100完整行为窗仍是read004，live026不增加行为窗口或资格信用。

read003实际完成actor3001–3100／REC9426–9525／TB36630–36729，exit0／429.624s；保存报告已由comparison003及independent_actual_review001核对单位、分母与算术，独立审查未重新读取远端TFRecord。read004实际完成actor4001–4100／REC10426–10525／TB37630–37729，exit0／104.227s，报告SHA `3fb4624120cc186b759d9e6b1b9cfd0fed3b23b1b07fb3a268cc54e30a6739f9`；root comparison004算术重算exit0，各95组／7池／9,830,400帧、缺必需tag0，未有这一个新窗口的独立审查。read004仅认证实际读取的CRC区间，earlier_records_crc_verified=false，不追认此前整段event。

| 指标 | read003 | read004 |
| --- | ---: | ---: |
| 230／305／430mm静站平面速度 | 3.36／2.95／3.25cm/s | 4.75／3.19／3.08cm/s |
| 三档高度MAE | 2.63／3.92／5.52mm | 2.86／2.22／5.93mm |
| 前／后0.5实际速度 | +.334／−.303m/s | +.372／−.373m/s |
| 正／反yaw角速度MAE | .190／.150rad/s | .131／.111rad/s |
| 条件驻停速度／超过3cm/s比例 | 3.985cm/s／40.714% | 3.887cm/s／47.116% |
| 制动区间速度 | .328m/s | .386m/s |
| 基础组有效10s窗最坏漂移 | .759m | 1.020m |
| 54恢复混课结束成功 | 1120／7941＝14.104% | 799／7648＝10.447% |

现在有恢复后接稳、放行和后续命令的实际成功曝光，前后行驶与基础转向也在改善；230mm静站、10s最坏漂移、驻停超线比例和恢复混课成功比例反弹，不能称全部改善或已学成。54组混合L0／L1／L2；L2端点累计计数是24帧均值差，瞬时full_chain_success也不是结束回合率，当前没有可独立计算的纯L2整数分母。两窗均为带Gaussian探索／DR／重置的训练观察，不授固定验收、机械或实车资格。

同不可变update34000／REC6795的mean与stochastic两臂实际exit0／267.739与251.049s，均600策略／2400物理／12s，0PPO；原已保存的三份独立review记录mean首出生L2成功203／993（其中前进17／480），stochastic三类恢复课首出生放行均0。mean仍有17个闭链guard及405.843mm峰，不授机械通过。这些是当时保存审查的结论；目前旧 `/tmp` 原始输出已缺失，本次不能重新hash或重算这些payload。trace003进程回执实际exit0／308.992s、App.close_returned=null；independent_review004只核验保留的生命周期／baseline比较回执，明确raw归因未完成，不能补造2400帧、17guard或峰值位置的当前独立复核。随后reboot_followup001实际读取uptime -s与boot_id，确认本机于2026-10-08 02:38:28重启且旧/tmp目录目前不存在；时间关系与缺失一致，未证明具体删除机制或执行者，raw事件独立重算仍未完成。

本机清理回执`maintenance/local_test_cleanup_20261007_003.json`实际删除48个cache目录、4,911,104分配字节；三个大型测试候选在该进程清理前已不存在，不声称本次删除5.4GB。模型、bag、文档和raw证据未由该清理进程删除。最新运行与质量引用为source_repair_023/live026.json、latest100_actor3001_3100.read003.json及comparison003、latest100_actor4001_4100.read004.json及comparison004；原生保存审查在latest_seal_mean_stochastic001，trace当前限制在latest_seal_mechanical_trace001/actual_native_independent_review004.json。此次只更新本地结果记录，不触发远端部署。

### 2026-10-08 Desmos核形与高速／台阶／跳跃课程复核（未部署修改）

已在用户原有内置浏览器Desmos实际绘制、验证并保留三条曲线及0–1滑块。横轴收腿误差单位cm：`f(x)=exp(-(x/4)^2)`为现有收腿Gaussian的峰值归一化形状；`g(x)=2-sqrt(1+(x/12)^2)`为12cm广域尺度的说明示例；`r(x)=g(x)+a*(f(x)-1)`展示逐渐增加精调成本。12cm与a并非已辨识、已冻结或已部署参数；图上统一顶点用于比较形状，不能据此给某相位增加恒定正奖励。误差10cm时f仅为峰值0.193%，说明远误差过程项区分很弱，不等于PPO零梯度：PPO根据采样回报／优势更新，还受实际相位曝光和其他奖励影响。截图与数值回执在`source_repair_023/reward_desmos_review001/`。

冻结source023的普通速度、yaw、高度和驻停已经同时支付广域pseudo-Huber与精核expm1；低速指令下的车体平面绝对速度罚−2、pitch／roll姿态及角速度税、动作平滑和制动池也存在。落地真实接触触发3s非续期宽＋精核已经存在于旧jumpv12，计划v18的落地字段与它完全相同。不能把旧跳跃完整成功为0解释成“没有宽落地核”，或将旧precision的.03作为当前有效驻停参数。具体源码／合同审计见`source_repair_023/latest_seal_mechanical_trace001/reward_width_audit004.md`。

| 任务 | 当前可证实设计 | 下一步建议与边界 |
|---|---|---|
| 平地高速 | 静止19%，低速／高速／启停常驻混采；从本域3563起360更新内同时扩到5m/s、3圈/s及0.23–0.43m | 分开平动、旋转、站高及组合的扩域时钟，常驻停车／低速复习；先核对各命令域实际跟踪、姿态和制动支付，再冻结新时长。不是缺静止池或姿态项 |
| 普通台阶 | 地形池有easy／medium／full各12组，几何为1/3、2/3、全尺寸；每组已绑定速度／高度；包括滚上小阶与底盘搭台收腿 | 保留普通前进控制，与触发跳台阶分开；按难度调整各级曝光，核对真实底盘支撑、抬轮缺口、收腿与通过后余程。200mm＋再升250mm、首层踏面800mm的双级为专项，不等价小楼梯 |
| 原地／行进跳 | 收腿4cm固定Gaussian；低到高7个跳高bin概率随本域actor时钟变；落地已有宽精成本 | 先按请求周期拆起跳、达高、落点、接触和连续落稳，再对过程窄核做单项广域伴随／渐进精调对照；不能仅靠mean reward或腾空授完整成功 |
| 助跳窗口 | 概率×力度确实双衰减，累计本域actor到300后两者为0 | 从jump5552接续不会重开助跳；是否需要重新建立定性课，要由失败子条件决定，不能误读enabled=true为当前仍有辅助 |

上述为只读审计与修订建议，当前REC正式worker、网络、合同、采样权重及生命周期均未修改。继续保持4096／35D81D MLP／50Hz策略／200Hz物理反馈PD160/2.5／轮侧响应及毫秒DR，腿侧bag仍未qualified；本次没有新gate或提前切换。

华南虎本次对照使用固定提交b8ff79f3的实际rough_dash交付run，而不是模板默认或可选代码。该配置命令保持5–15s、10%站立，线速度指数项＋平方罚、yaw与姿态角速度项并存；低／高速楼梯分别绑定0.5–1.5／1.5–2.7m/s、0.22–0.34m及小yaw域。该run特殊模式iteration_start均为0，jump/stair状态机与助跳关闭，stand_still为−0且不含stand_still_lin_vel，不能泛化成所有run都激活了迭代门、助跳或−2静止项。实际已有腾空相位专项奖励，也不能再称其奖励“完全无相位”。原始来源：[SCUT实际配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml)。引用这些结构并不保证我们的闭链机械参数、任务尺寸及学习预算可直接照搬。

另核对实际`rough_rotation_stair/2026-07-30_09-16-27/params/env.yaml`：jump／stair状态机均关闭，粗糙地形发生器课程打开，小楼梯单级高度为5–15mm及25–40mm；这不是已证明200＋250mm双级攀爬或腾空跳阶的基准。该run静止速度权重是−1，而非rough_dash没有激活的−2，说明必须按任务run对照，不能用某一模板权重代表所有华南虎模型。[SCUT实际台阶线配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_rotation_stair/2026-07-30_09-16-27/params/env.yaml)。本地SCUT源码HEAD核验为同一b8ff79f3；源码/保存配置描述能力分布，不自行授予其实车成功率。


## 2026-10-08 优先任务收拢：非稳态接管、高速、小楼梯

用户明确本轮先攻破特殊地形、高速和非稳态自起，常规楼梯先采用 5–40 mm。沿用独立单任务模型与原生自动顺序 recovery → speed → terrain；原地跳跃、行进跳跃及 200 mm＋250 mm 双级特化排在这三项之后，不能把其未通过成绩记成已掌握。

本次 v23 只替换 terrain 的普通台阶训练集合，直接承接当前实际运行 v21；v22 仅是此前暂存候选，不制造 v22 学习谱系。四档每阶增量为 5／10／20／40 mm，三阶顶面分别为 h、2h、3h；上／下楼分别采样。四档在 .42 的普通台阶池内固定占 35／30／20／15%，每档上、下各半；踏面继续用本机原有 0.6 m，不冒称复制参考项目的 0.1 m。绑定连续前进 0.3–0.75 m/s、yaw ±0.02 rad/s、目标车高 0.29–0.34 m，32 s 回合；不开普通 STEP／激光请求，不用定时高度触发。5–40 mm 是本轮同时覆盖的难度档，不宣称已实现按成功率动态重分配。

从本轮普通地形采样集合移除 body_climb、大台阶及双级 200＋250 mm；旧合同、固定 322 项评测与大台阶专项定义保留。rough 表面、基础静止与运动复习份额不变；地形训练继续保留真实延伸支撑跑道与余程。高速现有静止／低速／高速混采、加减速约束、姿态奖励及摩擦／毫秒延迟随机化保持；非稳态恢复继续原动作轨迹完成后 RL 接管，并区分接管、连续稳定和后续运动的实际指标。

不增加阻断式能力 gate，不以总 reward 或腾空判任务完成，不改 MLP／35D81D／4096／50 Hz 策略／200 Hz 仿真反馈 PD／轮响应／各域 20k 与总 121000 消费上限；腿 bag 仍未 qualified。当前 REC 原 worker 连续运行，新版本须在本轮 REC 正常端点后通过现有唯一边界接续入口采纳，不能热改运行合同或另开 PPO。代码冻结、原生读回、远端暂存与边界等待的实际结果随后记入部署回执。

### 04:15 实际冻结、验证与边界队列

已冻结源码 `b3c51633b01fadd75c81e5d7015e206a565c13df`，归档 SHA-256 为 `458704d5bc9eda4afd639684b46d275235359c666f5684df244fc3e35ba75790`。239 份合同包含 4 份新 v23 合同；235 份已有合同逐字节保持，其中 231 份属于当前正式 source023。原入口、迁移及生命周期相关 CPU 检查通过，冻结源集合实际 103 项通过；独立调用原生 API 核对 30 角色、旧 terrain 自有 v15→v18 来源及部分 critic 预热续接，未用伪造 v22 模型初始化。

Isaac 原生检查实际 exit 0、150.939 s，101 组／4096 环境，其中 8 小楼梯组 1720 环境；验证几何、材料、支撑高度查询及接触传感路径后运行 1 个零动作策略间隔／4 个物理步，35D／81D、真实反馈 PD 与闭链摘要峰值 0.610353 mm 通过。这仅支持启动与单间隔工程读回，不证明爬阶、整回合稳定、真实学习状态恢复或实车资格；PPO 更新数为 0。完整报告以可逐字节还原的 gzip 保存，原始 SHA 不变。

新源已实际同步 Kaiser，并逐项核验 3691 文件。此前等待的 v22 观察进程 3681855 已单独退休，其状态、manifest 与正式 job 字节保留；未向正式 wrapper／supervisor／worker 发信号。v23 观察进程 3838548 已唯一启动，实际状态为 `waiting / rec_normal_endpoint_not_complete`，四项接续动作产物尚不存在。它等待 REC actor 15000 正常结束，再等待下一 speed attempt 的首个完整原生 seal，通过既有一次边界接续采用新源；不提前结束 REC、不并行启动 PPO、不重置消费或重做已完成的同版本 critic 预热。

04:15:36（UTC+8）最新读回仍为正式 source023／v21／r23／REC attempt009，三 owner 3662137→3662142→3662149 未变。REC actor 10889，本 attempt 完成 4474＝10 critic＋4464 actor；完整学习谱系 24347，与消费时钟分开。账本消费 38095、actor 消费 37924、剩余 82905；其中 1 次更新在途预留，已完成消费为 38094，不能把预留补作已学会或退款。该读回未读取新 TensorBoard 行为窗口，最近完整质量统计仍为 read004，不宣称新小楼梯已经训练或改善。

结果与可复核证据：[部署结果](../reports/v6_scut_posture_completion_20261006/source_repair_025/deployment_result001.json)、[原生独立审查](../reports/v6_scut_posture_completion_20261006/source_repair_025/native001/independent_review.json)、[实际边界等待回执](../reports/v6_scut_posture_completion_20261006/source_repair_025/boundary_continuation_queue002/remote_observer_launch_receipt.json)。当前正式身份仍以 `reports/v6_task_campaign_20261001/deploy/launch.json` 为准，不能把已排队 source025 写成正在运行的正式源。
