# v14 起跑阶段原生指标快照

2026-10-04 21:44:01 UTC（北京时间 10-05 05:44）只读原有 speed attempt_002。worker 3271963，累计本次恢复消费 55、其中 actor 45、task actor 3608。训练仍由冻结源码 `b4ce80964d47` 运行；这个结果提交不触发部署。

原生 event 为唯一曲线来源。每次远端读取最多 16 MiB 尾部，未下载 event、未启动 GPU 进程或评测、未启动 TensorBoard，也未访问 6007。最新读取 CRC 校验 4180 个选中 TFRecord，未定位匹配 0；原生 event 当时 86,163,197 字节，实际只读 16,777,216 字节。

`v14_startup_summary.json` 保存紧凑结果，`v14_startup_baseline.json` 保存累计端点供下一次差分。快照在尾部只覆盖一个 Behavior 端点，因此没有伪造两端小窗：其端点 23917 对应本次消费 50、actor 40，包含此前 10 次 critic-only 更新，下面的行为数值均为本 worker 启动以来累计值。

| 原生行为组 | 实际平面速度 cm/s | 实际高度 mm | vx MAE cm/s | yaw MAE rad/s | 结束 episode | failure | boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| stand_230mm | 13.29 | 230.82 | 9.05 | 0.439 | 73 | 24 | 0 |
| stand_305mm | 12.43 | 302.25 | 8.13 | 0.490 | 59 | 7 | 0 |
| stand_430mm | 7.43 | 424.02 | 6.07 | 0.243 | 61 | 9 | 0 |
| forward_05 | 60.40 | 301.98 | 14.08 | 0.492 | 57 | 8 | 42 |
| backward_05 | 59.32 | 302.95 | 13.59 | 0.500 | 47 | 9 | 19 |
| rotate_1 | 21.72 | 298.62 | 18.68 | 1.749 | 54 | 6 | 0 |
| rotate_1__reverse_train | 17.10 | 302.72 | 13.12 | 0.673 | 61 | 14 | 0 |

累计全部 135 组共 4,915,200 policy-rate lane-frame、6971 个结束 episode、1300 failure、3563 boundary truncation；boundary 与 failure 按原生定义分列。原生 tilt 最大值是 worker 出生以来的最大倾角，不能称作这个小窗的 pitch/roll 最大值。

逐轮指标覆盖 event step 23912–23922，共 11 次更新（恢复后 actor 35–45，按 24×20ms 为每环境 5.28 秒仿真数据），所列字段均有 11/11 点。这个尾部的 230/305/430 静站平面速度均值分别为 12.80/11.73/7.26 cm/s。正式训练 delay strength=1.0，delay enabled fraction 均值 0.72265，transport enabled fraction=0.50244，探索 std 均值 0.44312。

release_stop 的 translate/yaw/combined 及三个 height_train 对应组，**完整 10 秒 quiet 窗均为 0**；完整窗速度、终点位移、峰位移均为 JSON null，不能据此评价制动完成质量。窗口资格需要真实零 visible command 后 2 秒预热、每步高度误差不超过 10mm，位移从有效停车窗口入口锚定。已有 post-release stationary_window 的 policy-rate lane-frame 统计不是完整窗口数量。三个名义高度组在此阶段的 pitch/roll 绝对值均值分别为 translate 4.57°/1.22°、yaw 4.42°/1.17°、combined 5.15°/1.20°，均采用实际相位样本数加权；其他松杆相位见摘要。分项姿态日志只覆盖这些松杆组，不能声称已经记录全部普通组的独立 pitch/roll。

11 次更新的 wall/env/optimizer/diagnostics/logger 中位耗时分别为 14.688/11.458/0.102/0.560/0.189 秒。带探索、重置、全强度延迟的短训练窗口仅表明 actor 已起跑；`qualification=false`、技能资格仍 null，没有新改善、收敛或实车通过结论。固定评测的 signal_delay=false，训练已启用延迟也不等于尾部鲁棒性已验收。

本地检查只编译读取器及远端纯 CPU 程序语法，并核对字节上限、CRC、连续 Timing 点及完整停车窗的 null 语义；没有新增采样或评测。后续在需要的 100/200 actor 时刻执行一次读取并保留新文件，再与这个累计基线按 frames 差分。不要为获得两个端点扩大 event 读取量或等待凑窗。

```bash
python -B reports/v6_locomotion_quality_repair_20261005/learning_audit/read_native_tail.py --tail-mib 16
```

默认输出带 UTC 时间戳，保留已有累计基线。`--window-updates 100` 只会在同一个有界尾部恰好存在两个原生端点时形成差分窗口；未覆盖时明确返回不足，继续保存最新累计端点和逐轮覆盖。

## 使用保存的累计基线，保持有界读取

读取器现在支持 `--baseline`，接受这里的独立 `v14_startup_baseline.json` 或后续 snapshot 的 `latest_cumulative_behavior`。验证同一个原生 event 路径和冻结 source、worker/actor/task 时钟递增、全部原生计数分母完整及无回退，再按累计均值×frames 相减。累计 tilt 等最大值不会被当作区间最大值。

2026-10-04 21:54 UTC 一次新的 16 MiB 尾部读取保存为 `v14_actor40_interval_snapshot.json` 及 `.summary.json`：CRC 4180 条、未定位匹配 0。最新 progress 是恢复后 98 次消费、88 actor、task 3651，但最近完整 Behavior 端点为恢复后 90 次消费／80 actor。因此区间是 **actor 41–80／task 3604–3643，共 40 次纯 actor 更新**，19.2 秒／环境，不把 progress 的后 8 次更新混入累计行为区间。旧 baseline 的 40 actor 累计包含 10 critic 更新，这些旧累计量相减后退出新区间。

| 区间行为组 | 平面速度 cm/s | 实际高度 mm | vx MAE cm/s | yaw MAE rad/s | 结束 episode | failure | boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| stand_230mm | 13.16 | 231.29 | 8.73 | 0.436 | 56 | 7 | 0 |
| stand_305mm | 12.22 | 303.12 | 7.84 | 0.484 | 56 | 4 | 0 |
| stand_430mm | 7.35 | 424.32 | 5.95 | 0.270 | 52 | 1 | 0 |
| forward_05 | 55.31 | 303.39 | 8.99 | 0.472 | 41 | 4 | 16 |
| backward_05 | 56.25 | 304.35 | 10.75 | 0.483 | 46 | 4 | 21 |
| rotate_1 | 18.70 | 300.05 | 15.12 | 1.486 | 52 | 4 | 0 |
| rotate_1__reverse_train | 15.75 | 302.76 | 11.56 | 0.674 | 52 | 4 | 0 |

135 组新区间观察 3,932,160 lane-frame、6227 个结束 episode、693 failure（11.129%）、2989 boundary truncation。这里的均值和计数可由两个累计端点恢复；完整驻停窗口及相位姿态不能从累计 Behavior 恢复。整个 40 更新区间缺逐轮数据，因此完整停车窗数量／速度／位移为 **null**；同一快照单列实际完整覆盖的 actor 78–88 尾部（11 点），六个 release 组在这个尾部的完整 10 秒窗数量是 **0**，速度／位移仍 null。二者覆盖范围不同，不能把尾部 0 扩展成整个区间 0。

探索 std 与 actor 41–80 区间只重叠 3/40 个原生点，观测均值 0.44198；缺少的点未填补。最新完整 11 点尾部的 wall/env/optimizer 中位耗时为 14.808/11.386/0.111 秒。资格保持 false，技能资格为 null；起跑训练差分仍含探索、重置和全延迟，不宣称质量改善或固定评测通过。

```bash
python -B reports/v6_locomotion_quality_repair_20261005/learning_audit/read_native_tail.py \
  --baseline reports/v6_locomotion_quality_repair_20261005/learning_audit/v14_startup_baseline.json \
  --tail-mib 16
python -B reports/v6_locomotion_quality_repair_20261005/learning_audit/test_read_native_tail.py
```

五项离线检查覆盖有效差分、跨 attempt/source、计数回退、无新增 frame 分母和时钟不递增；全部通过。没有追加远端读取、训练或评测。

## 第二个完整40更新窗口

2026-10-05 06:04（UTC+8），一次16 MiB原生尾部读取，以已保存的80 actor累计端点为基线，恢复 **actor81–120 / task3644–3683** 的完整40次行为差分。135组计数分母齐全，3,932,160 lane-frame；CRC4180条、未定位匹配0。同一event、冻结源码和固定人口，两个比较区间均为纯actor更新，各19.2秒／环境。运行progress已到126 actor，不把后6次混入累计端点。

| 指标 | actor41–80 | actor81–120 | 局部变化 |
|---|---:|---:|---|
| 230mm静站平面速度 cm/s | 13.16 | 12.93 | 小幅下降 |
| 305mm静站平面速度 cm/s | 12.22 | 12.06 | 小幅下降 |
| 430mm静站平面速度 cm/s | 7.35 | 7.99 | 变差 |
| 前进0.5 vx MAE cm/s | 8.99 | 8.08 | 改善 |
| 后退0.5 vx MAE cm/s | 10.75 | 9.73 | 改善 |
| 正向旋转yaw角速度MAE rad/s | 1.486 | 1.322 | 改善 |
| 反向旋转yaw角速度MAE rad/s | 0.674 | 0.683 | 略差 |
| failure /已结束episode | 693/6227 (11.13%) | 691/5848 (11.82%) | 不同结束分母，非整链成功率 |
| boundary truncation | 2989 | 2732 | 与failure分列 |

这不是全面改善或固定评测结论：训练包含探索、重置及全强度延迟，移动命令难度也在随全局actor时钟增长。不得将这些静站速度解释为连续10秒停车窗速度，更不能把普通运动累计位移当停车漂移。探索std只有5/40区间点，均值0.44289，未补全缺点；当前完整actor116–126尾部11点std均值0.44300。

40更新区间完整quiet窗指标仍因逐轮数据覆盖不足为null；实际完整覆盖的11更新尾部，六个release组的10秒窗数量均为0，速度／位移仍null。尚不能判定松杆驻停达标。最新尾部wall/env/optimizer中位数14.658/11.429/0.110秒，logger0.223秒，未归属wall2.431秒；这些中位数不能直接相加作总耗时分解。

06:06:35独立只读核验：仍是唯一wrapper→supervisor→worker进程链，4096，新增145次消费（10critic＋135actor），task actor3698，总消费24013。不可变update24000已封存，模型SHA `430c1c395e93c886899dfeffd46b7d19a958f6f743198d30eb15ba9a9fb3c3e0` 与contract SHA全部核对，task actor3685；已下载该封存版本供本机同条件诊断，运行中的PPO继续向前。自动终点speed5063→terrain2186不变，不授予技能资格。

第二窗口证据为 `v14_actor80_interval_snapshot.json` 及 `.summary.json`；该snapshot可作后续差分基线。下载检查点导出的35个ABI向量数值误差最大1.20e-6；这仅验证导出，不证明行为改善或实车可用。


## 第三个行为区间与早期固定回放

06:23（UTC+8）只读快照 `v14_actor120_interval_snapshot.json` 从120 actor端点恢复 **actor121–190，共70次更新**，不是40次窗口；135组6,881,280 lane-frame、9748个结束episode，1236 failure（12.68%）、5349 boundary。移动难度在这段继续增加，窗口时长33.6秒／环境也与前两个19.2秒窗不同。

| 行为指标 | actor81–120（40次） | actor121–190（70次） |
|---|---:|---:|
| 230mm静站平面速度 cm/s | 12.93 | 13.32 |
| 305mm静站平面速度 cm/s | 12.06 | 11.56 |
| 430mm静站平面速度 cm/s | 7.99 | 7.95 |
| 前进0.5 vx MAE cm/s | 8.08 | 9.24 |
| 后退0.5 vx MAE cm/s | 9.73 | 9.20 |
| 正向旋转yaw角速度MAE rad/s | 1.322 | 1.538 |
| 反向旋转yaw角速度MAE rad/s | 0.683 | 0.706 |

仍是混合趋势，没有持续全项改善。探索std只覆盖该70次区间的2个点；当前完整11更新尾部为0.4481，不将它填入缺失区间。六个release组在该11更新尾部完整quiet10s窗均为0，整个70更新区间的完整窗指标依然null；不据此计算全区间停车完成率。

本机独立同条件回放的封存模型为update24000／新增122 actor；结果见 [native_003](../probe/native_003/README.md)。固定均值策略、名义延迟、305mm与带探索/全延迟训练数值的条件不同。其低速跟踪和漂移较旧speed改善，但仍弱于旧flat；出现零速signed yaw −0.0879rad/s，不能把总体reward上涨认作稳定性恢复。新模型±3m/s侧向boundary截断，未将其记为fall或合格停车窗。

06:33:29独立回执确认同一唯一进程链仍连续PPO：本轮244次消费＝10critic＋234actor，task3797／总消费24112，speed消费3827；其余五域消费不变。原生tmux自动终点speed5063→terrain2186及后续课程保持；只更新结果，不重启或热改合同。模型update24000与合同SHA再核验一致。最新完整进度见 `../deploy/current_live_014.json`；该进度晚于行为区间及固定回放模型。

## 06:41与06:52的新行为区间

06:41原生端点差分覆盖actor191–260（70次、33.6秒／环境），可与actor121–190等长比较。06:52覆盖actor261–300（40次、19.2秒／环境），不得把短窗当同等长度独立复现。所有端点均为同event/source、135组完整计数；只读16MiB、CRC有效，无GPU评测或重启。

| 指标 | actor121–190 | actor191–260 | actor261–300 |
|---|---:|---:|---:|
| 230mm静止平面速度 cm/s | 13.32 | 13.20 | 13.67 |
| 305mm静止平面速度 cm/s | 11.56 | 12.38 | 12.55 |
| 430mm静止平面速度 cm/s | 7.95 | 8.42 | 7.75 |
| 前进0.5 vx MAE cm/s | 9.24 | 9.84 | 10.65 |
| 后退0.5 vx MAE cm/s | 9.20 | 9.77 | 9.68 |
| 正向yaw角速度MAE rad/s | 1.538 | 1.295 | 1.118 |
| 反向yaw角速度MAE rad/s | 0.706 | 0.691 | 0.618 |
| failure /已结束episode | 1236/9748 (12.68%) | 1574/11460 (13.73%) | 967/6442 (15.01%) |
| boundary truncation（单列） | 5349 | 5130 | 2986 |

旋转误差局部改善；低/中姿态静止没有持续改善，前进精度变差。不能称作全面收敛。探索std尾部约0.448→0.453→0.460，移动难度也继续上升；静止命令虽固定，探索和其他技能优化仍改变。固定均值策略回放只测新增122actor封存模型，不是本表最新300actor。

06:52完整11更新尾部耗时中位数：wall16.579s、env13.427s、optimizer0.118s、logger0.229s；全70/40区间缺全部逐轮Timing，不能假造全区间均值。六个release组在完整11点尾部仍没有完成qualified quiet10s窗；整个区间数量/速度/位移为null，不能把尾部0扩展为全区间0。qualification维持false。

06:43只读GPU进程清单发现同卡还有另一项transformer训练；见 `../performance/co_resident_gpu_20261005T0643.json`。GPU总利用率不等于轮腿任务独占利用率，未测单进程GPU占比，不能定量归因新增reward或logger导致多少减速；未停止或修改其他任务。

06:52:03全量live核验新增309次消费（10critic＋299actor）、task3862、总消费24177；稍后原生event读到301actor。进度、行为端点和固定回放各自保留时钟，不混写成同一个模型。最新回执继续由原生tmux驱动自动课程，结果提交不触发部署。

## 07:11：达到全移动难度，仍未证明质量恢复

一次16MiB只读尾部从300actor累计端点恢复actor301–360，共60次纯actor更新、28.8秒／环境、135组5,898,240 lane-frame。8655结束episode中1341 failure（15.49%）；4342 boundary与failure分列。最新progress为367actor，immutable仍是新增122actor的update24000。

| 行为组 | 实际平面速度 cm/s | vx MAE cm/s | yaw角速度MAE rad/s | 高度MAE mm |
|---|---:|---:|---:|---:|
| stand230 | 13.19 | 8.61 | 0.448 | 6.17 |
| stand305 | 12.28 | 7.97 | 0.480 | 6.30 |
| stand430 | 7.63 | 6.40 | 0.294 | 5.59 |
| forward0.5 | 56.65 | 10.82 | 0.478 | 6.87 |
| backward0.5 | 52.13 | 9.35 | 0.488 | 6.63 |
| rotate1 | 17.55 | 13.95 | 1.287 | 7.04 |
| reverse rotate1 | 13.69 | 9.19 | 0.591 | 6.73 |

静止没有持续收敛，前进MAE较早期40次窗仍差，反向旋转局部改善，正向旋转波动。此表包含探索、重置与全强度延迟，不是部署均值策略测试；探索std在完整11次更新尾部约0.467。难度在新360actor达到5m/s指令、3rps旋转指令与0.23–0.43m运动高度的配置末档，**指令域已解锁不等于这些能力通过**。耗时尾部中位wall16.599s、env13.023s；不停止其他同卡训练以假造提速。

源码/CPU审计确认零yaw已有未门控稠密成本，不能把偏转归为项没接；普通俯仰目标实际是0，不能声称已有加速度补偿。另见 `../yaw_audit/README.md`、`../posture_reference_audit/README.md`。完整10秒窗口计数语义见 `stationary_window_qualification/README.md`，没有新增完整窗不意味着没有停车样本或奖励停付。本次结果提交不改PPO/合同/自动终点。

## 07:25：最新完整50次行为差分

保持同一冻结源码、原生event和唯一worker。07:25:25全量live确认本轮418actor/428消费、task3981、总消费24296；随后有界16MiB读取progress到420actor，最近完整Behavior累计端点为410actor。以下是**actor361–410（50次、24秒／环境）**，与前一60次窗口长度不同，均包含探索、重置和全强度延迟，不能当固定均值策略验收。

| 指标 | actor301–360（60次） | actor361–410（50次） |
|---|---:|---:|
| 230mm静止平面速度 cm/s | 13.194 | 13.544 |
| 305mm静止平面速度 cm/s | 12.276 | 13.375 |
| 430mm静止平面速度 cm/s | 7.629 | 7.650 |
| 前进0.5 vx MAE cm/s | 10.817 | 10.716 |
| 后退0.5 vx MAE cm/s | 9.347 | 9.471 |
| 正向yaw角速度MAE rad/s | 1.287 | 1.419 |
| 反向yaw角速度MAE rad/s | 0.591 | 0.617 |

135组4,915,200 lane-frame，8758结束episode中1239 failure（14.15%），3688 boundary单列。低/中姿态静止仍未持续收敛，正向旋转反弹、反向旋转较早期改善但本窗略反弹。前进MAE基本持平；不能把结束分母下的failure下降或mean reward作为全域通过。

最新原生10更新尾部（actor410–419）有完整逐轮覆盖，wall中位16.554s、env13.253s、optimizer0.111s、logger0.301s；探索std均值0.47424。完整50次区间缺全部逐轮数据，整体quiet10s窗统计仍为null。**尾部实际新增两个完整10秒窗**：translate高度变体一窗，平均平面速度4.652cm/s、峰位移5.518cm；yaw高度变体一窗，平均速度5.833cm/s、峰位移44.987cm。计数跨PPO更新持续累积，10更新尾部只有4.8秒／环境也能观测此前开始的窗口完成。各仅一个窗口，不能作为稳定通过率，更不能与名义height混用。其余四个release组本尾部零新完成窗。

快照 `v14_native_snapshot_20261004T232559Z.json` 和紧凑 `.summary.json` 保留原生累计端点、CRC与覆盖回执；最新运行进度独立见 `../deploy/current_live_014.json`。immutable仍update24000（仅本轮122actor），正在运行的420actor没有被冒充成已封存固定回放版本。speed5063→terrain2186自动终点不变，本次仅结果提交，无训练源码／合同热改、无重启、无新资格。
