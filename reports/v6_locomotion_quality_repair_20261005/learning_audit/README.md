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
