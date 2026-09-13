# Round3-A Kaiser 并行容量评估

评估时间：**2026-09-14 03:25–03:30 BJT**。本次采用只读 SSH、进程/系统资源采样、实际运行日志和冻结源码检查。

## 结论

1. **保留当前 1024 env / 10000 updates 正式 run，继续完成训练。** 收尾检查 PID `446292` 为 `Rsl`，原 tmux 存在，日志已推进到 iteration `1141/10000`（零起始编号）。
2. **本次不启动 2048 env bench：并跑的 RAM 安全余量未得到证明。** WSL 总 RAM 15.19 GiB，正式训练 RSS 4.86 GiB；实测可用 RAM 在 7.11–9.86 GiB 间变化。新增 2048 进程的启动峰值未知，且已观测到其他短时资源占用。VRAM 充裕不能替代 RAM 容量判断。
3. **下一阶段优先验证单进程 2048 env。** 现有 256→1024 数据表明增加 env 可以摊薄每步固定开销，但冻结实现的 2048 建场估计约 20 分钟。宜在当前正式训练及已安排的后评估释放资源后，协调 native GUI 的资源使用，再做有完整启动预算的独立工程测试。
4. **多独立实验是后续提高实验覆盖面的选择，不作为本轮加速手段。** 两个训练进程复制 Kit、USD、PhysX、模型/优化器等固定开销；当前不追加第二个正式训练。先确定单进程规模，再比较多实验的总 transitions/s、每实验完成时间和峰值 RAM。

本次没有启动 Kit/训练/GUI，没有给正式进程发送信号或替换运行文件。没有创建 bench tmux 或 bench resource marker；本评估未占用新的 GPU 资源，也不代表 native GUI 已获容量验证。未来实际启动 bench 前，应先写独立 resource marker，与其他 agent 协调。

## 运行身份与证据位置

远端根：`/home/kaiser/robot-rl-sim60`。

| 项目 | 核验值 |
|---|---|
| 正式 PID / tmux | `446292` / `r3a-train-35f06fa3504049f8924f9ccd1ed02b93` |
| code | `isaac_wheeled_rl_train` |
| snapshot commit | `6ed83408d1d1cb02ea687e77fa80e5c20932f36c` |
| run | `round3-runs/train-formal01/train` |
| contract | `contracts/own_v40_round3_a.json` |
| contract semantic SHA-256 | `e31ba1538be4f26d5a5c6452ca8b06e1d09f41db999118957909d13f7248e73e` |
| contract file SHA-256 | `91ede15d5f5ae2de7d381e1f3a66f229005f213973f71084aca9a241335676c3` |
| parent checkpoint SHA-256 | `f94919f8f7ae1cfee95e45b922a9939e21c9f6ad9dff5656ff6f29dc0f4e6360` |
| 训练配置 | 1024 env、10000 updates、seed 43、locomotion、cuda:0 |
| warm-start | manifest 记录 `optimizer_reset=true`、`initial_iteration=0` |

远端 code 是 archive 部署目录，不含 `.git`；commit 身份读取自 `round3-a-deployment-<commit>/snapshot.json`。逐文件 SHA 已核对 snapshot 中的 `scripts/train_v40.py`、contract、`contact_sensor.py`、`env.py`、`v40_ppo_cfg.py`、`wheeled_world/assets/v40.py`，全部匹配。未导入模拟器验证这些文件。

实际 agent config 为 RSL-RL 5.5.1 split MLP，actor/critic 均 `[256,128,64]`，rollout 48 steps/env，5 epochs、4 minibatches；运行时 Torch 2.11.0+cu128、Sim 6.0.0.1。配置源码中的旧版本注释不作为运行版本证据。

可复查远端文件：

- `round3-runs/train-formal01/audit/train.started.json`、`train.log`。
- `round3-runs/train-formal01/train/{run_manifest,agent_config}.json`。
- `round3-runs/capacity-{256,1024}-cap01/audit/{train.log,train.status.json,worker.status.json}`。
- `round3-a-deployment-<commit>/snapshot.json`。

## 实际 RAM、RSS、GPU

单位：GiB = 2³⁰ bytes；`/proc` 的 kB 按 KiB 换算。采集使用 `free -b`、`/proc/meminfo`、`/proc/446292/{status,smaps_rollup}`、`ps`、memory PSI/cgroup，以及 `/usr/lib/wsl/lib/nvidia-smi`。SSH shell 默认 PATH 找不到 `nvidia-smi`，显式路径查询成功。

| 资源 | 实测 |
|---|---|
| WSL MemTotal | 16,311,783,424 bytes = **15.19 GiB** |
| 首次 MemAvailable / MemFree | 10,588,692,480 / 8,014,651,392 bytes = 9.86 / 7.46 GiB |
| 正式进程 RSS | 约 5,095,000 KiB = **4.86 GiB** |
| 正式进程 VmHWM | 5,098,632 KiB = **4.862 GiB**，是该进程至今 RSS 高水位 |
| 正式进程 PSS | 5,087,977 KiB = 4.852 GiB（03:25:43） |
| RSS 构成 | anon 3,091,040 KiB；file 883,540 KiB；shmem 1,120,256 KiB |
| Swap | 系统总 4 GiB、已用约 2.27 MiB；正式进程 `VmSwap=0` |
| memory PSI | 起止 avg10/60/300 均 0；累计 total 未增长 |
| 正式 session cgroup | `memory.max=memory.high=max`，`oom=oom_kill=0`；没有隔离新进程 OOM 的现成硬边界 |
| GPU | RTX 4090，24,564 MiB；初始使用 6080 MiB、利用率 37% |
| GPU 进程归属限制 | compute-apps 初始列出 PID 446292，但 WSL 返回 per-process memory `N/A`；以下显存均为**整卡** |
| CPU | 正式进程生命周期平均约 132% CPU、95 threads；32 个可见逻辑 CPU，loadavg 约 1.33/1.28/1.13 |

Windows 同时通过限时 PowerShell/CIM 只读查询：物理 RAM **33,444,524,032 bytes = 31.15 GiB**，FreePhysicalMemory **10,983,592 KiB = 10.47 GiB**；`vmmemWSL` WorkingSet 7,588,663,296 bytes、PrivateMemorySize 12,703,109,120 bytes。Windows free 与 WSL MemAvailable **不能相加**；VSZ、Windows private bytes、cgroup charge 也不能直接当作可用物理 RAM 或与 RSS 相加。

### 动态采样摘要

下表 RSS/available 单位 KiB，GPU used 单位 MiB；所有 GPU 峰值均是离散采样最大值，不是连续记录的启动峰值。

| BJT | 日志 iteration | MemAvailable | 正式 RSS | 整卡 used | GPU util |
|---|---:|---:|---:|---:|---:|
| 03:26:55 | 1032 | 7,456,612 | 5,095,296 | 7756 | 32% |
| 03:27:00 | 1035 | 7,471,668 | 5,095,296 | 7756 | 37% |
| 03:27:05 | 1038 | 8,094,872 | 5,095,296 | 7756 | 39% |
| 03:27:10 | 1040 | 8,095,584 | 5,095,296 | 7756 | 39% |
| 03:27:15 | 1043 | 8,092,404 | 5,095,296 | 7747 | 39% |
| 03:27:20 | 1045 | 8,090,216 | 5,095,296 | 7756 | 40% |
| 03:27:25 | 1048 | 10,328,584 | 5,095,296 | 6080 | 38% |
| 03:28:24 | 1082 | 10,323,216 | 5,095,296 | 6080 | 41% |
| 03:28:34 | 1087 | 10,333,228 | 5,095,296 | 6080 | 37% |
| 03:28:44 | 1093 | 10,336,240 | 5,095,296 | 6080 | 35% |
| 03:28:54 | 1099 | 10,335,616 | 5,095,296 | 6080 | 35% |
| 03:29:04 | 1104 | 10,334,676 | 5,095,372 | 6080 | 37% |
| 03:29:14 | 1110 | 10,333,620 | 5,095,372 | 6080 | 36% |
| 03:29:24 | 1116 | 10,334,872 | 5,095,372 | 6080 | 36% |

短时整卡显存增加 1676 MiB，WSL available 最低降至 **7.11 GiB**，正式 RSS 基本不变，之后恢复。可以确认资源状态变化，未采到该瞬时占用的进程归属，不能指定为某个 GUI/agent。该时段数据不是隔离 benchmark。

### 为什么不尝试“先开起来看看”

将当前 4.86 GiB RSS 写成固定项 F + env 相关项 V，在线性模型下，单个 2048 进程约为 F + 2V。仅凭一个规模的 RSS，粗略敏感性范围是 **4.86–9.72 GiB**；它不是实测预测区间，更不是启动峰值上界。库页共享可能减少增量，USD/PhysX 临时副本、接触容量与 allocator 峰值可能增加增量，尚无对应测量。

本次按保护正式 run 的要求，在 WSL 可用 RAM 中保留 **3 GiB 的评估余量**（工程判断值，不是 SDK 定额），新增任务只有约 **4.11–6.86 GiB** 的空间；不能覆盖上述敏感性范围高端，更无法覆盖未知初始化峰值及 GUI 并发。Windows free 也只有约 10.47 GiB。独立 tmux 和 timeout 只隔离生命周期，不隔离主机 OOM；几秒到几分钟后的超时无法保护已被 OOM 选中的旧训练。

## 吞吐与代码瓶颈

### 当前正式训练

实际安装的 `env/lib/python3.12/site-packages/rsl_rl/utils/logger.py:175–217` 定义：

`logged FPS = num_envs × 48 / (collection_time + learning_time)`。

它是 rollout transitions/s，不是 physics substeps/s；不包含整个进程启动，也不完整覆盖日志、checkpoint 等迭代外开销。

| 日志窗口（含两端） | 平均 logged FPS | collect s/update | learn s/update |
|---|---:|---:|---:|
| 803–902，复核用户原始基线 | 29,053.27 | 1.60142 | 0.09151 |
| 903–1002 | 28,802.11 | 1.61622 | 0.09181 |
| 1032–1047，短时资源占用期间 | 26,905.38 | 1.73119 | 0.09856 |
| 1083–1116，整卡恢复 6080 MiB 后 | 28,665.76 | 1.62453 | 0.09085 |

只读 wall 采样读取两个端点的已记录 `Total steps`，以 monotonic 实际时间相减：

- 03:26:55–03:27:25：50,774,016 → 51,560,448，**786,432 transitions / 30.028865s = 26,189.20 transitions/s**，16 个 update 增量。
- 03:28:24–03:29:24：53,231,616 → 54,902,784，**1,671,168 transitions / 60.060023s = 27,824.96 transitions/s**，34 个 update 增量。窗口跨过 iteration 1100 的 checkpoint 保存点。

这是正在运行的 1024 正式训练的 wall 观测，不是新启动 2048 的成绩。日志端点以完整 update 量化，60s 窗口约有一个 update / 819 transitions/s 的边界误差，另有日志可见性延迟；不能把它与 logged FPS 的差全部归因于 I/O。资源占用窗口 logged FPS 比原始窗口低约 7.4%，属于相关观测，未做因果归属。

原始窗口 collection 占 collect+learn 的 **94.6%**。即使把学习耗时完全消除，该计时口径的加速也仅约 **1.057×**。因此应优先关注 rollout 每步的 CPU 调度、GPU 同步、小 tensor 操作及 PhysX，而不是扩大 MLP 或仅追 GPU 利用率。

冻结 `src/wheeled_tasks/direct/v40_serial/env.py` 中，每 physics substep 的 `_apply_action()` 有 `if finite.any()`，每个 reward term 有 `if not torch.isfinite(step_reward).all()`；CUDA tensor 转 Python 条件会形成同步点，布尔索引/动态 reset 索引也可能同步。大量每 tick 的 reward/diagnostic 小 tensor 运算与这些同步是**代码支持的候选瓶颈**；本次没有 profiler，不能将各项耗时定量归因或断言 PhysX 本身已饱和。

### 初始化复杂度：不仅 contact sensor

- `src/wheeled_tasks/direct/v40_serial/contact_sensor.py:18–28`：先列出全 stage prim，再对每个 env 扫描全部 prim，为 O(NP)；P 随 N 增长时约 O(N²)。
- `src/wheeled_world/assets/v40.py:40–65`：每 env 扫描全 stage joint 集合。
- 同文件 `:170–199`：每 env 对全 stage prim 做路径筛选，再构建 collision filter plan。
- `env.py::_setup_scene()` 顺序调用这些步骤，并使用 `clone_environments(copy_from_source=True)`；stage、clone 和验证数据确实有主机内存开销。

| 既有 run | scene creation | simulation start | 10-update 学习 wall | 状态 |
|---|---:|---:|---:|---|
| capacity-256-cap01 | 20.931402s | 3.066394s | 16.967877s | 10 updates 完成，后续 CPU export 失败；不是通过的完整容量门禁 |
| capacity-1024-cap01 | 299.819115s | 6.511197s | 17.601127s | completed，export/hash 已验证 |
| train-formal01 | 299.321413s | 6.590159s | 持续训练 | 本次未触碰 |

256→1024 的 env 为 4×、scene creation 为 14.32×，符合近二次增长趋势；学习阶段 10 updates wall 只增加约 3.7%，而 samples 增为 4×。1024 capacity 的 aggregate 为 **27,925.49 transitions/s**，与当前 wall 量级相符。历史时段未重新确认所有并发负载，不能直接当成严格受控缩放实验。

按 1024 的约 300s 建场 ×4，**2048 建场约 1200s（20 分钟）**，并跑可能更慢。这是初始化时间估计，不是 RSS 四倍估计；上述扫描并未显式保存 N×P 矩阵，不能由 O(N²) 时间直接推断 O(N²) 峰值内存。单独修复 contact sensor 也不会消除另两处嵌套扫描。后续版本可考虑一次遍历建立按 env 分组的索引，同时保留现有完整性和顺序检查；本轮不改冻结代码。

## Sample budget 对齐后的选择

固定 rollout 48：

| 方案 | 每 update transitions | updates | 总 transitions |
|---|---:|---:|---:|
| 当前 1024 | 49,152 | 10,000 | **491,520,000** |
| 2048，等 sample budget | 98,304 | **5,000** | **491,520,000** |
| 2048，仍做 10,000 updates | 98,304 | 10,000 | **983,040,000** |

同 updates 的 env 翻倍等于 samples 翻倍，不能承诺结束更早。以 27,825 wall transitions/s 外推，当前规模完整 sample budget 的学习 wall 约 **4.91h**，另加启动/最终导出；这是稳定吞吐外推，不是剩余时间或完成承诺。

以下仅是下阶段决策用的条件计算，**不是 2048 实测预测**：

| 假设 2048 wall transitions/s | 491.52M samples 学习时间 | 再加约 20min 建场（未含其他启动/导出） |
|---|---:|---:|
| 30,000 | 4.55h | 4.88h |
| 40,000 | 3.41h | 3.75h |
| 50,000 | 2.73h | 3.06h |

只有测得更高的有效 transitions/s 并摊薄额外建场成本，2048 才能缩短等 sample budget 的 wall。等 samples 也不等价于训练轨迹：2048/5000 的 batch 和 minibatch 更大、优化器更新次数减半，adaptive KL 学习率演化及 100-update checkpoint 对应的 sample 间隔也改变。最终还要比较策略质量，不能把工程吞吐收益直接当成 sample efficiency 收益。

下一次容量测试宜使用同 contract、同 parent fresh warm-start、独立 run/cache 和 3–10 updates，并明确完整启动硬超时；短学习预算不能省略约 20 分钟的预估建场。记录全生命周期 RAM/RSS、整卡 GPU 采样峰值和实际 wall transitions/s。若届时与 1024 正式训练并跑，应明确标注并发状态，同时对照旧训练 FPS；这类成绩不能冠以单进程纯 benchmark。本次资源证据不足，未执行该测试。

## 验证范围

实际完成：运行身份和六个冻结文件 SHA 核验、Windows/WSL RAM、正式进程 RSS/HWM/PSS/swap、GPU 动态采样、30s 资源波动窗口与60s恢复窗口 wall 计数、历史容量/当前日志解析、实际 logger 计时口径及初始化/同步点源码审阅。03:30:08 BJT，原 PID、tmux 和更新进度正常，memory PSI 累计未增长，正式 session cgroup OOM 计数仍为零。

本次唯一新增交付为此文档；2048 的 FPS、RSS、启动峰值和多进程 aggregate throughput 均未实测。没有修改代码、合同、资产或模型，也没有提交。
