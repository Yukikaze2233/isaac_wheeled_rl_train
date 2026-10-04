# 新增逻辑的性能审计

2026-10-05。当前部署仍为冻结源码 `b4ce80964d47`；下面只是离线审计与局部微基准，没有修改或重启 Kaiser PPO。

三个新增 reward 成本在 50 Hz policy 侧计算；松杆命令、相位统计也是 policy/reset 侧工作。非标准出生的主动力矩掩码和时钟推进在 200 Hz physics 侧执行。没有新增场景、训练 worker 或调度线程。不能把全部 wall/env 变化归因于这几个新增项。

`nonstandard_start.py` 的 `elapsed[held] += dt` 有 CUDA 布尔索引同步。临时全长张量表达式 `elapsed.add_(held * dt)` 与原式比较：4096 lanes，本机 RTX 4060 Laptop，0%/1%/100% hold；每组四次、每次800步，报告中记录全部重复值。原式16步有32个 `aten::nonzero`、32次 `cudaMemcpyAsync` 和32次 `cudaStreamSynchronize`；候选表达式均为0（测量末端显式同步单列）。

| hold 比例 | 原式 µs/physics步 | 候选 µs/physics步 | 96物理步估计节省 |
|---|---:|---:|---:|
| 0% | 52.25 | 34.08 | 1.74 ms |
| 1% | 70.47 | 33.20 | 3.58 ms |
| 100% | 70.34 | 34.20 | 3.47 ms |

CPU/CUDA × 三种hold比例 × 39/40/41物理步边界，共18条轨迹，各96子步；每个子步的时钟、状态、力矩、learning/bootstrap mask 均逐值相同。200 ms hold仍是40步，之后才允许主动矩。

结论：同步点真实存在，但单更新仅毫秒量级，相对于当前约14.7秒 wall不值得单独重启训练。候选未写进运行源码；局部1.5–2.1倍不代表整轮PPO提速。CUDA event区间包含CPU launch停顿，profiler带测量开销，本机微基准也不能直接外推Kaiser。

可复现代码和完整数据分别为 `nonstandard_advance_microbenchmark.py` 与 `.json`。该程序仅使用Torch，既不创建Isaac场景，也不访问Kaiser。学习审计的 `drain()` 已一次批量转CPU，不应再次把它误判为逐tag同步；RSL episode logging、额外诊断和未归属 wall仍需独立归因。
