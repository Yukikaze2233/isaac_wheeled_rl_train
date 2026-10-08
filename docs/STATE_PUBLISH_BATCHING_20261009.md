# 状态快照批量取回与独立计时

基线：`f99d3cbb556db873a299b8c9bc2cd68fcf34957a`。这是本机隔离性能候选，
不修改物理、奖励、PPO、训练合同或发布节流条件；没有部署。

## 原调用链与改动

`train_chassis.py` 的策略步 wrapper 在 `original_step`、诊断完成后，按原有
`--publish-state` 与 `time.monotonic() - last_publish >= .25` 条件写
`live_state.json`。状态是 **post_step_after_auto_reset**，不是终止前的评测样本。
`view_chassis_live.py` 用根节点的 env 0 字段及逐组 `environments` 列表渲染镜像；
该快照不参与策略控制。

旧实现每次重选每个组首次出现的环境。姿态虽然已有一次逐组批量读取，原点、命令、
episode 步数仍逐组读回，env 0 还被重复取回。对于 G 个组，有 `3G + 4` 处
`.cpu()` 或 `int(tensor)` 主机物化调用；190 组对应574处。这是调用结构计数，
不代表574次独立内核同步或已测得耗时。

新 `state_snapshot.py` 只负责选择与读取：

- 在 worker wrapper 初始化时缓存首次出现顺序的索引 tuple 和同 device 的 long 索引张量。
  场景分组在该 worker 内固定，episode 重置不改变分组；不会每次发布重新扫描。
- 姿态、原点、命令和 episode 步数分别 `index_select → cpu → tolist`，合计四次批量取回。
  各字段保留自身 dtype；episode 的 int64 不会装入浮点 pack。
- 根 env 0 复用批量结果首行。首次出现顺序必然从 index 0 开始，所有组、字段、
  quaternion 顺序、surface 内容、JSON 编码选项与原实现相同。
- 每次仍读取当前值；只缓存选择索引，不缓存实时姿态或命令。没有输入、RNG 或训练状态写入。

发布频率上限、时间戳含义、`live_state.tmp → replace(live_state.json)` 原子替换与
成功写完后更新 `last_publish` 的顺序不变。停用发布时不创建缓存，也不读取快照。
启用发布时把新 helper 加入运行 `source_files` 摘要；关闭时来源列表不变。

## 计时

整个读取、JSON 编码、写文件、原子替换及发布时钟更新通过已有
`timed_call(timing, "state_publish", publish_state)` 包裹。原 `TrainingTiming`
动态注册 `Timing/state_publish_seconds`、`Timing/state_publish_calls`，从原来的
unattributed 中分离这段主机墙钟，沿用原 event 文件与 step。

它是独占主机计时，可能包含等待先前 GPU 工作的时间，不是 GPU 内核计时；没有新增
`cuda.synchronize`、后台发布线程、频率降采样或独立日志系统。

## 验证范围

CPU 差分测试直接提取冻结基线与候选实际 publisher：

- 3／190 组、连续／非连续张量、单一／混合浮点 dtype 的完整 JSON 字节一致。
- 原点 float64、命令 float16、姿态 float32、步数 int64 各自保持；`2^53+1`、
  `2^62+31`、`2^63−1` 步数精确保留，且没有 tensor 标量读取。
- 索引只构造一次，实时命令与步数仍更新；env 0 与首个逐组环境一致。
- 0.25 秒边界、禁用路径、原子替换及 NaN 编码失败时保留旧文件和旧发布时钟。
- 实际 wrapper 使用独占 `state_publish` 计时；除发布实现外其余训练 AST 与基线一致。

与既有 timing 测试共同通过18项 CPU 检查。未启动原生场景、PPO、SSH 或 CUDA
基准；没有整机训练提速结论。先前其他 reporting 微块的 GPU 数值不能套用到此快照。
