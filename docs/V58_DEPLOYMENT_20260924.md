# V5.8名义恢复部署回执

方案与失败核对：[V5.7复盘及V5.8修正](V57_FAILURE_AND_V58_RECOVERY_20260924.md)。
执行合同：`contracts/v5_nominal_recovery_v58.json`。
机器可读回执：[evidence/v58_launch_20260924.json](evidence/v58_launch_20260924.json)。

## 候选选择

在Kaiser使用同一27案例、每案例4个episode的固定评测重新对比：

| 候选 | 通过/27 | stand高度MAE | stand漂移 |
|---|---:|---:|---:|
| 原12486 | 18 | 1.932mm | 10.441mm |
| V5.7 5000 | 14 | 2.901mm | 10.810mm |
| V5.7 6500 | 5 | 12.344mm | 301.248mm |

选择原12486 actor，SHA `e6307dd7419c052a96285e58ce6a6624b8832e6c5986aadf568287bfcfc37f81`。
评测原件已回收到`reports/v58_seed_comparison_20260924/`，不是根据训练MAE或最新文件时间选种子。

V5.7在WSL调整前后的最终记录为10027更新（平地9750＋接触恢复277），状态`stopped`。
新轮不继承它的退化actor或优化器。

## 已落实的修正

- 50%原能力排练：原名义高度、命令范围和幅值采样；已学组不受初学cap限制。
- 50%补四项任务：后退0.5m/s、前后1m/s精度及反向4rad/s旋转。
- 27案例分成18保持、4学习、5观察；保持集另外报告相对初始基线的量化margin，固定通过判据不改。
- 原固定高度核及平方项；关闭按组变核、宽伴随项和新增位置锚点，避免课程同时改变奖励目标。
- 未学任务的晋级要求真实近边界、稳定、有足够时长的样本；无边界曝光不会虚假晋级，早期失败仍被统计。
- 边界回退节流；变更事件记录真实评分命令和时长，并在PPO更新边界额外保存快照。

## 并行与预算

WSL上限已从约15GiB增至约23GiB；宿主物理内存仍约32GiB。
正式使用**12288环境**，相对8192增加50%，相对6144翻倍；按用户要求没有再运行容量探针。

- rollout24，每次更新294912 transitions。
- 6个minibatch，每个49152条样本，保持8192环境／4 minibatch时的批量。
- LR=3e-5，新critic／优化器；critic预热34更新，近似等效8192环境下50更新。
- 阶段上限667实际更新，约等效8192环境1000更新；总计196706304 transitions，最长6小时。
- 每250更新评测；每100更新、首次10更新及边界变化时封存。
- 本合同只执行这一个恢复段，结束后根据分层结果续约，不自动进入七个后续长阶段。

## 运行身份

首个启动在分块入口的旧8192上限处退出，优化更新数0；已修正并补12288／16384入口回归测试。

- 当前冻结commit：`bf4b85e244cd8cb6e812d11ac260f939816ce838`。
- 课程实现commit：`6006078f87c1cccb8e9d272ec902556e995acad6`。
- 启动：2026-09-24 21:04:40，北京时间。
- 根目录：`/home/kaiser/robot-rl-sim60/experiments/v5-scut35-20260924T130433Z-87c442`。
- 源码归档SHA：`dba8d6c74c685a346b979cd065ef3660ac649a164279c6ef5363baa3a13ba630`，无overlay。
- 启动回执：`reports/v58_formal12288_retry_20260924/launch.json`。
- 119项课程／接口相关测试及2项大并行入口测试通过；阶段合同preflight通过。

### 首次正式运行核验

21:13:05北京时间，已完成26更新、7667712 transitions，仍在34更新的critic预热阶段；
supervisor、worker和tmux均存活，初始27案例复验仍为18项通过。

第10更新快照已回收并实际加载：actor／critic参数有限，课程schema v3包含20个训练组，
已掌握的前后3m/s组仍保留3m/s边界；actor全部参数与原12486检查点逐元素相同。
优化器此时有8项critic参数状态，符合预热阶段尚未更新actor的预期。
快照SHA：`7447dd1241917e27d9cd20326d89baac96cc9e99eea4d554a1316863b1e14a43`。

TensorBoard早期最近10点吞吐中位数46949.5 transitions/s，采样约6.17s／更新、critic学习约0.107s／更新。
查询时GPU利用率76%、显存10334MiB，WSL可用12649316KiB。
这是当前预热阶段的实测，不是严格控制变量的跨版本加速比，也不代表新策略已经恢复成功。

21:17:21复查已到66更新、32次actor更新，进程继续存活。
第42更新出现首个已核验的边界事件快照：反向4rad/s修复组的学习上限1.0→1.5rad/s；
真实评分命令均值0.930rad/s、评分时长均值17.97秒、高度误差4.60mm、yaw误差0.089rad/s、完成比例1.0。
该快照已回收并加载，17项actor／critic优化器状态完整，已掌握forward_3仍为3m/s边界。
事件是课程学习证据，不等于固定反向4rad/s案例已经通过；第一次新策略固定评测在250更新后进行。

## 监控与恢复

TensorBoard：<http://127.0.0.1:6006>，`current/`对应V5.8，`v57/`保留旧轮。
本机使用持久user service，并已加入登录启动：

- `v58-tensorboard.service`：本机查看器，不消耗Kaiser的训练主存。
- `v58-event-sync.service`：每30秒同步event文件到`reports/v58_tensorboard_active/`。
- `v58-artifact-recovery.service`：每60秒校验回收封存产物到`reports/v58_active_batches_20260924/`。

训练在Kaiser独立tmux中运行，本机查看器或SSH断开不决定其生命周期。
持久服务解决本机临时unit重启后丢失的问题；硬件断电或关闭WSL仍需依据已保存检查点恢复。
同合同恢复包含优化器、课程v3状态及CPU／CUDA随机数；不把预算结束当作异常无限重启。

```bash
python3 -B scripts/check_chassis_remote.py docs/evidence/v58_launch_20260924.json --brief
systemctl --user status v58-tensorboard v58-event-sync v58-artifact-recovery
```
