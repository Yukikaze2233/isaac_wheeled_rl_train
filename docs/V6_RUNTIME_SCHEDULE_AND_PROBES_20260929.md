# 新资产课程时钟、请求协议与 CPU 功能探针

本实现只在冻结工作区更新训练代码；主动轴连续/物理膝止挡事实见 `V6_CONTINUOUS_AXIS_AND_STOP_DATA_20260929.md`。本说明不是 GPU 场景验收、硬件辨识或策略能力通过回执。

## 时钟和样本

`training_schedule.py` 读取 runner 写入的 `global_actor_update` 与 `stage_actor_update`；阶段计数优先使用后者，不按环境数或 transition 数换算。S1 固定九档/连续高度混合在阶段第 200 次实际 actor 更新从 80/20 切为 50/50。后续 cross-height 连续样本可用 `boundary_fraction_of_range=.1,boundary_weight=2`，将研究高度区间两端各 10% 的采样密度加倍；独立端点配额保持。这个区间不是硬件可达域已验收声明。

静止 jump 按七档和配置概率在 episode reset 采样；确认通过的档位仅转移超出最低 5% 概率部分的 50% 给未通过档位。晋级名单读 `env.confirmed_jump_bins_m`，没有凭训练迭代自动宣称通过。`full_tasks.cfg` 的逐环境 apex 与 COM-rise 判定同步更新。行进跳跃保留各规格的 0.03/0.06/0.10 m 目标。普通命令重采保持当次跳高、manual 请求起点、保持时间和每 episode 的随机化身份。

带显式请求事件或命令序列的协议规格保留固定 0.10 m 目标，不参与普通七档重采。`ground_after_recovery` 还等待本跳达到完整任务验收并计入 `successful_jump_count`，之后才发下一跳或恢复移动，避免提前清掉一秒稳定验收历史；协议计分同时要求物理完整跳跃数与任务成功数。

S3 起使用固定在 episode 内的随机左/右侧三轴 0.5 倍输出降级，抽样率 5%。10% 的边缘恢复先验从合法既有出生姿态施加有界速度：vx ±0.3 m/s，roll/pitch/yaw 角速度各 ±0.5 rad/s；不移动或旋转 root pose，避免穿地。这是合法姿态上的物理扰动，不是从验证轨迹提取的坏姿态状态库，也不替代姿态域恢复验收。

## 请求和技能序列

随机训练请求由配置的驻留和 stress 比例生成。显式 `skill.request_events` 在固定评测中仍执行，随机化保持关闭。字段为 `time_s`、`type=new/repeat/cancel` 和可选 `after_phase=flight/ground_after_recovery`。`after_phase` 等实际状态到达后才递交，不能用固定时刻假装机器人已腾空或已落地。

忙时重复不重置 reference 起点；腾空或 LANDING 时的 cancel 保持落地请求。经过实际 FLIGHT 后进入 RECOVERY 时释放请求 latch，避免状态机把同一个持续请求解释成新的起跳；状态机随后靠原有稳定接地计时回到 GROUND，才增加 `completed_jump_count`。该流程不直接改写物理 phase。下一次明确 new 请求开始独立周期，消费端需重置每跳 COM/腾空/成功历史，不能复用第一跳的峰值。

`skill.command_sequence` 依序发出已有三维命令和 mode，支持时间与恢复门控，不增加观测或网络结构。跨技能序列为前进、停止、跳跃、实际恢复后重新前进。`skills.sequence_pending()` 包含未发完事件、未结束跳跃恢复及尚未执行的序列部分，消费端须抑制提前 episode success，同时只能给一次成功事件奖励。

`RequestProtocolMetrics` 按 episode 保存事件计数、cancel phase、完整跳跃数和技能序列跟踪秒数。固定评测拒绝缺日志，要求每个 replica 分别通过。重复、起跳前取消、空中取消、连续两跳均有专属判定；跨技能序列要求初始前进跟踪至少 0.5 s、停止至少 0.5 s、恢复前进至少 1 s。普通单跳 success 不能替代这些判定。

## 可运行的启动前探针

从冻结包根目录运行：

```sh
python scripts/validate_v6_functional_probes.py --contract runtime.json --output functional_probes.json
```

脚本直接实例化 CPU Torch 模块，按真实输出执行断言，记录依赖源码 SHA-256；失败写 `passed=false` 并非零退出。它不导入主工作树，不启动 GPU/PPO。

- `UsbCommandTransport`：1 ms 物理/1 ms CAN，逐 tick 检查配置 0/0.08/1.2 ms 延迟、disabled、全丢包和固定每 4 包连丢 2 包，输出实际力矩与命令 age。启用通路即便配置 0 ms，新批次也最早下一 CAN tick 消费；只有 disabled 通路是当 tick 直通。
- `V5SignalPerturbations`：固定 seed 重放、启用/关闭 mask、观测和动作 lag 范围及实际迟滞、同 tick 重读不推进、命令/模式槽不受扰动、每 episode lag 恒定。Gaussian 噪声的实测最大值只是样本统计，不是硬上限。
- `TrainingSchedule`：高度切换、确认晋级后 bin 概率、驻留区间、单侧降级恒定性、roll/pitch 角速度扰动、busy repeat 与空中 cancel 的实际状态行为。

离线验证包括功能入口失败路径和反例（将启用通路错误改为直通必须失败）、真实 PhaseTracker 恢复联动、固定事件的真实 phase 门控、逐 replica 协议计分及旧课程回归。远端冻结后应重新执行此脚本生成该部署源码的独立回执；本地 passed 不替代远端启动前结果。
