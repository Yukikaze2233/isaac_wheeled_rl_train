# source028 原生启动 CLI 修复独立审查

日期：2026-10-09（UTC+8）。仅只读源码/保存回执与三个聚焦CPU测试；0 SSH、GPU、PPO或生产修改。

## 判决

**无剩余具体源码缺陷；本修改与真实CLI认证根因一致。** source028失败调用没有 `--learning-rate-scale`，`train_chassis.py` parser默认值为None，而 `integrated_production.authenticate_invocation` 强制要求1.0。因此在 `train_chassis.py:404`、AppLauncher创建之前拒绝。实际保存的失败exit1、wall10.390s、没有scene/report与PPO launch intent；不应称物理仿真失败。

本审查通过AST只提取并执行原生parser声明，解析`boundary_live007.json`保存的真实失败argv，再添加三flag。结果：

| CLI字段 | 原真实argv解析 | 修复argv解析 |
|---|---:|---:|
| learning_rate_scale | None | 1.0 |
| global_actor_update_offset | 0 | 0 |
| block_index | 0 | 0 |

其余账本、父模型、stage/task时钟、seed、4096、scene1参数完全保持；唯一改变的解析值是LRscale。此修复不给训练加/减消费、不更改LR合同本身，也不改变PPO参数。`runtime_directory()`抽出原路径表达式，默认仍是`boundary_integrated_028.native`；它提供独立命名失败修复的复用点，**本身并不授权在失败目录重提**。

聚焦CPU实跑两个test节点，含24/25参数化：3 passed / 1.02s，项目`isaacsim60-venv`。首次系统Python收集失败（缺torch）只属于测试运行环境，负结果保留在工具输出；不冒称该轮通过或叠加为测试项。

## 显式 named 一次修复必须核验的guard

1. **冻结修复辅助代码身份**，绑定CLI修复SHA与已review的命名目录；生产source335/归档450489…/plan db693…/auth31e4…全保持。不得修改旧source/archive、028 manifest、失败intent/receipt/log。
2. **确证失败在仿真之前且没有PPO提交**：认证失败stderr/argv/exit1与receipt SHA，scene/report缺失、旧launch intent与queue receipt均无；发现不明原生部分执行或已有launch意图就拒绝，不能一律重试任何failed。
3. **真实旧所有者退出/归档完成**：wrapper/supervisor/worker PID+starttime已退，archiver finished，旧tmux/job delivery无未完成状态；核对运行根唯一训练owner。另一项目不属于本次停止/修复范围。
4. **旧SIGTERM仅一次**：继续保留028 signal intent与它认证的授权floor，不再次发送信号、不回滚REC或重造15000端点。
5. **真实REC父与账本不变**：final SHA ed24fe4…、sealed=true、REC14169、stage27537、lineage27627、消费41374/actor消费41204；原成功7754=10critic+7744actor，stopped/sigterm；相应completion、campaign与immutable origin原字节/SHA绑定。
6. **复用原native admission/prepare认证**：mode=rebase_from_authorized_REC，fresh speed task0，既有四域合同/reward/DR/exams/预算来源保持；资格为null。所有旧消费照计，10critic适应按原allowance计费，critic不推进actor课程。
7. **新目录排他创建、named修复唯一**：使用单独且合同外的原生probe目录，绑定原失败intent/receipt SHA、原manifest SHA与selection身份，exclusive intent/receipt/log；新named intent若部分存在则拒绝，不能删除后重提。沿用共享边界lock，不并行第二observer。
8. **再次静态准入后做真实native1**：明确LRscale1、global0、block0，4096×1策略/4物理；exit0、scene_validated、runtime_configuration_passed、目标worker/selection/report哈希一致、成功/尝试PPO均0、qualification=null。CPU通过不能代替这一物理读回。
9. **原始028失败产物永远保留**：新passed probe需在独立修复回执中关联旧failed，不覆盖旧receipt或改成passed，不冒称full行为评测/资格。
10. **只有通过才一次launch**：真实prepare/preflight复核后，以原排他launch intent/queue receipt一次启动wrapper；再次核验唯一owner链、首10critic与首actor真实9/8/17张量和Adam/RNG恢复、真实消费/谱系/新task时钟。若出现launch意图不明，不自动重提。

`boundary_live007`保存读回为2026-10-08 19:37:56 UTC（10-09 03:37:56 UTC+8）：owners空、旧archiver finished、REC14169 sealed、新native failed且PPO_actions=0。它只是该时刻证据；执行前仍需最新只读所有者/once产物复核。

## 本次绑定的输入SHA

- `/home/yukikaze/Documents/workspace/robot_rl/.worktrees/native-cli-repair-20261009/scripts/queue_task_campaign_boundary_continuation.py`：`fd2f874e1a9338a70e2889a965d732de5be3c737b684d4b9e3c346e50d7917f9`
- `/home/yukikaze/Documents/workspace/robot_rl/.worktrees/native-cli-repair-20261009/tests/test_integrated_boundary_continuation.py`：`96707dd1785765d6fd86b13e27e1ea1857284898305f1596a93db6eb112289d5`
- `/home/yukikaze/Documents/workspace/robot_rl/.worktrees/native-cli-repair-20261009/tests/test_integrated_early_boundary.py`：`5f69fa26e1c270552215f26ff380e516725cf9fa674631f06f15149ddcef7bc8`
- `/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/source_repair_028/boundary_live007.json`：`a64b660e8e6eab47eda833e8a867cd91ee1704ad45684bce83f3c9f5a7cf0f02`
