# 一次等待 controller 取消准备

本文件对应 `cancel_waiting_source030.py`，尚未在 Kaiser 执行取消。它只为已经静态暂存、通过 30 角色 CPU 预检的 source031 腾出未来边界 observer；不改变正在运行的 source029 PPO、预算、合同或已消费学习。

只允许原 coordinator `4166067/start114488334` 的等待 controller 子进程 `4166078/start114488781`，且当前 speed 剩余 actor 更新必须大于 1000。实时认证旧 source030 全部冻结输入、新 source031 4343 文件与 tar/源 commit、正式三 owner、controller 独占 FLOCK、旧 v26 claim/state、等待 phase、原始 queue intent 和未产生任何边界/prepare/entry/launch 产物后，显式 `--apply` 才写排他 intent 并打开 PIDFD。

PIDFD 打开后再次认证全部身份与输入；唯一 TERM 发给该 controller。父 coordinator 自然记录 `attention_no_retry/phase=controller` 和 controller `exit_code=-15`，自然退出后核对锁释放、原三 owner 不变与账本不回退。60 秒观察上限届满仅保存失败证据；不会再 signal、kill、retry 或退款。失败保留 intent，禁止二次执行。

固定新输入：commit `ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a`，stage SHA `b7dae15f3878aa92039580387bc98e526078fc4f617064a35c18da8994f93828`，preflight SHA `9c14b08e90dc4eb7a8f3933ba29035e025b3b36083cae37de64002445bc9a83f`。根代理的实际远端只读回执为 `../source031_static_remote_readback001.json`；本 helper 的 CPU fixture 不替代该远端真实性证据。

在 Kaiser 使用与原 observer 相同的解释器；`CANCEL_HELPER` 应由根代理转移为校验过 SHA 的脚本实际路径。以下为命令准备，本文未执行它们：

```sh
KAISER_RUN=/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358
/home/kaiser/robot-rl-sim60/env/bin/python -B "$CANCEL_HELPER" --run-root "$KAISER_RUN"
/home/kaiser/robot-rl-sim60/env/bin/python -B "$CANCEL_HELPER" --run-root "$KAISER_RUN" --apply
```

输出使用 `source031_queue/cancel_source030.intent.json` 与 `cancel_source030.receipt.json`。intent 的 provenance 位于 `snapshot`，receipt 的 `old_controller/old_coordinator` 复用原 native identity（`starttime/ppid/entry/command`），并加 `exited=true`。新 source031 coordinator 读取真实这两个文件并复核旧产物、退出、FLOCK 和候选源 SHA，不能只依据成功字符串。

CPU 44 项通过（0.24 秒）：实际取消函数/排他 JSON/真实 fixture FLOCK、只读默认、PID 复用、18 类旧 transition 产物、末 1000 actor、安全阈值/phase/源/账户、PIDFD 打开后的身份/源/intent 变更及超时一次发送。native admission、进程表、归档服务和 PIDFD signal 为合成服务；没有真实信号、SSH、Isaac/GPU、PPO、账本写入或能力资格。另一个 peer 的跨模块测试直接调用此真实取消函数，再把产物交给新队列 guard，单独记录其结果。
