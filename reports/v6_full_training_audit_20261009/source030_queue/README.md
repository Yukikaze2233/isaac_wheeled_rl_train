# source030：一次正常 speed 边界接续队列

本目录只实现外部一次协调器；不改 source029/source030 冻结源码、合同或部署 helper。默认只读；本次本机验证没有 SSH、信号、仿真或 PPO，队列尚未部署。source030 是静态候选，不是已学成的大台阶模型。

## 唯一适用入口

实际 source029 正在完成其已付费 speed 更新。正常目标为 **14,897 actor + 10 critic = 14,907 更新**，包括此前 speed 已消费 5,093；不能把目标写成新的 15,000 actor，不能退款。约 14,000 余次尚需数十小时，不能现在用默认 3,600 秒等待并误称可自动接续。

协调器调用已冻结 `hold_completed_speed_boundary.py`，显式等待最多 **604,800 秒（7 天）**。原控制器只在最后 **20–100 actor** 的真实窗口暂停认证的 supervisor；speed worker 继续正常 PPO。worker 真正退出、终局 full seal 与实际消费认证后，控制器才对 supervisor 执行原生停止/继续，保留所有记录，并在原 source029 下完成 **0 PPO** 终局记录闭合。协调器本身没有 signal/kill API，不增加早停、退款、评测 gate 或重试。

只接受 `native_supervisor_stopped_after_normal_speed`、`worker_signals=0`、`next_domain_dispatched=false`、原生闭合 `closed_native_completed_speed`。随后严格顺序为：

1. 从实际控制器 receipt 算 SHA，交给原 helper 的 `boundary-prepare`。它认证真实 completed-speed 父、账户、原 REC/speed 身份和完整恢复状态；未来 terrain actor 从 0 起，10 critic 正常计费。
2. 准备认证通过后，独占硬链接保存 helper 已认证的旧 `delivery.tar.gz`、`delivery.json` 与 `archiver_status.json`。不重新压缩或覆盖旧字节。
3. 再核对冻结输入，调用同一 helper 的 `launch`，只提交一次原生 wrapper。提交回执仍是 `first_terrain_actor_pending`；实际首 actor 要由既有只读核验器另行验证。

## 冻结输入

| 输入 | 固定身份 |
| --- | --- |
| source030 commit | `81ad124554c4ab45cc274e2d482d46c76ba125b0` |
| Git archive | `7afdf9e0fdb08c217d1f034f88ad52f7dad1eb7af1f86c9d710347cf5b0a9dab` |
| 原 deploy helper | `612c66e8f7cbdd56d65ed88a1861c0d2add1fcefba33704adfa177664bcea6cb` |
| 实际 stage.json | `870194901f935c9c024996b0e05317f008fbf3f067039355ec14c5c4fbf42177` |
| 原 runtime | `c76cc88f7273a400430ded6251ebc32d984a36fed366733a27ed76e2c94dcff7`，776 B，0644 |

runtime 是 `/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh`，由 bash `source`，不要求可执行位；不修改其 OMP8。只读等待进程使用 OMP/MKL1，原 wrapper 启动时重新 source 原 runtime。

启动控制器之前，复用 SHA 已认证 helper 的 `archive_files(..., verify_tree=...)` 校验全部 **4,341 文件**的归档、权限及解包字节。控制器/原子 publisher 两个直接入口另行 SHA 核对。等到正常边界和提交 launch 前重新校验，防长等待期间源码、helper、runtime 或解释器漂移。

## 本机 CPU 验证范围

`tests/test_coordinate_source030.py` 用临时 synthetic stage/archive/账户回执与 spy 执行检查唯一调用顺序；同时真实导入冻结 helper 的归档校验与原 `durable_json`，以及临时 Python 子进程持有的真实 `flock`。不会运行真实控制器、helper 的准入/launch、原生账户图、Isaac 或训练。

覆盖：只读默认；7 天显式等待；实际 receipt SHA 接入；先保留旧交付再 launch；控制器/准备/launch 错误不重试；直接入口与非直接依赖字节/权限漂移拒绝；一次 intent 不覆盖；另一持锁进程拒绝；原子替换交付路径后旧硬链接字节保留；取消 Popen.wait 不主动 kill 控制器。它们证明协调器接线和 guards，不授完整恢复迁移或任何机器人能力资格。

```bash
python -B -m unittest discover -s tests -v
```

## root 审核冻结后的远端调用

以下是待审核的调用方式，不代表已经执行。协调器脚本本身应按最终 receipt 的 SHA 上传并核对；不要只复制可变文件名后直接执行。

```bash
python -B /绝对路径/coordinate_source030.py \
  --run-root /home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358
```

只读输出为 `validated_once_queue_not_started`，不会建立队列锁或 intent。正式一次入队加 `--apply`；运行于独立持久进程，不依赖 TensorBoard 或浏览器。

## 故障边界

固定 `source030_queue.lock` 贯穿控制器、准备与提交；独占 `queue.intent.json` 阻止另一次同路径重排。它不持原生账户锁等待 worker，也不占用原控制器锁造成自锁。

所有原 stdout/stderr、started/execution 及 `status.json` 保留。任意异常为 `attention_no_retry`；没有自动删除 intent、重新启动、补学或退款。native 控制器自己拥有有限等待及正常异常清理；协调器不设会杀 child 的 subprocess timeout。

若协调器被 TERM/KILL、宿主断电或父进程取消，子控制器可能继续运行；**不能承诺所有崩溃都有 finally**。必须检查原持久 intent、PID/starttime、控制器 receipt 与 prepare/launch 产品，再决定人工处理。不得重排另一个协调器“补上”。默认 PID 改变、原 worker 接续、不同父或不同来源都由原控制器/准入拒绝。

硬链接保留适用于同实验、同文件系统及原 wrapper 使用原子 `replace` 发布交付文件的既有流程；不允许之后原地改写旧 inode。保留 SHA 文件名和 `old_delivery.json` 可复核。若保留过程中失败，部分硬链接仍保留、不会 launch；失败产品不覆盖、不自动重试。新课程是否改善普通上阶、高速、跳跃及倒地接稳，仍需实际新 actor 和同协议评测。
