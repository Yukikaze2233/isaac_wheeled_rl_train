# source030 队列运输入口最终独立审查

当前 `run_source030_queue.py` SHA `a114e0ed39598505fe69b630649d85eb959f1b1f4b6876ed1919d4d2a7a89479` 无剩余静态阻断。独立重算与 root 的四份远端片段语法回执绑定一致；coordinator `0478bfe…eb339`、C peer receipt `2d1afa…f50b75`、作者 CPU receipt 和五份工件 SHA 仍一致。没有重跑 16 CPU，也没有 SSH、信号、GPU、PPO 或原合同变更。

以下顺序与守卫已逐行核对：

- dryrun 在执行之前验证远端 script 的 nonsymlink、0644、0478 SHA。
- runtime c76 的真实 SHA、0644、776 B 在 bash `source` 前认证；环境只在远端捕获，不打印，去除 `PYTHONPATH`，等待 OMP/MKL1。原新训练 wrapper 之后重载原 runtime。
- enqueue 在导入 publisher 之前先认证固定 stage870 的 nonsymlink/SHA，再核对 stage inventory 中的 publisher SHA。
- 原 wrapper/supervisor/worker 的固定 PID/starttime、非 Z/T 状态、原 job 精确 argv、PPID 4141407→4141412→4141576 以及原 `owned_processes(root)` 的精确三 owner 集合都要一致。
- 本机 action `.intent.json` 的 `open('x')` 防并发；Git 最后提交的运输工具 blob 必须与实际源码逐字节相同，才可创建 intent。
- 远端先通过原 atomic publisher 独占 launch intent，再 Popen 一次协调器：`start_new_session=True`、stdin DEVNULL、独占日志，没有会对该协调器执行 kill/terminate 的 Popen context 或等待 timeout。

工具的 180 秒 timeout 仅用于有界本机 SSH 运输调用，不是 7 天边界协调器的运行时长。已提交的协调器在独立 session 中；运输异常不能据此当作未启动。先查原远端 intent、PID/starttime、日志与 receipt，不能重复 enqueue。

root 需先冻结提交当前运输工具，再依次执行 `transfer`、`dryrun`、`enqueue` 三个模式。各模式有独立不可重用的本机 evidence 前缀；只有实际远端 dryrun 成功回执与 exit0 才允许 enqueue。此审查不等于三项已执行，实际正常 speed 完成、v27 准入、首 terrain actor 和质量仍待独立核验。

机器回执：`source030_transport_peer001.json`。当前审查只认证运输接线与既有守卫，不授机器人能力或实车资格。
