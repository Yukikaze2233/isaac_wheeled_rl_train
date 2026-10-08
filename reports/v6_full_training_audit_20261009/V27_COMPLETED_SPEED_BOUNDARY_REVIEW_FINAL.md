# v26 speed → v27 terrain：最终独立审查

审查结论：**认可本地冻结候选，当前审查范围内无未闭合问题；不代表已经部署或获得技能资格。**

审查对象为 `b26a3bd461c0fad76e0681916d727639b840a5b5`，父提交 `bc99f12d2fd802117115fe2ca082e8c513f6c256`。独立逐项比对全部 12 个 Git blob、作者文件 SHA 与干净工作树，核验作者归档日志 SHA，并用 AST 确认 `run_integrated_domain` 与实际 source029 `e5bd9c1367b875d7f229b9a1930647013d023372` 相同。未重复运行测试，未执行 SSH、Kaiser 信号、Isaac 场景或 PPO。

机器回执为 `v27_completed_speed_boundary_independent_review001.json`。本文接续 `V27_COMPLETED_SPEED_BOUNDARY_REVIEW_INITIAL.md`，保留首轮问题记录，不覆盖它。

## 正常边界与学习状态

入口限定为经过认证的 source029/v26 正常 speed 终局。证书绑定原 v26 plan、lineage、原 REC 来源及 correction、source029 wrapper/receipt/previous-job、完整 speed 学习状态和原消费账本。旧 speed record 与 seal 保持原身份，不能改名为 v27。新 v27 dispatcher 跳过已经完成的 speed，首个新 worker 为 terrain，随后正常串行到 jump、stepjump。

CPU 生命周期使用真实预算形状：speed 已消费 5,093，剩余 14,907 = 10 critic + 14,897 actor。因此 **14,897 actor 是合法 charged-cap 终点**，不能强制补至 15,000，也不能退款。测试从“已完成并计账，但只保留旧 preterminal speed record”开始，通过原生终局分支零 PPO 补写正常记录，再完成新 wrapper、准入和 actor/critic/Adam 完整恢复。

原消费、attempt、学习谱系和 recovery/speed component 时钟继承；新 terrain actor 从 0 开始，先计费 10 次 critic。partial-critic 测试在已完成 3 次后重入，只剩 7 次，actor 消费不增，课程起点不变。八行普通大阶的 origin 和 allowance 从实际边界证书计算，后续域复用同一 introduction；不新增 JUMP/STEP 请求，不改旧运行合同。

## 首轮问题的闭合

| 问题 | 冻结修复与核验 |
| --- | --- |
| 拒绝准入先冻结错误 before 文件，污染合法重试 | 先以权威文件完成内容和 SHA 认证，再冻结；负例确认拒绝不创建目录，恢复原输入后可准入。 |
| 只看 record 指向父模型，漏同钟 fork | 使用原生完整候选扫描；跨 attempt 同钟 fork 拒绝，正常原生副本规则保持。 |
| 原 supervisor 会立即派旧 terrain | 新 opt-in controller 提前暂停 supervisor，worker 自然完成后再让原 supervisor 计账退出；不靠慢轮询抢空档。 |
| STOP 已送达但异常发生在 cleanup arm 前 | 先 arm 再 STOP；TERM/INT handler 延续到 finally CONT 和关闭 pidfd 完成。真实自建 CPU 子进程覆盖送达后异常与取消。 |
| 不同 intent 可能启动两个 controller | 固定 run flock 覆盖全流程，intent 排他原子发布；同名、异名并发均拒绝第二个调用。 |
| 成功 double-open 不等于三写中断恢复 | claim、consumption、state 每次写完后分别故障注入；仅 exact frozen old state + exact new claim + 原 attempts 可单向完成，伪造身份拒绝。 |
| 边界认证动态依赖未入 source inventory | v27 单独补入 `scripts/chassis_remote_job.py` 与旧 speed contract，并有 membership 检查。 |
| v27 budget schema / partial critic | v27 登记旧预算校验路径；最终补充用例实际验证 terrain/jump/stepjump 预算及剩余 critic。 |

before/entry 采用同目录临时文件完整写入和 fsync，再原子不覆盖发布；已有不同内容不会被替换。半写和发布竞争负例保留。

## Controller 的实际边界

监督进程、唯一 speed worker 和 wrapper 均认证 PID、starttime、完整命令及父子关系。pidfd 防止信号误送复用 PID。只向 supervisor 发 STOP、TERM、CONT，不向 worker 或进程组发信号。worker stdout 原本直接写日志文件，progress 自行落盘，不依赖被暂停的 supervisor 读取管道或释放 account flock。

只有真实 worker 退出或成为 zombie，且正常 completion/full seal 与原准入 actor/critic 份额吻合，才排队 TERM 并 CONT。原 handler 看见 child 已退出，不中断学习；成功分支不派旧 terrain。旧 wrapper 封存后，使用原 source029 原生方法、原 account flock、no-live 检查补闭合记录，`train_command` 与 `execute` 均临时 fail-closed，禁止创建 worker。该工具自身不启动 v27。

普通异常、超时及 TERM/INT 取消会恢复原 supervisor，**错误恢复可能继续旧 v26 后续课程**；“零后续派课”仅属于成功分支。SIGKILL、主机丢失不能保证 finally，必须保留持久 intent 并重新认证后恢复，不能自动重放含糊的信号。

## 验证口径与剩余工作

作者归档结果 `V27_COMPLETED_SPEED_BOUNDARY_RESULT001.json` 已独立核验 SHA。`final002` 为 24 项通过／947.47 秒；`final003` 为 16 项通过／180.42 秒，其中 11 项重叠，合计 **29 个新增独立用例**。另有旧兼容 68 项／84.66 秒自然完成。早先 fixture 软链接错误和 collection-error 日志均保留，没有当作通过。审查者未复跑这些测试。

这是本地 CPU 和自建子进程证据，不是原生训练成绩或 GPU 性能测试。完整来源认证仍有启动开销，但未发现它进入 physics/PPO 热环；本轮未为它增加 live-account 缓存。

未来还需 root 组合冻结当前 audit/publisher/source inventory 修复，使用真实正常 speed 父 full seal 执行边界准入，并核验首个 terrain actor。root 组合树已有的 `recovery_phase_audit`、`state_snapshot`、`step_climb` inventory 差异不属于本候选遗漏，组合时必须保留。当前没有未来父证书、部署回执、首 terrain actor 或新增技能资格，`qualification=null`。
