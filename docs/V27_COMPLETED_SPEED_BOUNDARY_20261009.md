# v27 普通大阶：正常 speed 终点接续

日期：2026-10-09。本文件描述隔离候选及 CPU 验证。未对 Kaiser 发信号，未部署 v27，未生成未来真实 speed 父检查点证书。

## 身份与继承

v27 的唯一入口是原 source029（`e5bd9c1367b875d7f229b9a1930647013d023372`）绑定下正常完成的 v26 speed。静态 `completed_speed_predecessor` 固定原 v26 plan/lineage SHA，动态 `revisions/v26_completed_speed_to_v27/entry.json` 认证实际 source029 retained job/receipt/previous-job、原 REC 入场证据、正常 speed full seal 和原账本。

原 REC origin、speed record 与检查点保持原身份，不重写成 v27。原消费、所有 attempt、学习谱系、stage、joint 与 recovery/speed component 时钟保留。v27 第一真实 worker 只能是 terrain，terrain component actor 从零开始，十次 critic 适应计入原消费；恢复 actor、critic、Adam、动作分布及 RNG 后继续学习。通用 `_joint` 身份认证保持严格；只有这个命名入口接受旧 speed 父。REC 重进、旧 terrain 已启动、未完成 speed、错源、同 clock 跨 attempt fork、退款和重复 owner 均拒绝。

普通大阶八行的实际引入时钟持久绑定在入场证书中。actor allowance 按原 terrain 和全局 paid 余量减去十次 critic，并取 endpoint 上限；不足 1,000 actor 时拒绝，不编造教学预算。标准 15,000 actor 下八行最迟 4,000 激活最终 200+250 mm；较短合法预算按原 core 缩放，后续 jump/stepjump 保留同一 terrain 引入源。普通上阶仍只有前进命令，不发送 JUMP/STEP 请求。

## 未来边界 owner 协议

原 v26 supervisor 会在 speed 结束后立即派旧 terrain，因此不能依赖慢轮询“恰好没有 owner”。新工具 `scripts/hold_completed_speed_boundary.py` 是明确 opt-in 的未来控制器：默认只有只读准入，`--apply` 才能产生信号。固定 run 级 controller flock 贯穿准入、等待、信号、wrapper 封存和原生闭合；一次性 intent 原子且排他发布。它不持有旧 supervisor 的 account flock。

1. 从 actual job 与 source029 receipt 认证 wrapper/supervisor、唯一原 speed child、pid/starttime/ppid/完整命令、worker admission SHA 和实际 actor/critic 请求。仅实际余量 20–100 actor 时进入。
2. 使用 pidfd 只向 supervisor 发 SIGSTOP；先 arm cleanup，再发送，确认 stopped 状态并重新认证同一 speed child。不给 worker 或进程组发信号。
3. speed child 继续到真实正常结束。只有 worker 已退出或 zombie、正常 completion/full seal 与 admitted actor/critic 数值一致才允许下一步；单有 completion 文件不足以发 TERM。
4. 对同一 supervisor pidfd 排队 SIGTERM，然后 SIGCONT。旧原生 handler 退出并完成计账；成功分支不会派旧 terrain。等待原 wrapper 实际完成封存。
5. 在原 account flock/no-live worker 下，以原 source029 Python/native `TaskCampaign` 重新打开旧账户。仅真实 normal complete full seal 已达到 actor endpoint 或 paid cap 时，用原 `run_integrated_domain(speed)` 的零 PPO 终局分支补闭合；`execute/train_command` 明确 fail-closed，禁止构建场景或启动任何 worker。账本与其他 domain 不变。
6. 此后才可通过新冻结 v27 wrapper/准入入口迁移并启动 terrain；工具自身不启动 v27。

超时、普通异常及 TERM/INT 取消时，在自定义 handler 仍有效期间 CONT 原 supervisor，恢复原 owner；错误恢复可能继续原 v26 后续课程，因此“零后续派课”仅是正常成功分支的结论。SIGKILL/主机断电不能保证 finally；持久 intent 保留 PID/starttime，必须人工/独立只读认证后恢复，不能重放一次性信号或宣称自动恢复。

## 小文件中断恢复

before_* 与 entry 在同目录临时文件完整写入、fsync 后，用 hard link 原子不覆盖发布，再 fsync 目录。已有目标必须逐字一致；错误旧证据不覆盖。半写失败不发布目标，残留未发布 pending 不能被当作证书。

全部入场认证在冻结 before_* 之前完成，拒绝不会污染合法重试。正式 account 仍复用原 native claim→consumption→campaign_state 原子单文件写入。仅初次迁移的 exact old state + exact new authoritative claim + unchanged old attempts 允许继续单向提交；不接受任意混合状态或一般放宽账本恢复。

## 验证边界

CPU 测试使用自建子进程和真实 native model/Adam 模板；不创建 Isaac 场景、不执行 PPO。控制器覆盖正常子进程完成后父终止/零后续派课、STOP 已送达后异常/TERM、超时及 validator 拒绝 finally CONT、PID 身份错误、同/不同 intent 的 run 互斥、证据半写与排他发布竞争。

完整生命周期覆盖实际 speed paid 数值 5,093 + 14,907（10 critic + 14,897 actor），以低于 15,000 的 actor 值正常触及 paid cap；原 e5bd9 `run_integrated_domain` AST 不变且实际零 PPO 终局闭合。其余合成 fixture 值是 CPU 测试数据，不是未来部署证书。测试还覆盖跨 source prefix 新 wrapper retained pair、两次 open、真实 terrain worker 准入、全学习状态恢复和三次 native 写入后故障重入。最终结果及冻结 SHA 以专用 CPU 回执为准。

最终 CPU 结果：`boundary-final002` 的原 13 项迁移/生命周期与 11 项控制器用例共 24 passed（947.47 秒）；随后按最终代码运行的 `boundary-final003` 为新增部分 critic 恢复及最终 15 项控制器/原子发布用例共 16 passed（180.42 秒）。两批有 11 个控制器用例重复，合计覆盖 29 个独立新增用例，不能按 40 项独立验证计数。已启动的旧 lineage/trainer/resize 兼容回归自然完成 68 passed（84.66 秒），之后没有扩大或重跑测试。CLI help 与 diff-check 通过。

保留的首批失败：`boundary-final001` 只有首个完整生命周期通过，后续三项因测试 fixture 已存在 symlink 发生 setup error，主动中断后为 1 passed / 3 errors / 275.82 秒；修复 fixture 后由上述 final002 复测，不能把首批错误记成通过。一次 `boundary-legacy004` 使用不存在的测试文件，pytest exit 4、没有执行用例；正确文件的 legacy005 是上述 68 项。全部原始日志与 SHA 保留在审计目录的 `completed_speed_boundary_cpu001/`。

947.47 秒是 CPU 测试整套墙钟，其中包含每个拒绝路径重复构建真实 canonical source graph、归档和学习状态模板；这不是每次 PPO 更新耗时。新增认证只位于 entry/source selection/admission/attach_runtime，不进入 physics 或 PPO 热循环；本轮没有做原生场景 startup 或性能实测。

## 未发生的事项

当前 source029 speed PPO 保持原合同和 owner。本候选没有新的行为成绩或机械/实车资格（qualification=null）。未来部署前仍需 root 将候选与正式静态 v27 authoring/源清单组合冻结，核验真实 normal speed 端点及停止封存协议，完成正常边界原生准入和首 terrain actor 读回。不存在可供本轮生成的未来 completed-speed 父值。
