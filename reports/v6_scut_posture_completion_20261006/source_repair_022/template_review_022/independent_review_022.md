# source022 部署模板独立窄审

本轮只审本机 emit-only/read-only 模板。未 SSH、发送信号、启动 native、GPU 或 PPO。新 source022 metadata/admission 尚未物化，本结果不授真实停止或部署准入。

已保留四份原模板到 `template_review_before_001/`，修正 root 授权的确定问题：

- `deploy_script_repair_022.py` 的标题改为 v19→v20，会话从误留的 r21 改为 r22；发射 launch 时追加 predecessor 的 022 schema、旧 source021、critic10、无 reopening/qualification 检查。
- 首 PPO 模板移除字符串替换后仍残留的旧 021 现场 metadata/predecessor/revision 以及顶层 `print(inspect(...))`。现在是 import-safe 的真实发射器，显式参数读取新准备回执；不导入即检查远端文件。
- 首 PPO runtime 同时核四项真实 script binding：原生恢复源码 SHA `04ff676…`、显式 SIDE=1.5、既有 gyro=.75、CAPTURE 的真实源码谓词说明。核新源码文件、REC 配置、startup、runtime observed 和 design_expected，不只相信版本标签。新 recipe 文件纳入 runtime source pins。

stop/progress observer 字节未修改。stop 仍仅在 emitted code 中核精确旧源、三个 owner `3506636→3506645→3506652`、attempt005 正式 actor 进度后记录 fsync 的请求，并只向 supervisor 3506645 发送 SIGTERM。发射器本身没有 kill/SSH；本轮只解析、编译该 code，未执行。progress observer 仍仅匹配新 source-root 的训练入口、拒绝重复 invocation，原字节捕获 1..11，critic_phase/actor11 由独立 progress 与不可变 seal 验证。

旧 revision schema 核为 v19，新 revision 为 v20。prepare 在旧 owner 消失、campaign stopped 后动态选择各域最新完整 seal；恢复父还须来自 attempt005 final、真实 Adam/std/RNG/clock 完整。原五个域父与来源、全部旧合同字节、全账本/claim 和 121000/20k 上限继续核验；新合同继承十次 critic 全计消费、不前进 actor，同版本 remaining 原逻辑保留。launch 使用非阻塞锁、唯一新 receipt/log 和 tmux 拒绝条件，不自动重试。

最终 21 CPU fixtures 通过，0.73 s、exit0：import-safe、prepare/launch 仅编译、十二种错误身份/准入拒绝、十 critic＋首 actor/Adam/seal 有界证据、progress10/11、原全账本无退款、当前 stop 身份、真实脚本绑定、重复启动拒绝，以及第一 PPO 动态 CLI 发射且 `inspect` 未调用。这些为合成 CPU 输入，不是现场父验证或部署结果。

尚需 root 以最终源码/合同/recipe/native 结果生成并认证真实 metadata、admission 与每个 evidence SHA，再物化 prepare/predecessor/revision，发射并核真实 argv。当前证据映射由已审 caller 提供，模板不逐一重读所有本机 evidence 路径；不能用本文替代该实际认证。新 v20 recipe 的 `validate_source_revision`、30 角色和完整源集成也等待最终源验证。运行中 claim/campaign 的并发快照不一致应保留失败原件，再取新只读回执，不能重启/退款或改判旧回执通过。same-v20 的 inspector 支持原 saved remaining，但 observer 对同 source-root 多次 invocation 会保守拒绝，不能宣称已解决通用重启调度。
