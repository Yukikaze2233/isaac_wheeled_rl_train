# source030 v27 正式静态合同独立审查

结论：静态审查通过，绑定最终冻结 `81ad124554c4ab45cc274e2d482d46c76ba125b0`；生成及 root 关键6项 CPU 执行时的代码仍为 `f627f678fae98d374c049d76a444430dda70306a`。本审查只读 JSON、代码和 Git blobs，没有重复 CPU 测试、SSH、GPU/场景或部署。

12个新文件全部与生成回执 SHA 一致；36个 v24–26 原文件同时与回执及真实 source029 `e5bd9c1367b875d7f229b9a1930647013d023372` 的 Git 字节一致。新 plan/manifest/lineage 明确指向 source030 的命名 v27 身份，predecessor 单独钉住原 v26 plan/lineage 和 source029 commit；旧 REC/已完成 speed 的身份并未在静态合同里伪写成新的父。已核对 54 个可本地读取的文件引用 SHA，无不匹配。speed 的 runtime recipe 逐值等于 v26；四 worker 的顶层差异只有 `integrated_production`、`task_campaign` 和 `design_preflight` 身份/引用，物理及原生基线没有借此次生成改变。

预算保持121000，4096环境、24 rollout steps、单帧35D actor/81D critic、50Hz策略/200Hz物理反馈、PD160/2.5、solver64/32、3mm闭链判据和连续1s接稳保持。`automatic_promotion=false` 是验收不自动赋予资格；执行 schedule 仍按时钟正常接续，不要求45项齐过。

普通上阶与跳阶保持不同命令/任务所有权：普通8行是 `body_climb`、mode0、无JUMP/STEP请求，几何从50/100/150/200mm单阶到四档双阶；最终两次升高200+250mm、首踏面800mm、顶面450mm。小阶8行仍为5/10/20/40mm上下阶，SCUT绑定仅小阶，普通8行不在24个heading映射组里。未激活大阶仍用0.1m/s前进/反向/停车预教，专用assessment先在真实物理transition起点arm；inactive不付大阶专用奖励/不记失败，普通PPO保持。无状态瞬移。

源码terrain源池的大阶10%来自rough/slope的surface_traversal10%→2%、surface_height_joint4%→2%，小阶42%保持；经过joint组合与静止10%floor之后，大阶实际总份额为terrain 5.934%、jump 3.551%、stepjump 2.826%，不能宣传为每个联合域10%。两种大阶几何保留复习；原地/行进跳分别由jump组件负责，单/双台阶跳由stepjump组件负责，互不冒充普通上阶成绩。

普通8行静态0/15000仅是 schema 占位，源 transform 显式标为 `schema_only_actual_admission_parameters_required`；真正 introduction 必须由未来正常 completed-speed 证书按付费剩余预算认证并runtime重绑，后续jump/stepjump沿用terrain组件时钟。标准预算最终3800→4000激活，合法短预算按core缩放且保留至少65% full难度学习；本轮没有实际父检查点或接续证书。

`recovery_phase_audit=true` 只出现在未来terrain/jump/stepjump patch，speed保持原recipe，36旧合同不变。LearningAudit消费原native `policy_learning_mask`、产生诊断，不改变恢复动作或PPOmask。完整来源清单包括新增边界/module、step_climb、publisher/state_snapshot和learning_audit依赖。

静态文件仍 `freeze_ready=false`、qualification=null，等待实际正常边界 owner封存、入场/source认证、原生startup和首terrain actor读回。这份结论不表示普通大阶、跳跃/跳阶或整链已训练通过。逐文件SHA及域摘要见 `source030_contract_independent_review001.json`。
