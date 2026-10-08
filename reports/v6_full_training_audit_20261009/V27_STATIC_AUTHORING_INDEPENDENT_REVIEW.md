# v27 静态课程合成独立审查

审查对象：`cda955efcc6b6686a64712ba9a54873eec6f3dfb`，工作树 `.worktrees/audit-logger-timing-20261009`；对照 `f99d3cbb556db873a299b8c9bc2cd68fcf34957a`。范围是 ordinary terrain 静态 authoring、未来三域恢复诊断开关和 source inventory。未重审 publisher 实现；未修改候选、执行 SSH、启动场景或 PPO。

结论：静态合成与诊断接线未发现语义阻断，存在一项逐文件来源摘要遗漏。该结论不代表 v27 已完成运行准入或部署。

## F1：v27 逐文件来源清单遗漏 step_climb.py

`integrated_production.production_source_files()` 的 v27 分支已包含 `ordinary_large_stair.py`、`state_snapshot.py`、旧 recipe 和新静态工件，但没有 `src/wheeled_tasks/chassis/step_climb.py`。`train_chassis.py:465–575` 的条件来源清单也没有该文件；实际 `integrated_runtime._references()` 从原生来源收集到的唯一 Python 文件是 `recovery.py`。

新增课程依赖 `StepClimbTask.assessing` 来关闭未激活阶段的 climb reward/outcome，并在激活前保留真实起点。这部分实现应进入实际 worker 的 `source_hashes`，否则单凭 worker 的逐文件摘要不能完整复核这次普通阶语义。完整冻结 Git 提交和 archive 仍包含该文件，因此这里不是“源码遗失”或未固定源码的结论。建议仅在 v27 inventory 补入 `step_climb.py`，并补 membership 断言；不改变旧版本的配方。

## 已核对的行为

- **旧版本和 speed**：版本分支只在 v27 非 speed 时启用新的 terrain transform 和诊断开关。24/25/26 的默认采样路径未改变；现有测试逐字节覆盖全部旧 recipe/design/worker。额外只读 CPU 断言确认 v27 speed domain 的序列化字节与 v26 相同。这里的 speed 保持指 domain 配方；新 v27 worker 的版本元数据仍会有意变化。
- **来源与分配**：新 8 行显式为 `derived_row`，`source_name=null`，有 derived_name、spec/terrain SHA 和 transform pin；原 v23 的 source_coverage 仍是 101 行，不把新增行冒充为旧源。重加权行保留 immutable_source_fraction。合成继续使用原整数分配器，最终 terrain/jump/stepjump 为 241/256/270 组、4096 lanes；新行各有非零分配。配额的 10% 是原生 terrain 池内份额，不是整个混合 batch 的 10%。
- **命令所有权**：8 行的 owner 均为 terrain；`ordinary_body_climb` 保留独立命令生成，未进入 SCUT planar/heading 映射。原 8 小阶继续显式绑定。已有 `validate_row` 拒绝第二命令拥有者和跳跃请求。该层没有改物理状态或发 JUMP。
- **真实时钟边界**：静态 0/15000 被明确标为 `schema_only_actual_admission_parameters_required`，recipe 的 freeze_ready=false、qualification=null，pending 包含 completed v26 speed 的实际父模型、terrain 起点及剩余 actor allowance。通用 attach_runtime 仍不能被当成这项重绑定已完成；独立实际边界准入必须绑定并在后续域继承，不能重新从 REC 入场。
- **诊断接线**：未来 terrain/jump/stepjump 的实际 patch 为 `recovery_phase_audit=true`，record_diagnostics=true，auto_reset 默认为 true；speed 不启用。训练仍通过原 LearningAudit 构造和 observe/drain 路径。observe 收到 env.step 返回的最终 policy_learning_mask（含 nonstandard start 合并）；恢复诊断快照在 reset 前克隆，既有终止分母仍来自 RecoveryOutcomes。新增计数只修改自己的 buffer，不回写 PPO mask、FSM 或 reward；33 项是按实际出生级别的全局统计，不是每组复制，也不是四种纯倒地整链成功率。
- **其余来源摘要**：learning_audit.py 已由所有 task_campaign 的既有 source_files 分支登记；state_snapshot.py 既有 publish_state 条件登记，也在 v27 production inventory 登记；ordinary transform 字节进入 recipe fingerprint，缓存不能掩盖 transform 修改。

## 本次验证与限制

主任务已报告组合候选 148 个 CPU case 通过，本审查未重复整套。额外执行一项只读 CPU 集成探针：从实际 v27 recipe patch 构造 LearningAudit（mock 仅限 PolicyUpdateProbe 和无仿真环境壳），确认 speed 为 0 个新增 tag，三个未来域各 33 个；三域 `auto_reset=false` 均明确拒绝。同时检查 24 个 heading 行均不包含新增 8 行、行校验及静态时钟标签。耗时约 6.79 秒，通过。

这不是 Isaac 启动、运行时 source receipt 认证、真实父模型 full-state 续训或台阶能力验证。没有产生正式 v27 campaign/lineage/migration，没有新增训练步，也不声称已部署。
