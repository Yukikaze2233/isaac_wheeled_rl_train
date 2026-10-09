# 跳阶落台停止目标修复候选

此候选基于冻结源码 `81ad124554c4ab45cc274e2d482d46c76ba125b0`，仅做 CPU 验证。现有 Kaiser PPO、source030 排队源、合同、账本、旧模型均未修改；`qualification=null`，不表示跳阶已学会或原生碰撞通过。

## 已证实的数据流冲突

`StepJumpTask.control` 在观测到腾空后首次 IMU 支撑时，把落台前向命令设为零。其原物理完成判断要求平台上速度不超过 0.25 m/s。旧 `JumpLandingWindow.update` 却在接受起跳时锁定非零前向/转向命令，并在首次物理触轮后的有限窗口中继续围绕该值计费。因此同一跳阶任务的停止命令、物理完成条件与落地奖励目标相反。

使用原生 CPU observer/controller，起跳目标 0.4 m/s、转向目标 −0.2 rad/s 时，落台停止的前向成本为 −1.1612496/s、转向成本约 −0.241046/s；保持起跳速度/转向反而使对应成本为零。上述是该 observer 的每秒密度，策略步积分由现有调用方执行，不能当作总 reward 或行为完成率。

## 最小修复

新增顶层显式布尔字段 `step_jump_landing_stop_target`，默认 `False`。`ChassisEnv` 向 observer 传入实际 `StepJumpTask.enabled` 布尔掩码。仅新接受请求且属于跳阶任务的环境，在启用字段时锁定单独的落台前向/转向零目标。任务身份不由 mode、平台高度或当前 live command 推断，释放请求和后续命令变化不能重写已锁定目标。

原 `goal_commands` 仍保存起跳命令，空中接触前轮速目标仍由它计算。普通跳跃，包括混合 batch 中的非跳阶环境，保持原数值语义；位置成本仍使用原锁定命令与原平台规则。三秒触轮成本窗、六秒任务期限、起跳/收腿/接触时序、成功门槛、StepJumpTask、FSM、物理参数、PPO 和采样配额不变。

默认关闭时，`runtime_state` 及独立 `runtime_expectation` 均保留旧完整字典。启用时两端增加同一标记：

```json
{"step_jump_landing_stop_target":{"enabled":true,"source":"explicit_step_jump_enabled_at_accepted_request","landing_vx_yaw":"zero","precontact_wheel_vx":"latched_takeoff_command"}}
```

## 来源与预检声明

在新的 v27 来源图重生前，此修复仍为候选。authoring 只为 v27 `jump` 和 `stepjump` 派生 recipe 显式启用；当前 `terrain` 段尚无 jump_landing observer，不注入无效字段。`revised_design` 同时写入实际 train contract 字段，`jump_landing_revision.runtime_expectation` 声明独立运行时标记。以后若其他混课段也实例化该 observer 且包含跳阶行，需在该段合同、recipe 和 design 三处一致启用，不能仅按 stage 名判断任务身份。

v27 专属 `production_source_files` 另登记 `jump_landing.py` 与 `jump_landing_revision.py`，让细粒度 worker 源码摘要绑定实际奖励及其独立声明；旧原生 v12 的登记条件不会覆盖 integrated v27 schema，完整 archive 也不能替代这两项登记。

旧 v24/v25/v26 的 recipe、四域 worker/design 均须逐字节保持。现有 source030 的 v27 合同也不得热改；若父代理采纳候选，应生成新的显式来源包及对应哈希，保存旧 81ad 字节，通过正常未入场更换或未来边界接续认证后使用。此提交没有生成正式合同、来源 archive、准入 receipt 或虚构未来父检查点。

## CPU 验证范围

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  /home/yukikaze/isaacsim60-venv/bin/python -B -m pytest -q \
  tests/test_stepjump_landing_stop_target.py \
  tests/test_integrated_terrain_authoring.py::test_old_recipe_bytes_remain_exact
```

覆盖正反向混合任务成本反例、普通跳跃精确兼容、接触前轮速精确兼容、释放/重复/重置事件归属、真实 StepJumpTask IMU 控制与物理 assessor 停稳条件、布尔/mask 负例，以及真实临时 recipe→design→canonical worker→有效配置→独立设计预检。删除/关闭已声明字段、只改 recipe 而缺少独立运行时声明都须拒绝。临时物化不改变原 plan/lineage/migration 字节，不认证未来边界或实际原生场景。

仍待：正式来源图采纳、原生 runtime 读回、真实触轮场景、固定跳阶整链与平地跳跃回放、实际 PPO 曝光及完成率。CPU 成本修复证明目标一致，不证明策略能力。
