# 跳阶落台停车目标：下一冻结源的原生采纳路径

本报告先只读审查冻结 `81ad124554c4ab45cc274e2d482d46c76ba125b0` 及独立修复候选的接线，随后依 root 明确授权，仅在独立候选树生成并提交新的静态来源图。没有 SSH、信号、PPO、物理回放或重复 CPU 测试；当前 source029 的高速 PPO 与 source030 等待队列均未修改。修复只消除已证实的目标冲突，不能据此授予跳阶能力或实车资格。

## 最小选择：入场前替代候选，继续使用 v27

新 source031 可以复用 `integrated_production.write_bundle(..., version=27)`、`TaskCampaign`、`integrated_completed_speed_revision.open_account` 和原 wrapper，不需要 v28 枚举或新 runner。保留原父 `source028-joint-serial-v1` 的真实 REC 与正常 completed-speed 来源；新目标仍使用既有 `source030-joint-ordinary-stair-v1` 名称，实际身份由新 plan、lineage、worker、源码和归档 SHA 联合确定。source031 是新的源发布编号，不是额外训练阶段。

适用范围必须限制在旧 81ad v27 **尚未入场**：不得已有 `train/revisions/v26_completed_speed_to_v27/entry.json`、旧 v27 账户转换、terrain/jump/stepjump 的新谱系选源或 worker。原准入已提供拒绝：账户 `contract_sha256` 只能是精确旧 v26 或本候选新 plan；完成父证书的 `new_lineage_manifest_sha256` 必须等于本候选 binding。旧 81ad 一旦入场，不能删除证书或伪装成未入场重新走本路径。

外部冻结采纳回执应明确 `supersedes_before_entry`，记录被替代的 81ad 源与其四个身份：

| 旧 v27 对象 | SHA-256 |
| --- | --- |
| campaign plan | `7980623dc024ff526bdd91e9bb7a50cd835220712c2359744561671bb4510f45` |
| lineage manifest | `3caf32164c15ff9cd29e791a3bef631087e81aa010c754f12fa5ea3baf75dabe` |
| runtime recipe | `b8334ff9756daeb5b7af2afae42231e747b1d30c3207fcdf0f69d657a8eb9ba4` |
| migration manifest | `723fb93914e3709394c92c315c95e4cd9e9121295a8862edb47054474b3a8341` |

该声明放在新候选、归档和一次入队回执中；不要给闭集 canonical 合同偷加未知字段。旧源、旧合同、旧 stage 与旧 intent 原字节保留。

## 必须完成的三处原生配置接线

候选采用显式 `step_jump_landing_stop_target: true`，构造器默认 `False` 且要求真布尔值。

1. `_training_recipe` 仅新 v27 的 jump/stepjump 训练 patch 打开开关；speed 整个 recipe 继续逐值继承 v26，terrain 没有该 landing window，不需要额外字段。
2. `revised_design` 将有效配置中的开关复制到 `train.contract`，并由 `jump_landing_revision.runtime_expectation` 得到与真实 `JumpLandingWindow.runtime_state()` 完全一致的开启标记。只加 reward 代码或只加 recipe，都会遗漏原生 preflight 宣告。
3. `env` 把真实、固定的 `self.step_jump.enabled` 传到窗口。在真实 accepted request 锁存时，仅这些 lane 的独立落台 vx/yaw 目标为零；原 `goal_commands` 继续供空中/接触前轮速匹配与原位移掩码使用。普通跳跃、原有限窗口、接触时钟和物理成功阈值保留。

另有随本修复闭合的来源登记：原 `train_chassis.py` 只在 native v12 schema 下登记 `jump_landing.py` 与 `jump_landing_revision.py`，实际 integrated v27 worker 使用另一 schema；81ad 的 `production_source_files` 也未登记二者。最终候选已在 v27 专属 inventory 精确加入这两个实际变更模块及 membership 断言，保证 worker 的 source hashes 认证新 reward 和启动预期。完整归档认证不能代替 checkpoint 的细粒度来源登记。

默认关闭或字段不存在时，旧 runtime 宣告不得新增标记。v24–26 的全部 36 份正式 recipe/design/worker/plan/lineage/migration 必须逐 SHA 保持；还应对源029全部旧 contracts 做完整字节核对，36 是专项集合而不是旧合同总数。

## 本地生成与验证顺序

在新的独立候选树合入已验证的最小代码修复，冻结代码父，然后复用原生成入口：

```python
production.write_bundle(
    candidate_tree,
    source_root="/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_serial_dispatch031_20261009",
    run_root="/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train",
    version=27,
)
```

这只生成新的 12 份静态 v27 对象。现 `generate_source030_contracts.py` 硬编码旧 f627 代码父、旧树、source030 和旧回执路径，不能原样再次运行；可复制其小型实例化脚本，替换这些明确输入，加入 jump/stepjump 开关、runtime、旧 36 SHA 与语义闭集断言。逐一调用现 `validate_plan`、`validate_static_worker`、`static_training_config`，并用真实 materialize/preflight 核对 train 配置和 runtime；错误类型、缺失宣告、ordinary jump 默认兼容必须有 focused 反例。

预算 121000、4096、35D/81D、50/200 Hz、160/2.5、64/32、3 mm、1 s、机械与 DR、原普通大阶/小阶课程及复习不因该开关变化。静态 ordinary origin 0/allowance15000 仍仅 schema 占位。实际 remaining allowance、terrain origin、完整 actor/critic/Adam/std/RNG 与所有已付费时钟，只能在正常 speed full seal 到来后由原证书和准入绑定；不能现在写未来 checkpoint 或首次 actor 值。

## 等待与部署的安全复用边界

source030 部署 helper 的原 API 可以复用，但其 `source030_deploy` 目录和产物前缀固定且已存在，原样调用会拒绝覆盖。下一实例需独立 `source031_deploy` 目录、归档、stage、日志、receipt 和 once intent；只修改实例路径及冻结身份，保留原 `load_native`、archive/permission 校验、账户锁、真实 `make_admission`/认证与 wrapper 调用。其旧合同比较对象仍为真实 source029/v26；新归档只有 12 份新增 v27，旧合同保持。

必须先完成新候选、完整合同和归档验证，再由 root 操作等待进程替换。不能只停止外层 coordinator：其 `Popen.wait` 没有杀 child 的语义，原 controller 可能继续。应认证旧 coordinator/controller 的 PID/starttime、持久 intent 与 phase；通过原 controller 的取消清理协议退出等待，确认两者真实退出、queue/controller 锁释放、无 boundary/prepare/launch 已发生，保留全部旧日志与异常状态。这个步骤只替换 observer，不提前停止当前 speed PPO。若实际已接近暂停父进程的边界，应先解析原持久状态，不能假定仍是只读等待。

随后新源执行独立 static-stage → 30 角色 CPU static-preflight → 新 once 队列。有限等待继续使用原 controller，在 speed 自然完整结束后，实际 receipt SHA 顺序驱动 boundary-prepare → 原 wrapper launch。旧 once intent 不能重用、热改或删除；不得同时运行两套协调器。新队列与发布回执需重新钉住新 commit/archive/stage/helper/脚本 SHA，并保存明确 supersedes 证据。

## 本轮实际完成的本地冻结

修复代码冻结 `0f4b5f673e5f7e2cb382a8fcbd5766b0e13d8a9b`，父 81ad，恰好 6 个 owned 文件。A 实际执行的 focused CPU 为 17 passed / 31.94 s；最后来源登记修改后，同一真实 materialization 正负例再过 1 passed / 23.97 s。后一次是复核，不应算成 18 个不同测试；B 没有重复执行这些测试。

`generate_source031_contracts.py` 已在此代码父上真实执行，exit 0，生成 12 个 v27 对象，其中 11 个有字节变化，speed expectations 逐字节保留。276 份旧合同、其中 36 份 v24–26 专项集合全部保持。jump/stepjump 的有效配置、train contract、runtime 预期与 source inventory 接线均通过生成器的实际原生 canonical/static 校验；speed recipe 精确继承 v26，其余课程仅增加该明确开关。静态图精确提交为 `ef2cf6cd9aceeb74b6a5fc6e7dca7412a7a23f4a`，独立树干净。

生成回执 `source031_contract_generation001.json` SHA `b3d938cf145fff595a4999d25a906e41380866c1a834ae08161bd54e569f0eb1`。生成时源码准确是 0f4，最终提交 ef2 只增加上述合同 diff，不能把生成执行源码记成后来的 ef2。当前仅本地静态生成成功；旧队列取消、新 stage/queue、实际 completed-speed 准入、原生场景读回与新 PPO 均未发生于 B 本轮。
