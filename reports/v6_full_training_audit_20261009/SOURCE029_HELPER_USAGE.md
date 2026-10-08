# source029 helper 调用

`deploy_source029.py` 已通过11项独占发布/安全归档守卫和3项真实canonical/native CPU入场验证。实际验证调用原生 `prepare_admission`、`make_admission`、`authenticate_receipt` 与修复后的 `run_domain`，41393/41223/REC14188/lineage27646及14907预算按fixture实际重算；仅模型SHA替换为明确标注的合成CPU full-state SHA。没有App/GPU/SSH/PPO。回执：`source029_helper_cpu_verification001.json`。

helper SHA-256：`3864f047ffea30989cce30db1bd697c20d5f217d70286d09c77c12198c05b963`。发布前逐字节复制并核对该SHA；stage后任何helper变动都会被拒绝。测试产物不授部署或技能资格。

由root先复制**已冻结source029**的git archive到实验根目录，同时将该helper复制到独立部署工具位置。不要执行本地helper来操作不存在的远端路径；该脚本没有SSH功能。新archive的PAX comment必须等于最终40位commit，所有v26及其它contracts与source028字节一致。

在Kaiser原Isaac Python及原runtime环境中，每步沿用完全相同的下列参数。只有 `--mode` 依次变化；`COMMIT40` 用最终已冻结commit填入，不可使用旧source028 commit。

```text
/home/kaiser/robot-rl-sim60/env/bin/python -B /ABSOLUTE/COPIED/deploy_source029.py
  --mode stage
  --run-root /home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358
  --new-source-root /home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_serial_dispatch029_20261009
  --source-archive /home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_serial_dispatch029_20261009.tar.gz
  --source-commit COMMIT40
```

1. `stage`：无owner、旧wrapper归档ready及latest REC41393精确seal检查；安全解压新独立source，冻结 `source_revision_029.json`，发布 `source029_deploy/stage.json`。不动训练账本、不创建场景。
2. `preflight`：逐文件和所有contract核对；真实CPU派课只接受speed→terrain→jump→stepjump；运行新 `run_task_campaign --preflight-only` 与新 `repair_known_integrated_origin --check-only`。这是CPU校验，不运行native场景。
3. `correct`：单次调用新冻结 `repair_known_integrated_origin.py --apply`，显式更正误跑19更新后的当前入场来源，同时保留旧origin快照、actor/critic/Adam/RNG和所有消费。此前preflight必须已通过。工具遇到旧intent或残留产物拒绝重复，先核查即可。
4. `launch`：真实CPU再读更正后的latest origin/完整账本，验证allowance为14907=10critic+14897actor；新独占launch intent，然后唯一调用既有 `chassis_remote_job --resume-existing --source-revision-receipt source_revision_029.json`。先释放account lock再启动supervisor，正常worker保留自己的原生startup。

各步都保存独立日志/回执；已有产物不会覆盖或隐式重试。旧boundary queue的v26 intent和原source028原件全部保留，不能用于029重启。029初次真实worker必须位于speed attempt且带integrated source-selection file/SHA，父模型为REC41393/e04c…3967；另用 `verify_first_actor029.remote.py` 核验新wrapper真实父子链、native startup、firstcritic41394及firstactor41404。launch提交成功本身不代表首actor已验证。

本次去掉的是重复建场预检，实际speed训练worker仍保留正常原生建场、物理断言与full-state恢复验证；不放宽机械判据，不加能力gate，不退费，不重放历史PPO。若actual seal/counters变化，固定helper会拒绝，应先解释变化并冻结新可核验pins，不能手改当前账本让它通过。
