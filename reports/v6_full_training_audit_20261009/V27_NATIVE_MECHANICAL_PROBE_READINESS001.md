# v27 原生机械探针准备：尚未执行

只读核对实际冻结源 `81ad124554c4ab45cc274e2d482d46c76ba125b0`。没有启动 Isaac、GPU 场景、SSH 或 PPO，没有修改生产配置。完整文件 SHA、命令参数模板和资源快照见同目录 `v27_native_mechanical_probe_readiness001.json`；资格仍为 null。

现成的正确入口是 `scripts/train_chassis.py --validate-scene 1`。它保留原 full 配置预检、joint 准入和运行时读回；执行4096环境的一策略步／四物理步零动作，不构造 PPO runner。输出是 `startup.json` 与 **`completion.json`**，不是旧边界适配器假定的 `report.json`。只检查启动／物理容量，不能证明爬阶成功。

v27 入口必须等真实 **正常完成的 v26 speed** 全状态父和 completed-speed→terrain 准入。即将产生的消费42000／task actor597是未完成 speed 的周期封存，仅用于均值回放，不能拿它或 REC41393充当 v27 入场父。以下命令是待真实准入后填值的模板，当前未提交；路径和所有时钟必须来自实际 admission／父 seal，不能猜测、另造未来父或改 actor 时钟：

```sh
"${AUTHENTICATED_PYTHON}" -B \
  "${ADMITTED_SOURCE_ROOT}/scripts/train_chassis.py" \
  --contract "${ADMITTED_SOURCE_ROOT}/contracts/v6_task_campaign_terrain_worker_v27.json" \
  --stage terrain --research --run-dir "${EXCLUSIVE_SCENE_OUTPUT}" \
  --device cuda:0 --num-envs 4096 --updates 1 --validate-scene 1 \
  --level 1 --seed 290927 --block-index 0 \
  --resume "${REAL_COMPLETED_SPEED_PARENT}" --resume-task-change \
  --learning-rate-scale 1 --global-actor-update-offset 0 \
  --consumed-updates "${ADMISSION_CONSUMED_UPDATES}" \
  --actor-updates-consumed "${ADMISSION_ACTOR_UPDATES_CONSUMED}" \
  --stage-actor-update "${REAL_PARENT_STAGE_ACTOR_UPDATE}" \
  --task-actor-update "${ADMISSION_FRESH_DOMAIN_ACTOR_UPDATES}" \
  --integrated-source-selection-file "${REAL_ADMISSION_FILE}" \
  --integrated-source-selection-sha256 "${REAL_ADMISSION_FILE_SHA256}" \
  --max-runtime-seconds 600
```

原 `probe_v6_step_runtime.py` 只支持旧 S5D、20k环境、1kHz；不能直接将 `--plan` 换成 v27。它的几何检查读的是 Python `Surface`，接触方向检查可复用，但不独立证明 USD 碰撞体几何。`prepare_chassis_probe.py` 会准备6次 PPO，排除于本次零 PPO 任务。

唯一无需未来父、已有可执行原生路径的是 `probe_v6_base_contact.py`：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/yukikaze/isaacsim60-venv/bin/python -B \
  /home/yukikaze/Documents/workspace/robot_rl/.worktrees/audit-logger-timing-20261009/scripts/probe_v6_base_contact.py \
  --output /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_full_training_audit_20261009/v27_contact_adapter_native001 \
  --physics-steps 250
```

此命令也未执行。它只测5kg方块落在200mm平台上的真实非零接触、力法向符号和分片载荷分类，时钟为1kHz；**不是机器人、200Hz大阶、小楼梯或两级爬阶验证**。

最小缺项是在正确准入的现有 scene 路径中保留有界读回：普通8行的实际 mode0、仅速度命令、manual STEP/JUMP请求为零；实际 USD collider 的位置／尺寸／碰撞属性及支撑 view；非零 `base_link` 接触点和法向力与 native matrix、已有全机体接触量交叉核对。混合场景本身含恢复等任务，所以“无 JUMP”只能对普通8行断言，不能全场景断言。空接触必须写 `not_observed`。

静态 recipe 已包含新8行，小阶5／10／20／40mm上下三阶的原生42%池保留，integrated terrain 中约24.923%；新普通大阶约5.934%。两级 **200+250mm是累计台面200／450mm，踏面0.8／0.6m**，末端实际路面延伸至6.2m。以上都是来源定义，实际 collider／支撑尚未原生核验。真实 terrain 入场时普通大阶处于 approach-pad preteach；不能人为将时钟改到3800／4000、把预教支撑计成正式爬阶。正向爬阶接触须在真实 actor 曝光后观察，或另有明确冻结的工程场景；当前不存在能直接覆盖它的完整 v27 探针 CLI。

本机另一任务的 GUI PID32672在运行，约8.27GiB RSS、2240MiB显存；RTX4060Laptop共8188MiB，读回总占2340MiB／利用45%。仅查询，未 signal。剩余显存不保证4096闭链场景容量；未来实际启动须先协调资源，不能停止他人的 GUI。
