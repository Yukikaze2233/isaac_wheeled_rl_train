付款 observer／driver 已冻结，尚未运行原生观察。26 新 CPU fixture＋15 原 wrapper 回归共41通过（.75s）；最终字节的两臂真实 source/PT CPU preflight 都是 `preflight_passed_scene_not_run`，545源码／资产复哈希一致。CPU回执见 `prefix_native_payments_019.cpu.json`（SHA `48cc4d89ec2da7a4ceebb284c996d07b722e7e3d159a3b075266399656f7f2db`）。

| 固定输入 | 身份 |
|---|---|
| 原生源 | `712d701473f38ef6797725f8efdda9e6beb85e87`；`/tmp/scut-ordinary-alias-019-oit43rqd` |
| 460文件源manifest | `efc4cf55f98ee1c4efcda9b6fd9193a9170762fb589f7611a5b01aab2bc7c268` |
| seal2579／update29500 PT | `7f91d355d1e033891d895661dd63365820eb8553728a878c226667fe374dbcca` |
| 原wrapper | `55f4511284ae09010ba9d79af2a747b70b1f9da3df6b4f713323340b64648b9d` |
| 新helper | `51580e8bd7c08025edd69588e70e62a2970cb6afb5f97f355d85ddf48460a8ac` |
| 新driver | `a9e7bc648f1a8e34d802695124a3faf72998a55fe278a0000cc5cf1b803a4ca5` |
| 新tests | `c85ea73a0af2aa70f8ce1b55d6c4afdd00f6f37b9dbe7f22e4206beefc0e4c9a` |

最终 CPU 产物保留在 `/tmp/scut-payment-final-019-613ird16/mean/{identity,report}.json` 与 `/tmp/scut-payment-final-019-613ird16/stochastic/{identity,report}.json`。原proposal、旧native回执、原wrapper、生产及合同未改；不是source020的热修改，也不回滚当前PPO。

审核后由root单GPU串行运行，每次 `--output` 必须不存在。下面是完整本机命令；去掉实际原生运行、仅做CPU审核时添加 `--preflight-only` 并把 `CUDA_VISIBLE_DEVICES=0` 改为空值。准备者本轮未执行这两条原生命令。

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/yukikaze/isaacsim60-venv/bin/python -B /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/learning_audit/probe_prefix_mean_stochastic_payments_019.py --arm mean --source-root /tmp/scut-ordinary-alias-019-oit43rqd --source-manifest /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/learning_audit/source712_native_probe_manifest_019.json --checkpoint /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/mean_comparison_019/recovery_update29500/model.pt --output /tmp/scut-prefix-payments-seal2579-mean-001
```

第一臂完成并独立观察进程退出、GPU空闲后，再执行：

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/yukikaze/isaacsim60-venv/bin/python -B /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/learning_audit/probe_prefix_mean_stochastic_payments_019.py --arm stochastic --source-root /tmp/scut-ordinary-alias-019-oit43rqd --source-manifest /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/learning_audit/source712_native_probe_manifest_019.json --checkpoint /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/mean_comparison_019/recovery_update29500/model.pt --output /tmp/scut-prefix-payments-seal2579-stochastic-001
```

driver原样调用冻结55的authenticate／run_native。原actor网络、native sampler、resume／reset及600策略／2400物理循环均复用；新hook只调用原begin／step／event各一次并返回原对象。whole actor mask来自elapsed interval，actor造成FAILED的终止付款保留，SCRIPT／BLEND中途到RL仍排除。旧lesson/profile和first-live在begin保存，payment捕获在base首次done封口前；原motion goal取immutable skill spec，L0清零后的adapter命令不用作原目标。

输出包括原 `identity.json`、`report.json`、`events.jsonl.gz`、`first_birth_rows.jsonl.gz`，及新 `payment_summary.json`。原生通过必须同时满足：

- `report.status == native600_completed_preclose`，600／2400计数、0PPO、原birth／environment RNG／private sampling首末stream强匹配；`actual_payment_complete`、`payment_artifacts_unchanged`、`payment_hooks_restored_before_app_close` 都为true。
- payment schema为 `native-prefix-actual-payment-ledger-019-v1`，`complete=true`、600原event call、2,457,600原生lane-interval，identity及payment摘要SHA与parent report匹配。
- 无 `payment_observer_failure.json`／`payment_admission_failure.json`／`payment_postclose_or_admission_error.json`。summary自身不替代parent report准入。
- root另记真实进程退出；原wrapper的Kit.close可能直接退出，`process_completion.json` 缺失时 `app_close_returned` 仍未知，exit0不推成close返回或行为合格。

payment每行有scope（所有原生出生／首生）、lesson、原目标、互斥payment partition、policy-lane interval数量与 `lane_seconds=n×.02`。分项stats包含sum／mean／std／min／max。density分项明确reward/s及 `sum_payment=.02×sum`；success／failure event和returned／actor-eligible immediate reward直接按policy interval记，事件不乘dt。状态字段包括控制顺序实际dq、轮缘／腿速度max、真实gyro／vx／yaw命令、height／gravity以及最后reference八门和hold／followup峰。

逐lane强验 `returned_reward=.02×Σactual_components+actual_success_event+actual_failure_event`，不重新执行奖励producer、不漏正tracking peaks、不补未登记项。允许误差 `2e−5+1e−6×abs(reward)`；报告最大residual。旧takeover配置／分项出现即拒绝当前测量。

空箱表示无样本，不当零性能；首生terminal-inclusive后永久关闭，后续reset只进all-native，两个scope不能相加。状态是50Hz末真实plant，末八门是200Hz反馈观测，二者不同输入；pass均值与历史peak不能当连续1s概率。actor-eligible immediate payment不是优势、reward梯度或entropy梯度；本观察不造GAE／PPO，保存σ原值，解析Gaussian H也不当实际付款或优化归因。12s仍存活继续记右删失，保留最终sensor／contact／followup连续1s及原固定验收标准。
