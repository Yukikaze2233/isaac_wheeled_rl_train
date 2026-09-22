# V5 integrated SCUT35：Kaiser 独立训练与随时回收

本页的正式运行身份记录9/21历史任务。9/22的新训练与TensorBoard当前身份见[V5.2部署回执](V52_DEPLOYMENT_20260922.md)。

执行合同：`contracts/v5_scut35_integrated_v1.json`。继承高速方案的技能定义，
以7个大阶段替代48个逐项训练阶段；每个阶段内并行采样多个技能，后续阶段保留所有早期速度档的回归案例。

## 配置和预算

- 华南虎V14参考：4096环境、200Hz物理、50Hz策略、rollout24；普通PPO，单帧35D actor与81D privileged critic。
- 使用V5机械资产、10MPa气簧曲线和原执行器先验。没有新增物理传感器或外部起跳辅助。
- Flat 20000、landing 3000、rough 5000、spin_translate 5000、high_speed 5000、jump 8000、mixed_robust 6000，共52000参考更新。
- 扩容时预算按`4096 / num_envs`换算，命令课程按采样量推进。2k/3k/4k为冲刺/低速自旋/中速自旋的参考解锁点，并设置渐进过渡。
- 每100更新封存一次恢复checkpoint；首次10更新额外封存以验证回收链路。
- 每500更新结束一个采样块，独立评测。块间恢复优化器，阶段间迁移已验收模型；完整课程验收不通过时不会越过门槛。
- 每阶段设最少训练量；预算内不会因三个早期评测未过门就结束整个课程。
- 单次远端作业最长7天。预计训练时间以容量探针为依据，轮次预算不等于技能收敛保证。

## 本地关机后的持续运行

训练编排器、训练子进程与封存器全部运行在Kaiser端tmux中，socket位于
`/home/kaiser/robot-rl-sim60/tmux.sock`，不依赖本地ControlMaster或本地watcher。

Kaiser宿主必须保持开机、供电，WSL不能被关闭。部署前确认交流供电下睡眠和休眠超时均为0。
tmux不提供宿主断电后的自动开机；已封存checkpoint可用于恢复模型、优化器、随机数与课程进度，物理回合重新初始化。

## 运行身份

每次运行使用独立目录，保存源码tar归档、完整合同、基准commit、显式overlay逐文件SHA-256与launch回执。
overlay用于冻结已审查但未提交的工作区代码；它不会把未跟踪的其他工作混入归档。
实际训练身份以归档SHA和合同为准，不只以基准commit为准。

## 查看与回收

### TensorBoard：训练图表的统一入口

当前浏览器地址：`http://127.0.0.1:6006`。
服务器在Kaiser的`v5-tensorboard-a4ae54`独立tmux会话中运行，监听Kaiser的`127.0.0.1:6006`，
读取本轮`train/`下的event文件，每15秒刷新。它不向训练进程下发命令。

本地转发已改为`v5-tensorboard-tunnel.service`用户服务，SSH退出后5秒自动重连。
本机关机只结束本地转发；这是临时用户服务，重新开机后在6006端口空闲时执行：

```bash
systemd-run --user --unit=v5-tensorboard-tunnel \
  --property=Restart=always --property=RestartSec=5 \
  --setenv="SSH_AUTH_SOCK=${SSH_AUTH_SOCK}" \
  /usr/bin/ssh -S none -N -T -o BatchMode=yes -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519" -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:6006:127.0.0.1:6006 -p 2222 kaiser@192.168.64.234
```

检查服务：`systemctl --user status v5-tensorboard-tunnel.service`。
如果本地地址打不开，先查本地6006监听和该服务，再查Kaiser的TensorBoard；不能据此断言训练停止。

常用tag：

- `Train/mean_reward`、`Train/mean_episode_length`：回报与回合长度。
- `Loss/value`、`Loss/surrogate`、`Loss/learning_rate`：优化过程。
- `Perf/total_fps`、`Perf/collection_time`、`Perf/learning_time`：采样吞吐与耗时。
- `/reward/*`：实际进入训练的分项奖励。
- `/task/pin_gap_m`、`/task/height_error_m`、`/task/wheel_contact`：物理与行为诊断。

不同训练块在TensorBoard中为不同run，名称如`stage_00_flat/block_000`；step为该阶段PPO计数。
reward曲线不能代替固定案例验收。图表数据以event为准，JSON用于合同、验收判定和恢复回执。
当前Kaiser宿主内存余量不适合额外启动Kit GUI，本轮已验证的是TensorBoard访问。

### 进度与产物

在仓库根目录，将`<launch.json>`替换为本次运行回执：

```bash
python3 -B scripts/check_chassis_remote.py <launch.json> --brief
python3 -B scripts/sync_chassis_batches.py <launch.json> --output reports/v5_integrated_batches --once
```

重复回收会跳过已校验批次。远端每30秒检查并封存新产物；所有归档和文件均做SHA-256校验，
只下载已完成的不可变批次，不停止训练。恢复快照路径形如：

`train/stage_00_flat/block_000/checkpoints/update_00000100/model.pt`

同目录`contract.json`与checkpoint绑定。训练保存包含actor、critic、优化器、学习率与成功更新计数；
`--resume`从该合同继续同阶段。不同任务阶段使用`--transfer`并做接口兼容检查。
完整课程编排也支持`run_full_chassis.py --resume <sealed-model.pt> --start-stage flat ...`；
此时`--updates`是包含已恢复轮次的累计上限，阶段切换仍使用通过验收的模型。

结束后可完整回收：

```bash
python3 -B scripts/watch_chassis_artifacts.py <launch.json> --output reports/v5_integrated_final
```

候选checkpoint和通过验收的`accepted_policy`分别标记。工程探针、ONNX导出成功与技能通过是不同的结果。

## 本轮正式运行

- 启动：2026-09-21 21:17:50（北京时间）。
- 会话：`v5-scut35-20260921T131715Z-a4ae54`。
- 远端根目录：`/home/kaiser/robot-rl-sim60/experiments/v5-scut35-20260921T131715Z-a4ae54`。
- 本地回执：`reports/v5_integrated_formal_20260921/launch.json`。
- 实测选用6144环境，完整预算34670实际更新，等效约52000次4096环境更新。
- 平地4096/6144各完成100次真实PPO；6144混合场景完成24次PPO，均导出通过、闭链越界终止0。
- 6144平地吞吐32617 samples/s，相比4096的28528提高约14%；8192因内存预留门限未尝试。
- [冻结身份及容量摘要](evidence/v5_integrated_launch_20260921.json)。
- 21:22:58检查：flat阶段44次实际更新、6488064 transitions；tmux无附着客户端，worker仍存活。
- 断开查询连接120秒后更新继续增加；远端已先于本地回收自行封存第10次更新。
- 首批4文件SHA-256校验通过，checkpoint含actor、critic与17项优化器参数状态；技能验收仍待训练完成。

```bash
python3 -B scripts/check_chassis_remote.py reports/v5_integrated_formal_20260921/launch.json --brief
python3 -B scripts/sync_chassis_batches.py reports/v5_integrated_formal_20260921/launch.json --output reports/v5_integrated_batches_20260921 --once
```
