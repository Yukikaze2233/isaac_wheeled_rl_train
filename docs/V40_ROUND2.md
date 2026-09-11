# V40 第二轮：显式 v2 训练 profile

**身份：`own-v40-jointspace-h5-v2`，通过 `--contract contracts/own_v40_v2.json` 选择。**
默认仍为 v1；v1 JSON 未改写，原始字节 SHA256 为
`97317c3b5c2263f8f85aac6f6b5e5cd44d44ca3f49fa2f6edb48edd3822c8c92`。
本轮是 **FD-inspired baseline**，复用自己的几何高度原点、关节空间动力学先验与共同奖励权重，不是完整 Fudan 复现。

## 设计边界

| 项目 | v2 实际行为 |
| --- | --- |
| 网络/时钟 | actor 125、critic 29、action 6；5×25 历史；MLP `[256,128,64]`、ELU；100 Hz policy / 200 Hz physics。复用原 stock PPO cfg，包括初始 std `.2`、48 steps、lr `1e-4`、5 epochs / 4 minibatches，其余超参数不变。 |
| 动作 | clip `100`；四腿位置 scale 均 `.5`，轮速 scale `10`。髋目标使用最近等价周期角，不再夹到 nominal ± `.35`；膝仅夹到机械硬限位（内角 35–80°）。 |
| 物理与软区 | 膝硬限位、连续髋/轮语义及 PhysX readback 检查保留。膝有限区间以中心缩到 `.97`，越界距离线性惩罚；不参与目标夹紧。旧 `knee_soft_margin` / `hip_soft_deviation` 字段仅保留历史声明，v2 不用它们夹紧或计算软区奖励。轮目标可达 ±1000 rad/s，这是期望值；实际力矩仍按当前转速查询原 torque-speed envelope，并受原 effort limit 限制，不是提高物理额定转速。 |
| 观测 | actor 加独立物理均匀噪声：角速度 ±`.2`、gravity ±`.05`、四腿 q ±`.02`、六关节 dq ±`1.5`，乘现有 obs scales 后实际幅度分别 `.1/.05/.02/.15`。命令和动作无噪声。每 policy tick 缓存 noisy obs25；同 tick 重读不消耗 RNG；局部 reset 只刷新对应行，首帧重复五份。critic 为干净当前 obs25 + 真值速度3 + 高度1。 |
| reset / evaluation | 训练 root 六维线/角速度独立 `U[-.5,.5]`；默认关节姿态、root quaternion 不变。固定命令评估 reset 速度为零、关闭观测噪声，快照 `evaluation_settings` 记录实际设置。种子及相同调用顺序可复现 CPU 随机序列。 |
| 终止 | 非有限数立即失败；gravity projected z `> -.1` 连续 **101 policy ticks** 才倾倒失败，恢复到 `<= -.1` 或 reset 清计数。同 tick 重读不重复计数。20 s episode 与 timeout 逻辑不变。AABB、低高度、非轮接触、膝小越界均不再触发 task done，物理限位仍存在。 |
| 日志 | `Termination/*` 与 snapshot 的六种 `termination_flags` 表示实际终止条件；v2 `tilt` 专指持续倾倒，其余禁用标志为 false。实际几何/接触/膝/倾角异常另记 `Diagnostic/*`；原 clearance、contact force 日志保留，不能把禁用终止标志解释成物理异常为零。 |
| 奖励 | 保留 velocity/yaw/height exp、upright、vertical_velocity、action_rate、effort、knee_soft_limit 的原权重；velocity/yaw kernel `.5`、height `.03`。独立 lateral_velocity、zero_command_translation 权重和 termination penalty 为零。没有 height square/enhance 或 alive bonus。 |
| 命令 | stand `(0,0,.32)`；height 高度 `.28–.32`；locomotion vx/wz 均 `[-2,2]`、高度 `.28–.32`。有限/有序校验保留；preflight 的 v1 特殊速度上限仅对 v1 生效，批准的机器人高度域不变。 |

## 服务器本地启动器

`scripts/start_v40_round2.py` 为 stdlib-only 本地 CLI。默认只输出 JSON 计划：不执行子进程、
不联网、不导入 GPU/Isaac、不创建目录。显式 `--launch` 才在**当前服务器**创建独立 tmux workers。
它不依赖旧 `start_v40_tmux.py` 的 DSH 通道。

| 参数 | 默认值 / 要求 |
| --- | --- |
| `--run-root` | 必填，绝对路径、新目录，父目录须已存在；包括已有空目录/符号链接也拒绝 |
| `--python` | 必填，目标服务器绝对 `ENV/bin/python`；不 resolve 解释器符号链接 |
| `--repo` / `--tmux` | 默认启动脚本所在仓库 / `/usr/bin/tmux`，均为绝对路径 |
| `--stages` | `stand locomotion`，每个 stage 独立随机初始化，不串接两个任务的权重 |
| `--num-envs` | 每个 worker `1024`；双 worker 共 2048 个环境，均使用训练 CLI 默认 `cuda:0` |
| `--total-iterations` / `--pilot-iterations` | 每 stage `20000` / `20`；pilot 完成后追加 `19980` |
| `--training-runtime-seconds` | 主训练 learn 阶段 `86400` 秒，不是整条流水线的 24h 总上限 |
| `--seed-base` | `41`：stand=41、locomotion=42；只选 locomotion 也仍为 42 |

所有 phase 都显式传入 `--contract <repo>/contracts/own_v40_v2.json --research --headless`。
启动器不提供外部 checkpoint 或 v1 contract 入口。每个 worker 按以下顺序执行：

1. v2 CPU preflight，要求退出 0、`ready=true`、`simulation_started=false` 及计划中的契约/stage 身份。
2. **fresh pilot**，默认 20 次完整 PPO update，使用当前全部 PPO 默认值，包括 **5 epochs / 4 minibatches**。
3. 复用 `wheeled_algo.v40_job.strict_json`、`validate_completion`、`artifact_record` 及
   `verify_export_sidecar`：要求 pilot 为 `completed`、次数准确、export verified，逐项核对所有
   receipt 产物的大小/SHA256，并核对本 worker 的 v2 contract/stage/seed 与导出 sidecar。
4. 仅以本 worker 的 `pilot/model_final.pt` 做真正 `--resume`，恢复模型、optimizer、stock iteration，
   在全新 `train/` 追加 `total - pilot` 次 update。pilot 未完成、退出异常、超时、凭据缺失或校验失败均阻断此步。
5. 主训练退出后重复校验。预算内提前停止可记录为 `stopped`；只有 `completed` 才表示本次请求次数全部完成。

训练 CLI 自身仍比较 profile ID、contract SHA、asset SHA、维度和 policy；v1 checkpoint 不能恢复到 v2。
显式 v2 JSON 原字节存为 `contract.json`。`SOURCE_FILES` 新增启动器，新的 `source_hashes.json`
会包含该文件；旧 run 的快照不追写，既有测试按当前列表动态生成快照，无需改历史 baseline hash。

### 时间边界与目录隔离

- preflight 外层上限 **120 s**；pilot learn 预算 **1800 s**；主训练 learn 预算默认 **86400 s**。
- pilot / train **分别在实际 phase 开始时**计算 `--stop-at = 当前 UTC + 1800 s 初始化余量 + learn 预算`。
  外层子进程 timeout 再加 **1200 s 保存、导出及退出余量**，覆盖完整初始化/训练/收尾；既有导出子进程上限为 900 s。
- 因此 pilot 外层上限为 **4800 s**，主训练为 **89400 s**，均不含超时清理。不是平台关机时间，
  启动器不读取或设置平台关机；平台断电仍可能留下未完成文件。
- 子进程使用 `start_new_session=True`。超时只对本次持有的子进程组发 TERM，等 **20 s** 后仍未退出才 KILL，
  再有界等待 20 s。不按进程名清理，不停止其他 tmux session。自行脱离该进程组的后代不在其清理范围；
  现有独立 CPU exporter 由训练代码自己的超时管理。

每个 stage 使用 `root/<stage>/pilot/`、`train/`、`audit/` 和 `usd_cache/`。
pilot/train 目录由训练程序独占创建，审计目录先建；重跑同 root 或同 worker 均拒绝覆盖。
新增公共 `--usd-cache-dir` 经 `make_env → V40EnvCfg → make_v40_articulation` 传入 importer，
recording 入口也传递此选项。两个 worker 使用不同的 `root/<stage>/usd_cache`，同 worker 的 pilot/main
顺序复用自身缓存，避免并发 `force_usd_conversion=True` 重写共享 USD。未传选项时保留精确旧路径逻辑
`Path(urdf).resolve().parents[2] / logs/v40_usd_cache/<asset_sha>`。

### 服务器命令：双 stage / 单 stage

以下命令在**服务器该仓库根目录**执行。`PYTHON` 使用既有目标环境路径，如服务器实际位置不同则改为实际绝对路径。
用户已接受 EULA；启动前显式导出已有接受值，启动器仅继承它，不自动声明接受。

```bash
REPO="$(pwd -P)"
PYTHON=/root/autodl-tmp/scut-isaac51-lab230/py311/bin/python
unset PYTHONHOME PYTHONPATH
export OMNI_KIT_ACCEPT_EULA=YES

# Default two-stage plan; creates no files or tmux workers.
"$PYTHON" "$REPO/scripts/start_v40_round2.py" \
  --python "$PYTHON" --run-root /root/autodl-tmp/v40-round2-001

# Launch the two independent workers using the defaults above.
"$PYTHON" "$REPO/scripts/start_v40_round2.py" \
  --python "$PYTHON" --run-root /root/autodl-tmp/v40-round2-001 --launch
```

只启动站立任务时，使用以下单 stage 命令（与上面的双 stage 启动方式择一）：

```bash
"$PYTHON" "$REPO/scripts/start_v40_round2.py" \
  --python "$PYTHON" --run-root /root/autodl-tmp/v40-round2-stand-001 \
  --stages stand --num-envs 1024 --total-iterations 20000 --pilot-iterations 20 \
  --training-runtime-seconds 86400 --seed-base 41 --launch
```

仅 locomotion 将 `--stages stand` 换成 `--stages locomotion` 并使用新的 run-root 即可。
`--worker <audit/plan.json>` 是启动器内部入口，用于 tmux 回调，不是重试/恢复接口。
tmux 使用真实多 argv 执行 worker，不经过 shell 拼接；清理 `PYTHONHOME/PYTHONPATH`、重建 PATH，
强制相机/直播为 0、Python unbuffered，并通过 tmux `new-session -e` 显式覆盖旧服务端环境，
包括继承的 `OMNI_KIT_ACCEPT_EULA`。未设置 EULA 时传空值，不会沿用旧 tmux server 的接受值。

### 查看进度与审计

启动输出给出各自的 `tmux attach -t v40-r2-...` 命令，pane 中显示阶段开始/校验/最终状态。
详细训练 stdout/stderr 合并写入各 worker 自身日志：

```bash
tail -F /root/autodl-tmp/v40-round2-001/stand/audit/pilot.log \
        /root/autodl-tmp/v40-round2-001/stand/audit/train.log
tail -F /root/autodl-tmp/v40-round2-001/locomotion/audit/pilot.log \
        /root/autodl-tmp/v40-round2-001/locomotion/audit/train.log
```

根目录 `plan.json` 包含所有命令、预算、session 和检查列表；`audit/plan.json` 为单 worker 计划。
`<phase>.started.json` 记录实际命令（含动态 stop-at）、子 PID、UTC/monotonic 启动时刻和 timeout；
`<phase>.status.json` 记录退出码/耗时/超时异常，`<phase>.checks.json` 记录验证结果；
`worker.status.json` 为流水线最终状态，`tmux.*` 是提交进程审计。
tmux 提交成功不等于训练完成；worker 退出后 pane 可能关闭，日志和 JSON 保留。
失败时查看已有 audit，使用新的 run-root 重启，不重用旧目录。

## 验证与交接限制

提交前最终验证：从 git 暂存区导出干净副本，使用 Python 3.11.15 / Torch 2.7.0+cpu /
RSL 3.0.1 运行 `tests/v40`，**695 passed，0 skipped**，27条现有 ONNX 弃用警告。
另在独立 RSL 2.3.3 环境执行 CI 的全部7个旧测试脚本，全部通过；两套依赖一致性检查、
actionlint 和暂存区空白检查通过。新增 CI job 与旧任务分离，补齐实际失败的 TensorBoard 依赖。
这是本地对 CI 配置的复现，尚未推送或触发新的 GitHub Actions 运行。

历史指定 Python 3.11 / Torch 2.7 / RSL 3 CPU 环境完整测试：**578 passed**，26 个 Torch ONNX 弃用警告。
命令：`PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /tmp/opencode/train-ci-v40/bin/python -m pytest -q`。
新增 `tests/v40/test_round2.py` 覆盖真实 core、AST 提取 env 方法、限位/软奖励、101-tick/reset、噪声缓存/clean critic、种子/evaluation reset、v2 ONNX 导出与 provenance、拒绝 v1 checkpoint。

**当前服务器已关机，本次仅实现本地代码与测试，没有连接服务器或启动训练。** 按当前交接记录，
continuous USD adapter 已在关机前完成实际 GPU 约 **3.15 圈**验证；本次未复测。
新启动器的真实双 worker / 1024 env 吞吐与显存、20-update pilot → resume 集成及训练质量尚待服务器运行验证。

本次使用持久本地环境 `.venv_mj314`（Python 3.14.7，配置的 Torch 2.14.0+cu130；仅 CPU 测试），
不使用重启后已删除的 `/tmp` CI venv，不代表目标 Python 3.11 / Torch 2.7 runtime 验证。
新增 `tests/v40/test_round2_launch.py` 覆盖无副作用计划、v2/自身 pilot 门、hash/export 失败、准确次数及种子、
每 phase 新 cutoff、分离缓存、tmux 多 argv/EULA、超时 owned-group TERM/KILL 与拒绝覆盖。

```bash
PYTHONPATH=src:../.rl_deps_rsl23 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/yukikaze/Documents/workspace/robot_rl/.venv_mj314/bin/python -m pytest -q \
  tests/v40/test_round2_launch.py tests/v40/test_round2.py \
  tests/v40/test_env_contract_static.py tests/v40/test_joint_import_limits.py \
  tests/v40/test_launch.py tests/v40/test_record.py \
  tests/v40/test_job.py::test_raw_contract_asset_snapshots_and_real_strict_export
# 257 passed, 4 warnings in 4.98s (existing Torch ONNX deprecation warnings)
```

第二轮显式纳入平地行走；`v40_metrics.validate_command()` 已按明确的 v1/v2 身份和实际训练 stage 范围校验，evaluation / recording CLI 可接受 v2 locomotion 的 `vx/wz ∈ [-2,2]`（包括 `2 m/s` 边界），缩窄的训练范围仍严格生效。v1 原速度上限、共同高度域 `.28–.32 m`、checkpoint 身份/hash 与 trained-stage 检查、metrics 的保守轨迹验收条件均保留；后者独立于 v2 task termination。质量验收不能从本地 tensor 测试推断。
质量/摩擦/PD 随机化、action delay、encoder 模型本轮均未实现；encoder zero/sign、链传动映射和电机曲线仍未标定，默认姿态不代表零位标定已完成。
