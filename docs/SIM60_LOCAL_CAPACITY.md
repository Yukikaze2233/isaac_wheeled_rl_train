# Sim 6 本机容量与 GUI 验证（2026-09-12）

## 容量结论

后续本机 headless 训练推荐先用 **512 环境**；这是本轮实际验证的最大规模，并非 8GB 显存的容量上限。
GUI 观察训练先用小环境数，避免把显示开销混入训练吞吐比较。

工作树仍为 `isaac_wheeled_rl_train-60` / `isaac60`，HEAD `4ca58112c1ab23bea805a4d40ce707e61d244ef6`；原有未提交兼容修改全部保留。运行时身份详见 [第一轮本机验证](SIM60_LOCAL_VALIDATION.md)。本轮没有服务器访问，也没有启动长期20k迭代训练。

日志根目录：`reports/local_sim60_capacity_20260912/`。

## 真实 PPO 容量短测

四档都直接执行 `scripts/train_v40.py`，没有给 benchmark 添加逐步审计 wrapper。
统一 V2 `locomotion` / seed40 / cuda:0 / headless；125D actor、29D critic、6动作及全部奖励、执行器、物理限位配置一致。
每档 **12次官方 PPO 更新**，前2次作为预热，用其后10次的真实 collection + learning 时间计算吞吐。

| 环境数 | PPO更新 | 预热后 transition/s | 训练阶段耗时 | 含启动/导出总耗时 | 整卡采样峰值 |
|---:|---:|---:|---:|---:|---:|
| 64 | 12 | 1277.3 | 29.60 s | 44.20 s | 3668 MiB |
| 128 | 12 | 2528.5 | 30.31 s | 49.29 s | 3690 MiB |
| 256 | 12 | 5428.4 | 28.12 s | 64.73 s | 3772 MiB |
| 512 | 12 | 10324.5 | 29.59 s | 136.60 s | 3826 MiB |

- 全部退出0、完成12次更新，checkpoint / optimizer 张量、TensorBoard scalar 全部有限；`Termination/nonfinite=0`，actor相对第一次更新后的权重确实继续变化，CPU ONNX 校验通过。
- `bench{N}/analysis.json` 为 CPU 事后审计；`train{N}/completion.json` 为训练产物回执；`summary.json` 保存精确argv、PID和wall time。
- `bench{N}/gpu.jsonl` 每秒采样显存、GPU利用率、温度。整进程平均GPU利用率分别约71.4%、71.5%、64.5%、67.2%，最高温度63–64°C。这些整卡数据包含用户原有 Sim 6 GUI（PID123023，约1135 MiB）及桌面，不能当作训练进程独占占用。
- 每档外层 wall budget 300秒，learn budget 220秒；整卡达到7164 MiB时 supervisor只给自己启动的进程组发停止信号。实际均未触发超时或显存停止。
- 512档的场景创建耗时约95.27秒，明显高于小档；目前保留独立USD克隆和原物理过滤检查，没有为了扩大规模改成未经验证的复制路径。
- 这是短时吞吐测量，尚未验证长期热稳定性、收敛和超过512环境的行为。

## 本机启动方式

在本工作树运行。脚本保留两个位置参数，默认64环境、100次更新；默认headless，`--gui` 可放在位置参数前后。

```bash
# 后续 headless 训练的推荐起点；本轮没有启动这条较长训练。
bash v40_train_local.sh 512 1000

# 小规模 GUI 训练，显示当前新训练策略及其探索动作。
bash v40_train_local.sh --gui 8 100

bash v40_train_local.sh --help
```

GUI 窗口属于训练进程。用户单独打开的 Isaac Sim GUI 不会自动显示训练进程中的场景。
每次脚本使用独立的 `runs_v40/local-{headless,gui}-train-*` 目录，保留默认8小时相对learn预算；模型初始化和收尾仍应留时间。

## GUI 依赖与兼容边界

本机原 venv 缺少 `isaaclab_visualizers` 的安装元数据，但对应源码已经存在。
只向 **`/home/yukikaze/isaacsim60-venv`** 从既有 Lab checkout 添加了 editable `isaaclab-visualizers==0.1.0`，无下载、无依赖升级、无全局安装：

```bash
uv pip install --python /home/yukikaze/isaacsim60-venv/bin/python \
  --no-index --no-deps --no-build-isolation \
  -e /home/yukikaze/Documents/workspace/robot_rl/.deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab_visualizers
```

Lab 3 必须显式选择 `visualizer=["kit"]`，仅省略 `--headless` 不足以启动GUI。
该 beta 还读取但未填充 `/isaaclab/has_gui`；本地入口先核对真实 Kit native window、window handle 和正尺寸，再补齐此标志，之后创建 SimulationContext。相机以env0为原点近距离观察，不改机器人状态。

GUI 首次审计 `gui01` 因上述缺失标志而在训练前失败，原失败日志保留，未计作成功训练。

## GUI 实测与截图

`gui02` 由训练进程自身打开 Kit 窗口，8环境、2次真实PPO更新、96个vector policy steps完成，checkpoint/CPU ONNX导出成功。
训练阶段9.554秒，总wall time32.970秒，整卡采样峰值5702 MiB（仍包含用户原GUI）。
实际 `has_gui=true`、`render_enabled=true`、visualizer为`kit`、viewport为1280×720，审计见 `gui02/gui_runtime.json` 和 `artifact_audit.json`。

截图来自该训练进程自己的实际 viewport，未截取用户桌面或其他窗口：

- [策略步20](../reports/local_sim60_capacity_20260912/gui02/viewport_policy_step20.png)
- [策略步70](../reports/local_sim60_capacity_20260912/gui02/viewport_policy_step70.png)

画面展示**本机V40 V2 locomotion / seed40 / 新初始化并训练两次更新的策略**，包含探索动作，不是服务器checkpoint的效果。
两个时刻画面不同，机器人位于真实地面场景；这些截图不能当作已学会站立的证明。
Niri没有按Python PID返回该X11窗口条目，因此没有生成整窗口截图；保留的是Kit viewport直接抓取的原始PNG。

## 地面接触：独立物理证据

另跑 `ground02`：2环境、200个policy steps / 400个physics steps、归一化零动作（腿为原nominal目标、轮目标速度0）。
`fix_base=false`，重力`[0,0,-9.81]`；没有固定基座或冻结根姿态。保留原任务的随机reset速度。
使用真实 PhysX `get_contact_force_matrix`，明确过滤到 `/World/ground/GroundPlane/CollisionPlane`，不是把未过滤的刚体净力当作地面力。

| 环境 | 最终root高度 | 最后50步左轮地面力均值 | 右轮地面力均值 | 基座地面力均值 |
|---|---:|---:|---:|---:|
| 0 | 0.24755 m | 40.15 N | 40.51 N | 40.09 N |
| 1 | 0.24749 m | 39.89 N | 40.63 N | 44.59 N |

两环境最后50步左右轮地面接触比例都是100%，提前终止0次。**确认车有真实地面接触、不是悬空；同时基座也触地，所以该零动作诊断不是合格站立姿态。**
原V2把部分接触/高度问题作为diagnostics的任务定义保持原样，未为了通过短测改变奖励或终止条件。
数据和逐步样本见 `ground02/ground_probe.json`；总wall time18.49秒。

## 修改与收尾

本轮新增/修改范围：

- `v40_train_local.sh`：可选`--gui`、help和位置参数校验；默认仍headless，支持原有两个位置参数。
- `scripts/train_v40.py`：显式Kit visualizer、真实窗口检查与Lab beta GUI标志兼容、env0近景相机。
- `tests/v40/test_launch.py`：更新GUI分支的预期visualizer参数。
- `README.md`、本记录及第一轮记录的链接。
- 新日志目录 `reports/local_sim60_capacity_20260912/`。

相关入口回归29项通过；`bash -n`、`git diff --check`通过。使用不启动Sim的argv替身验证headless/GUI参数、`--gui`前后位置和非法参数退出2。
本轮短测使用的独立tmux会话前缀为`v40-local-cap*`、`v40-local-gui*`、`v40-local-ground*`；完成后全部自动退出，用户原Sim GUI和其他会话保持运行。兼容改动仍未提交。
