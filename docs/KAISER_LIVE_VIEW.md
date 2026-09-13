# V40 训练时实时可视化（Sim 6 / Lab 3）

## 范围与架构

`train_v40.py --live-view` 显式开启；默认关闭时没有 renderer、网络、额外 GPU 回读或 viewer 进程。

训练使用当前 PhysX 七 body 状态。独立 Linux/WSL viewer 进程读取当前合同审核过的 canonical URDF/STL，只进行浏览器几何渲染。保留 URDF 的每个 visual 局部 translation、RPY、scale、RGBA，尤其底座的局部 Rz(π/2)，不重新设计短杆、不重算 FK、不运行第二套物理。

`LiveViewSession` 在现有 RL wrapper 的 `step()` 返回后旁路采样，保留原始动作、返回值与 PPO 实现。只有墙钟距离上次采样至少 50 ms 时，才读取 `body_link_pose_w.torch[env_id]`：在 GPU 上选一个 env，然后 `detach().cpu()`。不在 rollout 中 sleep，也不把 simulation time 冒充墙钟视频速度。原始 PhysX xyzw 四元数归一化后传输，viewer 转成 wxyz 并与 URDF visual 子节点组合。

同机传输使用非阻塞 Linux abstract Unix domain datagram，名字绑定随机 run UUID，消息同时绑定 scene manifest SHA-256。此选择来自 Kaiser 实测：localhost UDP 1,000-byte 自发自收成功，1,600-byte 以上静默丢失，而真实七 body 加 telemetry 的包超过该阈值。Unix IPC 避开该 WSL 网络路径，单包上限 60 KB，不对外开放物理控制。

页面显示 `TRAINING · LIVE`、step、sim time、包年龄和 sim/wall 比值，并标明是训练最新采样状态。1 秒没有新训练姿态显示 STALE；优化器、导出或其他阶段可能导致正常的采样间隔。收到 ended 后显示 ENDED 并约 3 秒自动关闭。超过 viewer 生命周期或父进程退出也会关闭。viewer 自然到期后训练继续，原因写入 `publisher.json`。

状态取样位于 **post-step / auto-reset 后**；显示的 reward 属于前一 transition，command/action 可能已因 reset 更新。因此它是调试观察工具，不是严格的 pre-reset 轨迹记录器。

## 隔离环境

使用现有独立 viewer venv，或安装一个新的（在有界 tmux 中执行）：

```bash
bash scripts/v40_live_view/setup_viewer.sh \
  /path/to/uv /path/to/python3.12 /new/path/viewer-env
```

要求 `viser==1.0.16`、`websockets>=13.1,<16`；拒绝在安装了 Sim/kernel 的解释器中启动 viewer。启动子进程会移除 Sim 的 PYTHONPATH、PYTHONHOME、LD_LIBRARY_PATH 和 VIRTUAL_ENV，且不安装 Newton。官方 Lab Viser extra 绑定 Newton，并不适合直接装进此 PhysX 环境。

原 Sim 环境的 `websockets==12.0`、`newton==1.2.0` 保留。上游 Pillow、packaging、coverage 等互斥依赖声明按部署 HANDOVER 原样记录，本功能不尝试解决。

最终 `uv pip check --python <Sim Python>` 仍退出 1，检查 246 个包并报告以下 4 项；独立 viewer 的检查退出 0：

```text
The package `isaaclab` requires `pillow==12.2.0`, but `11.3.0` is installed
The package `isaacsim-core` requires `packaging==26.0`, but `23.2` is installed
The package `isaacsim-kernel` requires `pillow==12.1.1`, but `11.3.0` is installed
The package `isaacsim-kernel` requires `coverage==7.4.4`, but `7.6.1` is installed
```

## 开箱观看：仅工程 smoke

在已部署快照的根目录运行，参数替换为 Kaiser 实际路径：

```bash
bash scripts/v40_live_view/start_smoke.sh \
  /path/to/sim60-runtime.sh \
  /path/to/viewer-env/bin/python \
  /existing/path/to/run-root 60
```

脚本创建新 run、新 tmux，固定 **2 env、3 次 PPO 更新**；外层 timeout 300 秒，训练软预算 90 秒，viewer 生命周期 180 秒。浏览器等待 60 秒只发生在训练开始前，超时明确失败。首次导入模型可能需要几十秒。看到 `V40_LIVE_VIEW_READY` 后，在 Kaiser Windows 浏览器打开：

```text
http://127.0.0.1:8088
```

TCP 8088 是页面/WebSocket，8089 是 JSON 状态，均绑定 localhost；Kaiser Windows localhost 已由此前环境探测确认可访问 WSL。姿态走 Unix socket，没有 UDP 监听端口。浏览器可以拖动、缩放。短 smoke 的真实训练画面只有数秒，随后为导出阶段 STALE 和最终 ENDED。

这里的三次更新只验证训练链路、资产导入、传输、浏览器和 ONNX 数值导出，**不是 Round3 长训、有效策略或实物验收**。

## 正式训练接入命令

在独立 tmux 中使用既有 runtime，然后对现有训练命令追加：

```bash
source /path/to/sim60-runtime.sh
timeout --signal=TERM --kill-after=20s 10m \
  python scripts/train_v40.py --research --headless \
    --num-envs 2 --max-iterations 3 --max-runtime-seconds 90 \
    --run-dir /new/path/to/run \
    --live-view --live-view-env 0 \
    --live-view-python /path/to/viewer-env/bin/python \
    --live-view-port 8088 --live-view-status-port 8089 \
    --live-view-seconds 180 --live-view-wait-seconds 60
```

上例仍是短 smoke；主 agent 决定正式训练的合同、阶段、并行数及更新预算。默认 `--live-view-wait-seconds 0` 不等待浏览器。viewer 生命周期默认 3,600 秒、允许 1–86,400 秒；等待允许 0–120 秒。端口占用明确失败，不自动换端口，不会错误显示另一个 run。

从另一台机器访问时可用现有 SSH 转发，例如：

```bash
ssh -p <SSH_PORT> -L 18088:127.0.0.1:8088 <USER>@<HOST>
```

该机器打开 `http://127.0.0.1:18088`。Kaiser Windows 本机不用此转发。

## Torchvision 严格构建契约

`TARGET_VERSIONS["torchvision"]` 为 **`0.26.0+cu128`**，与 `torch==2.11.0+cu128` 对齐；其他 TARGET 不变。现有严格匹配继续拒绝无后缀、cu130、CPU 和 dev 构建，绝不 strip CUDA 后缀。

远端这一构建已完成 CUDA NMS 验证。此前本机 PyPI `0.26.0` wheel 实际 `version.py` 是 cu130 且 NMS 不可用，因此现在本机应在 preflight 被拒绝；需由环境负责人换为经过校验的 cu128 wheel，再验证 CUDA NMS，而不是放松 gate。

## 证据、复查和已知局限

每次训练的 `live_view/` 保存：

- `scene.json`、七个 run-local STL：合同/资产哈希、视觉局部位姿、七 body 顺序、2 env 的 solver limit 报告。
- `ready.json`：单次 run ID、manifest 哈希、端口、Unix socket 名、mesh/vertex/face 计数。
- `published.jsonl`、`received.jsonl`：仅 ≤20 Hz 的实时消息，用于核验，未参与离线播放。
- `publisher.json`、`viewer.json`、`viewer.log`：收发数量、终止原因、子进程回收。
- 原有 `completion.json`、`model_final.pt`、`policy.onnx` 和 sidecar：PPO 完整更新与 9 样本 CPU ORT 验证。

`browser_check.py` 用专属浏览器 CDP 采集真实 WebGL 截图、DOM 阶段和 worker WebSocket。它会关闭目标诊断浏览器，不能连接用户日常浏览器的 CDP。`audit_run.py` 在独立 viewer env 中对比浏览器消息与训练发布的当前 PhysX 姿态，并检查进程及端口回收。

当前显示 canonical visual mesh，不是碰撞 hull、RTX 材质、相机视频或接触力可视化。没有改变原资产短杆、碰撞过滤、动力学或关节标定。GPU 单 env 回读仍有同步成本；大并行规模与长时间性能尚需测量。多次 reset 或快于墙钟的仿真会表现为状态跳变，这是最新采样的真实语义。

源码可用 `make_snapshot.py --output /new/archive.tar.gz` 打包：包含 tracked 和 untracked 源码当前内容及 dirty 修改，排除运行目录，不修改 Git；记录每个文件 SHA-256。主 agent 负责最终全仓测试、commit/push。

## 2026-09-12 Kaiser 实测交接

有效部署：`/home/kaiser/robot-rl-sim60/v40-live-snapshot03`。该快照包含打包时的 dirty 工作树；没有覆盖之前的环境、快照或 run。

Kaiser 的可直接运行入口：

```bash
S=/home/kaiser/robot-rl-sim60/v40-live-snapshot03
bash "$S/scripts/v40_live_view/start_smoke.sh" \
  /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh \
  /home/kaiser/robot-rl-sim60/live-visualization/viewer-env/bin/python \
  "$S" 60
```

打开 `http://127.0.0.1:8088`，首次启动尚未完成时刷新页面。每次运行会打印唯一日志目录；该目录的 `train.log` 包含 `V40_LIVE_VIEW_READY`。这仍然只运行 3 次更新。

有效 run：`v40-live-20260912T234215-100466`，可视化 UUID `c31c30a0aa2843538f7769b2cf406f92`。

| 项目 | 结果 |
|---|---|
| 原资产导入 | 2 env、7 body、6 DOF；实际 solver limits 形状 `[2,6,2]`，连续关节和有限膝关节均通过 |
| PPO | 3/3 次完整更新，144 policy steps；learning 墙钟约 5.616 秒，无 rollout sleep |
| ONNX | CPUExecutionProvider，9 样本通过；最大绝对误差 `6.333e-7`，atol `1e-6` / rtol `1e-5` |
| 传输 | 75 条发送 / 75 条接收，其中 73 条 training；丢弃 0、拒绝 0 |
| 真实模型 | 浏览器实际收到 7 个 mesh；245,168 个三角面，735,504 个未合并 STL 顶点 |
| Windows Edge | 391 个 WebSocket 帧；JS 异常 0；观察到 TRAINING · LIVE、STALE、ENDED；step 57→131 截图排除侧栏后 37,460 个场景像素变化 |
| 姿态核验 | 73 个采样 policy step × 7 body = 511 组浏览器位置/朝向与实时 PhysX 发布值一致，容差 `1e-10` |
| 退出 | 训练退出 0、viewer 退出 0；专属 Edge 退出 0；TCP 8088/8089 及 abstract Unix socket 已释放 |
| 环境隔离 | Sim 完整 freeze 仍与部署前逐字节相同；独立 viewer 的 44 个包依赖检查通过 |
| CPU 检查 | `tests/v40/test_live_view.py`：8 passed；Git diff whitespace 检查通过；本次功能源码与有效快照 SHA-256 一致 |

证据位于有效快照下：

- `<run>/train/completion.json`、`policy.onnx.json`：PPO 和 ONNX 数值结果。
- `<run>/train/live_view/integration-audit.json`：七 mesh、511 组姿态、全部关节限位、清理结果。
- `browser-evidence/browser.json`、`websocket-frames.json`：实际 Windows Edge DOM 和消息。
- `browser-evidence/frame-005.png` / `frame-011.png`：step 57 / 131 的训练画面。
- `snapshot.json`：241 个源文件的打包身份；归档 SHA-256 为 `e7e34cb40485ef4ec20bc6fd7cb97380de14783ca981d50f150789bdfccabbe9`。

本机临时回执副本在 `/tmp/opencode/v40-live-validation/`，Windows 浏览器原始证据在 `/tmp/opencode/v40-live-browser03/`；这些大文件不作为源码提交内容。

早期 snapshot01 因打包漏掉 provenance 要求的 `tools/prepare_v40_assets.py` 在建 env 前失败；snapshot02 已完成 3 次 PPO/ONNX，但因这台 WSL 的 UDP 长包问题未实现持续显示。最终判定以 Unix IPC 的 snapshot03 为准。当前全部诊断进程已停止，后续观看需重新启动入口。没有启动 Round3，也没有对策略质量、机械归属或硬件安全作验收结论。
