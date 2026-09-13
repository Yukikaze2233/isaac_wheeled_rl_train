# Round3-A 原生 V40 可视化：当前状态与 checkpoint 回放准备

## 结论（2026-09-14 03:27 北京时间只读核查）

**当前正式训练没有可消费的姿态 publisher，不能把任何新窗口称为当前 PID 446292 的实时训练状态 GUI。**可推进的是：复制一个稳定 checkpoint，在新的 Windows 实验目录中用同一 V40 资产和策略做独立 PhysX 推理回放，经原生 Kit/WebRTC 显示到 Yukikaze 当前 Linux。

该窗口、状态和回执必须标为 **TRAINING CHECKPOINT REPLAY**。它不是离线 pose 轨迹，但也不是正在训练的那个仿真实例；动作由所选 checkpoint 的 deterministic mean actor 产生，Windows 中重新 reset/求解物理，不做学习更新。

本轮只完成兼容审查、CPU 元数据/内存检查、已有 checkpoint 的只读哈希，以及新脚本准备。**未启动 Kit、CUDA、推理或训练，未复制/修改 frozen repo，也未停止回传 watcher 或评估 watcher。**

## 当前正式训练事实

- PID：`446292`，代码根 `/home/kaiser/robot-rl-sim60/isaac_wheeled_rl_train`，frozen commit `6ed83408d1d1cb02ea687e77fa80e5c20932f36c`。
- argv 包含 `--headless --num-envs 1024 --max-iterations 10000`、Round3-A contract 和 Round2 `--warm-start`，**没有 `--live-view`**。
- run：`/home/kaiser/robot-rl-sim60/round3-runs/train-formal01/train`。
- 未发现 `live_view`、pose/publisher/state-stream 输出；只有常规 checkpoint、TensorBoard、合同/来源清单等。checkpoint 不包含可连续消费的当前七 body 姿态流。
- 03:27 可见的最新文件为 `model_1000.pt`；`completion.json` 尚不存在。1000 是文件名索引，不能据此宣称最终更新数或达标。
- 只读前后 stat 一致的该 checkpoint：1,493,429 bytes，SHA-256：

```text
15d878dd5b796866f0e565d0c3e53eef6d1adfdc0ee53afc24ff9a9be14e70e0
```

其余输入文件本次哈希：

| 文件 | SHA-256 |
|---|---|
| run_manifest.json | `2e38ae3c507243c1bb7ec220fc72c9d2b1dcb50394846e0a74e5958b09a38b6d` |
| contract.json | `91ede15d5f5ae2de7d381e1f3a66f229005f213973f71084aca9a241335676c3` |
| source_hashes.json | `d5783adfb1d1a6107ed1f137c15c46c72fee9cd5df81d69cf4ffec5bdf1caece` |
| asset_manifest.json | `df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886` |

之后可能已有更高编号 checkpoint；准备回放时重新选择已经写完的文件，记录其编号/mtime/size/SHA，再复制并复核。不要对正在写入的最新文件直接 `torch.load`，也不要等待 completion 才读取 periodic checkpoint；现有 final-artifact pull 协议与 periodic-copy 是不同用途。

## 资源与兼容性

| 项目 | 本次实测/审查 |
|---|---|
| Windows RAM | 总 33,444,524,032 bytes（31.15 GiB）；空闲 **8,660,987,904 bytes（约 8.07 GiB）** |
| WSL RAM | 上限约 15.19 GiB；available 约 7.71 GiB，不能与 Windows free 相加 |
| Windows 原生进程 | `D:\isaac60-native` 下没有正在运行的原生进程 |
| Windows 包元数据 | Lab 6.1.14 / PhysX 1.1.3、Torch 2.11.0+cu128、torchvision 0.26.0+cu128、RSL 5.5.1、ONNX 1.23.0rc1、ORT 1.30.0；NumPy 2.5.2、Pillow 12.2.0 |
| 已有 GUI 证据 | Windows 原生 Kit → Linux 官方 WebRTC 的完整菜单/Stage/viewport与鼠标回传已验证；仅 Cube 场景，不能代表 V40 成功 |
| 安装后运行 | Windows Lab 安装完成，但 CUDA/NMS、V40 URDF 导入和推理尚无成功回执 |

之前 Cube Kit 本身占约 5.5 GiB host RAM，V40 导入/物理还会增加开销。当前长训与可能的 2048 benchmark 共用这台机器，**8.07 GiB free 不构成并发重 Kit 的充分余量**。新启动包装默认要求至少 12 GiB Windows free RAM（保护长训的操作门槛，不是官方硬件规格）；由主 agent 排期再启动，不能为过门槛擅自结束其他进程。

既有 `play_v40_onnx.py` 复用了 `train_v40.py` 的 preflight/launch/make_env，受 Linux `os.geteuid`、原生窗口检查及 pip `isaacsim` metadata 假设影响。当前 periodic checkpoint 也不是已经导出的 ONNX。**不能直接把现有 Linux replay 命令换成 Windows 路径便称兼容。**

新准备的入口避开该 CLI 适配层，但仍复用 frozen repository 的 `load_actor_checkpoint()`（weights-only、actor/critic 全量形状/有限值/合同验证）与 `V40Env`。它不放松现有 train gate、不伪造 `isaacsim` 分发 metadata；Windows binary 使用已经验证的官方 ZIP 和 Lab 源码回执标识。V40 importer 的 `run_asset_transformer=False` / `run_multi_physics_conversion=False` 及 canonical URDF/STL 由原资产代码保留，验证七 body/六 joint，禁止外部 USD seed。

## 新入口：仅准备，未运行

- `scripts/windows_native/checkpoint_replay.py`
  - 默认 CPU/stdlib preflight，不 import Torch/Isaac，不启动 Kit；`--launch` 才进入原生推理。
  - 验证复制 checkpoint SHA、训练 source_hashes 所列代码、合同与 canonical 资产；使用独立 1-env CUDA PhysX 场景。
  - 显式 Kit visualizer 与 Fabric transform 更新；固定 checkpoint，不热替换成更高编号。
  - 窗口内显示 `TRAINING CHECKPOINT REPLAY`、文件名/SHA、独立 Windows rollout 语义、step/command。
  - 默认 90 秒、最多 300 秒；没有 PPO/optimizer 调用，`learning_updates=0`。
- `scripts/windows_native/start_checkpoint_replay.ps1`
  - 默认只打印计划和当前 free RAM；`-Launch` 为显式启动开关。
  - 使用已有 `run_bounded.ps1`，生命周期为 replay 秒数 + 600 秒初始化/关闭余量。
  - 要求新实验代码副本位于 `D:\isaac60-native\experiments\...\isaac_wheeled_rl_train`，不访问 frozen WSL 源码作为可写代码根。

仍需主 agent 运行后验证：Windows API/DLL 兼容、实际七 body STL 显示与关节运动、原生 PhysX 有效 CUDA view、Linux 收到持续 V40 视频，以及进程退出。当前通过 Python 语法/`--help` 和 Windows PowerShell AST 解析检查；PowerShell 仅解析文本，没有执行启动脚本。没有把“脚本写完”标为回放成功。

## 主 agent 协调后的执行步骤

1. 等容量 probe 结束，确认 Windows/WSL/GPU 余量；不要停止两个既有 watcher 或改 PID 446292。
2. 从固定 `6ed8340...` 提取完整代码副本（含 assets），仅放入新的 Windows 实验目录；代码目录名仍为 `isaac_wheeled_rl_train`。可用 `git archive --prefix=isaac_wheeled_rl_train/`，不改正式 repo。
3. 把选定稳定的 `model_N.pt`、`run_manifest.json`、`contract.json`、`source_hashes.json`、`asset_manifest.json` 复制到实验目录的 `input/`，复核 source/copy SHA。不要把它命名为 Round3 final。
4. 将两个新脚本复制到 `D:\isaac60-native\tools`，先计划/preflight，再由协调 agent 显式启动。示例参数（本轮未执行）：

```powershell
$E='D:\isaac60-native\experiments\replay-model1000-UNIQUE'
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File D:/isaac60-native/tools/start_checkpoint_replay.ps1 `
  -Repo "$E\isaac_wheeled_rl_train" -Checkpoint "$E\input\model_1000.pt" `
  -CheckpointSha256 15d878dd5b796866f0e565d0c3e53eef6d1adfdc0ee53afc24ff9a9be14e70e0 `
  -SourceRun /home/kaiser/robot-rl-sim60/round3-runs/train-formal01/train `
  -RunName replay-model1000-UNIQUE -Seconds 90
# 资源协调、输入复制与预检确认后才追加 -Launch；通过独立有界 tmux/controller 执行。
```

Python 预检示例：通过 bundled `python.bat` 调用 `checkpoint_replay.py`，提供相同 repo/checkpoint/SHA/source-run，另提供新的 `--output` 目录；省略 `--launch`。该预检不是 CUDA/NMS 测试，也不保证所有原生模块运行通过。

5. 原生 WebRTC 路径继续复用 `KAISER_NATIVE_GUI.md` 中已验证的组合：**TCP 49100 的 SSH forwarding + `media_tunnel.py local` 到 Windows 原生 UDP endpoint + Linux 官方 client / `connect_client.py`**。普通 `ssh -L` 不承载 UDP，不可省略媒体隧道。新 replay 配置广告地址 `127.0.0.1`，它指向 Linux 隧道入口，不代表渲染发生在 Linux。
6. 以 Linux 解码视频、真实 V40 七 body 场景/关节变化、窗口标签和 Windows replay-state 回执共同验收。原生 GUI 图像不能用 Cube 或 Viser 补位。

## 若要“同一训练状态”实时 GUI

需下一次经过主 agent 审查的训练入口显式带 pose publisher：在现有 PhysX 数据刷新后，仅选一个 env，墙钟限频采样 body-link world poses，并带 run ID、seq、sim-time、wall-time、canonical geometry hash。Windows Kit 只渲染这些姿态，不能再对它们启动第二个物理解算器；界面标明 WSL 解算 / Windows 原生 renderer。

这条桥目前没有接入正在运行的正式任务。不能通过读取 checkpoint、TensorBoard 标量、离线轨迹或后台重新模拟来假称当前训练的同一物理状态。本轮不补丁注入、暂停或重启正式进程；回传与评估 watcher 继续按原计划运行。
