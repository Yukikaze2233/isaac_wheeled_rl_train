# Kaiser 原生 Isaac Sim/Lab GUI → Yukikaze Linux：实际部署与验证状态

## 当前结论

**用户已授权 Windows 原生安装。Windows Sim 6 已安装且完整原生 GUI 曾在 Yukikaze Linux 实测收到；官方 Lab core 与训练依赖安装已完成，但安装后的 Lab 运行、原生 Windows PPO/V40 训练尚未验证。**

2026-09-13 **14:37:47 北京时间**通过有效 SSH 会话重新只读核对。工具会话中止没有撤销此前已经完成的安装，不能把状态恢复成“尚未下载”。

| 项目 | 真实状态 |
|---|---|
| Windows Sim | `D:\isaac60-native\sim` 已安装；官方 ZIP MD5 与解压文件校验通过 |
| 原生 GUI → Linux | **已实测**完整菜单、Stage tree、Property、RTX viewport、连续视频，以及鼠标选择/相机回传 |
| 实际网络路径 | SSH 转发 TCP signaling + SSH 承载原始 UDP 媒体；远端 UDP 半段为 Windows 原生 Python，Kit 负责渲染/编码，Linux 官方客户端负责解码 |
| Lab 源码 | `D:\isaac60-native\source\IsaacLab`；指定 tag/commit，当前 tracked tree 无改动 |
| Lab 安装 | `lab-install2` 在 **01:45:28** 退出 0；今日 CPU 元数据再次确认 editable 包和版本存在 |
| 原生 Lab/PPO/V40 运行 | **未完成验证**；未发现 Windows 训练输出、checkpoint 或 completion 回执 |
| 当前进程/端口 | 部署目录关联 Windows 进程为空；TCP 49100、UDP 47998 无监听；相关 Linux client/media 进程也为空 |
| 当前服务 | 已停止，没有正在运行的 GUI 连接；本次只读复查未启动任何服务 |

**GUI 成功发生在 Lab 安装之前。**不能将它外推为安装 Lab 之后仍通过 CUDA/NMS、Kit 兼容性或 PPO 验证，也不能将 `gui-state.json` 的渲染帧数解释为训练步数。诊断场景是脚本运动的 Cube，回执明确标为 `not training`。

本次检查的 Windows 训练输出位置包括部署根目录的 `logs/runs/experiments/outputs/checkpoints`、`source/IsaacLab/logs/runs/outputs` 和 `audits`。前述运行目录均不存在，audits 中未发现 `.pt/.pth/.ckpt/.onnx`、训练 `completion.json` 或 TensorBoard event 文件。没有扫描私人物件，也没有把 Sim 自带示例模型和缓存权重当作训练产物。

主 agent 已独立核对 WSL：既有结果是 3-update 工程 smoke 和 Round2 五 height 评估，没有 Round3 正式产物。本节的 Windows GUI/安装回执不构成额外训练结果。

## 已有 GUI 成功判据及限制

- 官方 Linux client 2.0.0 的实际视频为 **1280×720**；一次采样中播放器帧数由 28 增至 152，WebRTC inbound 视频解码计数为 150，统计 FPS 为 60。
- 两张不同时间截图显示真实 Kit 的 File/Edit/Create 等菜单、Stage tree、Property 面板、RTX viewport 和远端 RTX 4090 统计。
- 从 Linux 打开原生 File 菜单，并点击 Stage 中的 `NativeGuiProbe`；Windows USD selection 从空变为 `/World/NativeGuiProbe`。
- 从 Linux 右键拖动，Windows 相机矩阵最大元素变化 **0.3055999909929723**；对应 Windows Kit PID 为 **31828**。
- 不只检查了端口或 `app ready`，也没有使用 Viser 替代这些画面。
- **60 FPS 是接收/解码统计，不是已证明的屏幕呈现帧率。**同次播放器样本 `frames=152, dropped=141`，存在大量呈现丢帧；后台/聚焦状态和流畅性问题未完成复测，不能宣称稳定流畅 60 FPS。

Linux 证据：`/tmp/opencode/native-stream04-client/{video.json,first.png,continuous.png}`、`/tmp/opencode/native-input01/{input-proof.json,file-menu.png,selected-prim.png,camera-after.png}`。本次核对回执为 `/tmp/opencode/native-recheck-20260913.json`。

## 安装及进程退出时间

以下均为 **2026-09-13 北京时间（UTC+8）**，来自现有回执；没有把 timeout 或人工中断改记为正常成功。

| 任务 | 开始 → 结束 | 实际退出/判据 |
|---|---|---|
| ZIP 分块下载 | 00:27:36 开始 | artifact 回执 `verified`，MD5/SHA-256 已记录；外层 finished/exit 文件缺失，不补造退出码 |
| 初次解压 controller | 00:34:27 → 00:36:30 | 退出 1；PowerShell 未取得 tar ExitCode，不能据此单独确认解压成功 |
| 独立解压校验 | 00:38:54 → 00:48:36 | 退出 0；124,631 个文件逐一校验大小/CRC，合计 19,336,044,534 bytes，发布为 `sim` |
| Lab clone | 00:49:03 → 00:49:37 | 退出 0，指定 commit 与来源验证通过 |
| stream01 | 00:49:32 → 01:03:51 | 人工终止切换网络方案，native/controller 退出 1；LAN 直连未通过 |
| stream02 | 01:03:51 → 01:23:51.844 | GUI/输入实测对应此实例；到达 1200 秒上限，`timeout=true`、native exit=1，外层于 01:23:52 退出 **124**，`process_exited=true` |
| 首次 Lab 安装 | 01:28:20 → 01:37:40 | 人工中断调整安装封装，退出 1 |
| Lab 安装第二次 | 01:36:42 → 01:45:28.362 | 退出 **0**、未超时、`process_exited=true` |

媒体隧道 remote 回执记录 `closed=true`、运行约 899.766 秒；没有绝对结束时间字段。Linux GUI client 也没有独立精确退出回执；本次确认相关进程已不存在，不将日志 mtime 推测成进程退出时间。

## 当前安装元数据（CPU 只读检查）

Python 为 bundled **3.12.13**，prefix 为 `D:\isaac60-native\sim\kit\python`。仅查询 distribution metadata，**没有 import Torch、创建 CUDA context 或启动 Kit**。

| 包 | 已安装版本 |
|---|---|
| isaaclab / assets / tasks / rl | 6.1.14 / 0.3.4 / 1.10.9 / 0.5.5 |
| isaaclab_physx | 1.1.3 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| rsl-rl-lib | 5.5.1 |
| onnx / onnxruntime | 1.23.0rc1 / 1.30.0 |
| warp-lang / numpy | 1.13.0 / 2.5.2 |
| Pillow / packaging | 12.2.0 / 23.2 |

Lab 是真实 editable 安装，`direct_url.json` 指向该 D 盘官方源码。官方 core installer 的 Torch 检查被部署封装限定为保留 2.11/cu128，避免其硬编码 2.10 降级；没有改官方 Git 源文件。安装日志有上游依赖冲突警告，**尚无最终完整依赖检查及安装后运行验证通过的证据**。

## 历史系统前置记录

部署前检测记录为 Windows 11 Pro for Workstations build 26100、Ryzen 9 7950X、32 GB 档内存、RTX 4090 / driver 610.88；Windows `vulkaninfo --summary` 退出 0，API 1.4.341。原生 uv 0.11.19 已存在，长路径已启用，D 盘 ACL 允许用户创建独立目录。随后实际完成了上文的安装和收流。

这不是今日的空闲内存、显存或负载快照；本轮没有重跑 Vulkan/GPU 探测。已有 WSL CUDA headless 环境仍与 Windows 原生安装分离。

## 官方支持与版本边界

使用固定 **6.0.0 文档**核查；当前 `/latest` 已指向后续版本，不混用其安装版本号。

- [Sim 6 系统要求](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)：支持 Windows 11、Ubuntu 22.04/24.04；最低 32 GB RAM、50 GB SSD、RTX 4080/16 GB VRAM 档，Windows 测试驱动 581.42。Kaiser 的 GPU、驱动和磁盘满足前置，内存仅在最低总量档，当前空闲量明显低于总量。
- [Sim 6 Python 安装](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)：Python 3.12，`isaacsim[all,extscache]==6.0.0.1`，NVIDIA extra index。Windows 3.11 不能直接用于该 pip 栈。
- [Sim 6 官方下载](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/download.html)：Windows standalone 6.0.0 和 Linux x86_64 WebRTC client 2.0.0 均有官方发布。
- [Sim 6 Livestream Clients](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/manual_livestream_clients.html)：Windows 服务端可用 `isaac-sim.streaming.bat`；PIP 安装可用 `isaacsim isaacsim.exp.full.streaming --no-window`。Linux 原生客户端显示远端 Isaac Sim 界面，支持菜单与交互；每个实例仅一个客户端。
- Lab 固定 `v3.0.0-beta2.patch1`，commit `ffff603eafc6b74264a5261cc0183d6a65390d78`。其源码 `docs/source/setup/installation/binaries_installation.rst` 明确说明：**binary 安装应使用 Sim bundled Python / `isaaclab.bat -p`，不再支持将 binary Sim 与 conda/uv/venv 混用**。因此优先 GUI-first standalone 路线时，不额外再安装一套 pip Sim，也不把 Hermes Python 或既有 WSL venv 拼进去。

本次没有找到可把当前 WSL 图形栈认定为官方已验证原生 RTX GUI 环境的依据；操作系统列表中的 Ubuntu 不能替代本机 WSL Vulkan 的实际失败证据。

## 已核准下载规模

### 已安装的 Windows standalone

官方 URL：

```text
https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.0.0-windows-x86_64.zip
```

| 内容 | 精确大小 |
|---|---|
| Windows ZIP | **10,668,464,877 bytes**，9.936 GiB |
| ZIP 中解压文件的逻辑总量 | **19,336,044,534 bytes**，18.008 GiB |
| 同时保留 ZIP + 解压目录 | **27.944 GiB**，未计文件系统分配、shader/cache、Lab 依赖和运行数据 |
| ZIP 条目数 | 147,069 |
| Linux x86_64 client 2.0.0 `.deb` | **97,199,072 bytes**，约 92.7 MiB |

已安装目录中确认存在 `isaac-sim.bat`、`isaac-sim.streaming.bat`、`python.bat`、`kit/python/python.exe`。Sim/Lab 使用其 bundled Python 3.12；Windows 既有 Python 3.11 仅被只读校验工具及媒体隧道复用，不作为训练环境。

前置核查读取过 ZIP 目录元数据；随后完整 Sim 和客户端都已下载、校验并解包。现有回执中的校验值：

| 文件 | 官方 MD5（已匹配） | 实际 SHA-256 |
|---|---|---|
| Windows ZIP | `4b49a4258792f09300ece31be1b6cfd9` | `7430847bb4a94124ddd0f25793e70adeba5c4846721821fea152672f09fc5850` |
| Linux client `.deb` | `07bd252432fb92b93bdb33b337455827` | `01bbd36df7b93612ee0e599c5a508f36564fc4ff4f2a4354931ecb2735134513` |

回执分别位于 D 盘 `downloads/<zip>.json` 和 Linux `~/.local/share/isaac-webrtc/<deb>.json`。今日只读取既有回执，没有重复下载或对 10 GB 文件重新做全量校验。

用户已授权 **80 GiB 工作预算**；它不是最终实测磁盘占用。本次没有重新统计整个安装树，也没有下载完整资产库五个大分卷。

### 历史 pip 发布渠道核查

Sim 的 `cp312-none-win_amd64.whl` 可获取。`isaacsim-extscache-kit` 在 PyPI 只有小型 bootstrap sdist，真正 Windows wheel 位于 NVIDIA index，不能只看 PyPI 的大小：

| 主要包 | Windows 压缩大小 |
|---|---|
| isaacsim-extscache-kit 6.0.0.1 | 5,937,962,428 bytes |
| isaacsim-extscache-kit-sdk 6.0.0.1 | 727,670,139 bytes |
| isaacsim-extscache-physics 6.0.0.1 | 252,183,834 bytes |
| torch 2.11.0+cu128 / cp312 | 2,753,189,216 bytes |
| torchvision 0.26.0+cu128 / cp312 | 9,585,013 bytes |

以上是前置渠道核查，不是实际采用 pip Sim 再安装第二套的记录。实际使用 standalone Sim，再由其 bundled Python 安装 Lab core、Torch/RSL/ONNX；完成状态及版本见上文。

## 已验证的网络方向与当前状态

**客户端是 Yukikaze 当前 Linux，而不是 Kaiser Windows 的 Edge。**

- 官方 Linux client 已私有解包到 `~/.local/share/isaac-webrtc/release-2.0.0/opt/Isaac Sim WebRTC Streaming Client/`，已实际运行并采集上述视频/输入证据。
- LAN 直连尝试失败：Windows 监听 TCP 49100，但 Linux 直连超时。仅转发 TCP 后仍出现 `Client sent STUN requests but did not receive any responses from the server`，不能算收流成功。
- 成功路径：Windows Kit 广告地址设为 `127.0.0.1`；Linux 的 TCP 49100 经 SSH 转发到 Windows loopback；Linux 的 UDP 47998 经 `media_tunnel.py` 的 SSH 字节通道送到 **Windows 原生 Python UDP endpoint**，再交给 Kit，回程同理。
- 因而客户端输入的 **Linux `127.0.0.1` 是隧道入口**，本体仍是 Kaiser Windows 的 Kit。WSL 只承载 SSH/进程控制，不解算或重绘这些 GUI 画面；没有改用 Viser，也没有改防火墙。
- 普通 `ssh -L` 不承载 UDP；必须配合已实现的媒体隧道。它会引入 TCP 队头阻塞等开销，性能未完整验收。
- 当前 client、媒体隧道及 Windows Kit 均已结束，没有在运行的连接。原生桌面 client 不提供 HTTP GUI URL；8210 是文档中的 Docker Compose web viewer，8088 是另一项 Viser 功能，均不是本方案入口。

## 可复用入口现状（本次未执行）

已有工具位于本地 `scripts/windows_native/` 和 D 盘 `tools/`，不是一套已经验证完的原生训练启动器：

| 入口 | 作用与验证边界 |
|---|---|
| `start_stream.ps1` | 启动 Windows 原生 Kit，支持 `-ServerAddress`、唯一 `-RunName`、`-Seconds`、`-Probe`；`-Probe` 仅为 Cube GUI 诊断，不训练 |
| `run_bounded.ps1` / `job.sh` | Windows 原生进程树及 WSL tmux 外层有界控制、日志与退出回执 |
| `media_tunnel.py local ...` | Linux 媒体中继；mode 是**位置参数**，`--audit` 是 **JSON 文件路径**，不是目录；当前已停止 |
| `client_cdp.py` / `connect_client.py` | 连接专属 Linux 官方客户端、验证解码视频；需先运行客户端并提供本地 CDP 9229 |
| `verify_gui_input.py` | 已验证鼠标选择和相机回传；用于自有诊断实例 |
| `clone_lab.ps1` / `install_lab*.py,ps1` | 源码和依赖已安装，今日不重跑 |
| `inspect_runtime.py` | 支持 CPU 元数据和可选 CUDA 检查；今日没有执行 CUDA 检查 |

后续仅在用户要求恢复 GUI 时，可按既有流程启动，例如 Windows 原生入口参数：

```powershell
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass `
  -File D:/isaac60-native/tools/start_stream.ps1 `
  -ServerAddress 127.0.0.1 -RunName manual-UNIQUE -Seconds 900 -Probe
```

该 Windows 命令应继续通过独立、有界 tmux/controller 管理。Linux 端需要三部分，以下是参数参考而非本轮执行记录：

```bash
# 使用当次已经验证有效的 ControlMaster，不新建认证。
CONTROL=/path/to/current-control-socket
HOST='<USER>@<KAISER_HOST>'
PORT='<SSH_PORT>'
ssh -S "$CONTROL" -o BatchMode=yes -p "$PORT" -O forward \
  -L 127.0.0.1:49100:127.0.0.1:49100 "$HOST"

# 在另一个有界 tmux 中运行；程序自身也有有限生命周期。
python scripts/windows_native/media_tunnel.py local --seconds 900 \
  --host "$HOST" --ssh-port "$PORT" --control-path "$CONTROL" \
  --audit /tmp/opencode/native-media-UNIQUE.json \
  --remote-audit D:/isaac60-native/audits/native-media-UNIQUE.json
```

官方客户端 executable 为 `~/.local/share/isaac-webrtc/release-2.0.0/opt/Isaac Sim WebRTC Streaming Client/isaacsim-webrtc-streaming-client`；此前使用独立 `--user-data-dir`、`--remote-debugging-port=9229` 运行，再以 `connect_client.py --server 127.0.0.1 --output <新证据目录>` 连接。**尚没有经验证的安装后“单命令启动原生 Lab/V40 训练显示”入口**，不要把以上 renderer 启动器称作训练命令。

本次有效 ControlMaster 为 `/tmp/opencode/kaiser-2222-control-current`，旧 socket 已失效；它属于当前会话，不应作为长期固定配置。

## 未完成事项

1. Lab 安装后的完整依赖检查、CUDA/NMS 和 Kit 共存验证。
2. 原生 Windows Lab 环境步进、PPO 与训练时 GUI 显示；本目录没有相应成功回执或 checkpoint。
3. Windows V40 启动兼容及 binary runtime 身份：`os.geteuid`、signals、目录 fsync、导出环境和无 `isaacsim` pip metadata 等差异尚未完成适配验证。
4. WSL 训练状态送 Windows 原生 renderer 的备选桥没有完成；现有媒体隧道只转发视频/输入报文，不是物理状态桥。
5. LAN 直连、前台流畅呈现及更长时间稳定性尚未通过完整测试。

安装授权已经给出，不再以“等待安装授权”列为阻塞。当前用户要求只读检查训练结果，因此以上事项保留给后续决定，本轮不启动验证负载。

## 本轮产物与阻塞

本次仅更新本文件；`scripts/windows_native/` 是此前部署工作的遗留工具，本次未改。独立前置核查工具和原始证据包括：

- `kaiser-native-inventory.ps1`、`kaiser-native-probe.py`、`kaiser-native-inventory.json`：Windows 资源、权限、Python/uv、真实 Vulkan 调用。
- `kaiser-native-public-metadata.py`、`kaiser-native-wheel-metadata.json`、`kaiser-native-doc-*.txt`：固定 6.0 文档和 Windows wheel 元数据。
- `kaiser-native-download-sizes.py`、`kaiser-native-download-sizes.json`、`kaiser-native-zip-audit.py`、`kaiser-native-zip-audit.json`：精确下载/展开规模；首次 urllib range 请求发生 TLS EOF 后，改用受大小限制的 curl range 成功读取目录。
- `kaiser-native-plan.ps1`：**preview-only** 非破坏计划，运行也不创建目录、不下载、不修改系统配置。

当前重新核对的完整只读回执为 `/tmp/opencode/native-recheck-20260913.json`。D 盘 `audits/stream01`、`stream02`、`lab-install2` 等目录保存已执行工作的回执。没有因工具中止而删除这些成果，也没有把失败/缺失的退出回执补记成成功。本次未安装、未启动训练或 GPU 仿真，未修改训练源、驱动、防火墙或认证配置，未提交/推送。
