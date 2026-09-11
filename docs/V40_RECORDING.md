# V40 final-policy 有界视频回放

`scripts/record_v40.py` 在指定源码树中重新运行一个 checkpoint 的确定性 mean actor，输出 `replay.mp4`、`final_frame.png` 和 `summary.json`。这是**新执行的 checkpoint 回放**；`recording_ok` 只表示录制成功，不是策略评估通过。

## 源码与物理版本

默认导入录制脚本所在仓库的 `scripts/train_v40.py`，复用它的 `preflight`、`make_manifest`、`checked_checkpoint`。也可用 `--repo-root /absolute/selected-repo` 选择另一份完整源码树；入口同时选择该树的 `src`，发现进程已混入另一份任务包时直接拒绝。

实际录制必须声明 `--source-case`：

- `source-final`：旧服务器 source-final 源码、旧髋部几何、物理轮关节 ±π 限位下的回放。
- `fixed-repo`：修复后仓库的物理模型回放，**不能标成原始训练过程的视频**。即使用的是旧权重，画面也属于修复后的模型。

此标签由操作者声明，不自动证明物理版本。摘要保存所选仓库绝对路径、Git HEAD（无 Git 历史时为 null）、全部 `src/**/*.py` 与 sibling `train_v40.py` 的逐文件 SHA256、源文件集合 SHA256、录制入口 SHA256，以及 checkpoint / run manifest / contract / asset manifest 的 SHA256。未提交或未跟踪的 V40 源文件也计入哈希；checkpoint 与源码在环境启动后再次核对。

选择旧树会保留旧树实际物理实现及其原有准入门禁。目标树必须已有 V40 的固定命令与 pre-reset snapshot API；缺失 API、资产证据失败、checkpoint 身份不匹配都会拒绝执行。

## 命令

以下命令是交给现有 `bounded_worker_v2` 在独立 tmux worker 中执行的载荷。仿真时长上限不等于启动、着色器编译、编码所需的墙钟时间；外层 worker 继续负责墙钟限制。

帮助与只读预检（不启动 Sim、不创建输出，Python 标准库即可执行）：

```bash
python scripts/record_v40.py --help
python scripts/record_v40.py --preflight-only --research \
  --repo-root /absolute/selected-repo \
  --source-case source-final --stage stand \
  --checkpoint /absolute/original-run/model_final.pt \
  --output-dir /absolute/new-recording-directory
```

预检复用所选仓库的全部 contract / asset / collision / 研究授权 / runtime 版本 / IsaacLab 源码身份门禁，额外查录制依赖元数据、命令范围与路径。为保证 stdlib-only，预检**不加载权重，不导入 imageio，也不执行 FFmpeg**；摘要中的 `checkpoint_status` 明确这一点。实际 apply 在启动 Sim 前执行原有 `checked_checkpoint` 和 FFmpeg 可执行性检查。缺依赖或门禁未通过，预检退出 2。

旧 source-final 回放：

```bash
python /absolute/tools/record_v40.py \
  --repo-root /absolute/immutable-source-final-repo \
  --apply --research --source-case source-final \
  --stage stand --seed 40 --device cuda:0 \
  --checkpoint /absolute/original-run/model_final.pt \
  --output-dir /absolute/new-source-final-video \
  --duration-s 10 --command 0 0 0.32
```

修复模型回放（输出目录与来源标签独立）：

```bash
python scripts/record_v40.py \
  --repo-root /absolute/fixed-repo \
  --apply --research --source-case fixed-repo \
  --stage stand --seed 40 --device cuda:0 \
  --checkpoint /absolute/compatible-run/model_final.pt \
  --output-dir /absolute/new-fixed-physics-video \
  --duration-s 10 --command 0 0 0.32
```

checkpoint 必须邻接其原始 `run_manifest.json`，且 stage、contract、asset 身份与所选仓库一致；不能修改 manifest 来绕过身份校验。已存在的输出目录（包括符号链接）一律拒绝。

## 录制行为与 IsaacLab 2.3 API

- 一个环境，默认最长 10 秒；允许 `(0,10]` 内 `.01s` 整数倍，最多 1000 policy ticks。physics dt `.005s`、decimation 2、policy 100Hz 均沿用原 contract。
- `set_evaluation_command` 先于 wrapper 构造与 reset；使用官方 `RslRlVecEnvWrapper`，`reset()` 返回 `(TensorDict, extras)`，`get_observations()` 返回 `TensorDict`，actor 输入为 `obs["policy"]`。执行 `.eval()` / `torch.inference_mode()`，不采样探索动作。
- 环境配置复制 sibling `make_env` 的 contract/research/stage/seed/num_envs/device 字段，再设置 viewer 与 `V40Env(cfg=cfg, render_mode="rgb_array")`；MDP 逻辑由所选原环境实现。
- 显式 `AppLauncher(headless=True, enable_cameras=True, livestream=0)`；覆盖继承的 `ENABLE_CAMERAS` / `LIVESTREAM`，root 时加 Kit `--allow-root`。固定相机 960×720，eye `(1.0,-1.4,.8)`，look-at `(0,0,.3)`，相对 env 0 原点。
- RGB 产品创建后，将 `/isaaclab/render/rtx_sensors` 标记为 true，使用原生 step 的 render scheduling。warmup 最多 20 次 render，不推进物理。
- `rerender_on_reset=False`；每 tick 的最后 physics substep 先渲染，然后才 done/reward/auto-reset。`render(recompute=False)` 在 RTX 标记开启时读取缓存，不渲染 reset 后姿态。首次 terminated 或 timeout 就停止，终止原因读取 `get_evaluation_snapshot()`。
- 初始帧 + 每 4 ticks 一帧；终止或到达上限时额外保留最终帧（已经采样则不重复）。逐帧 `imageio.get_writer(..., format="FFMPEG", fps=25).append_data()`，只持有常数个 RGB 数组，不收集整段视频数组；独立 PNG 保存最终缓存帧。
- MP4 为 25FPS 的 H.264 / yuv420p。由于含初始帧及非采样网格末帧，**媒体时长与精确仿真时长不同**：完整 1000 ticks 是 10.00s 仿真、251 帧、10.04s MP4。摘要分别保存 `sim_duration_s`、`encoded_duration_s` 与逐帧 `frame_policy_ticks`；后者乘 `.01` 即真实采样时间。
- 复用已有 snapshot 校验；非有限 observation / action / reward / state、nonfinite 终止标志、时钟或 done 不一致均失败。物理跌倒但有限状态可成功录制，同时保留真实终止原因，不赋予策略通过结论。
- writer、环境都在 `finally` 中关闭；encoder trailer、最终 PNG、视频哈希与成功/失败摘要**先于 `app.close()` 落盘**。默认 fast shutdown 可能直接以 0 退出解释器，因此 worker 必须检查 `summary.json.recording_ok` 及产物。摘要明确 app 清理尚未观测。

本地只读核对路径：`../.deployment_sources/IsaacLab-v2.3.0/`：

- `source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`：66–67 constructor reset；139–165 reset/get_observations/step 返回类型。
- `source/isaaclab/isaaclab/envs/direct_rl_env.py`：349–397 physics-render / done / auto-reset 顺序；419–477 RGB 缓存读取与 lazy annotator。
- `source/isaaclab/isaaclab/app/app_launcher.py`：842–855 offscreen 与初始 `rtx_sensors=False`。

## 依赖与验证

真实回放需要 sibling train 的全部目标运行时（包括 Python 3.11、IsaacLab v2.3.0、Isaac Sim 5.1.0、对应 torch/RSL），额外需已存在的 `imageio`、`imageio-ffmpeg`、NumPy、Pillow 和可执行 FFmpeg（含 libx264）。入口只查元数据并调用现有编码器；找不到会报告错误，不下载、不安装。可使用 imageio-ffmpeg 原有的 `IMAGEIO_FFMPEG_EXE` 指定已有二进制。

```bash
PYTHONPATH=src python -m pytest tests/v40/test_record.py -q
```

测试用假环境复现 render-before-auto-reset、TensorDict 返回路径、首终止停止、精确时长、帧采样、非有限状态拒绝、关闭失败以及 `app.close()` 不返回前的落盘顺序。还用本机现有 imageio/FFmpeg 实际流式编码并解码 960×720 / 25FPS 测试视频。测试视频是合成像素，不是机器人训练结果；真实 Isaac 渲染须由目标 worker 验证。
