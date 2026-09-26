# 本机Isaac Sim：V5 ONNX键盘试玩

入口：`scripts/play_v5_grounded.py`；键盘指令组件：`chassis/teleop.py`。
这是本机独立物理仿真，使用回收的ONNX控制六个真实主动关节，不是训练位姿镜像。
actor输入、动作顺序、缩放和机械资产身份均由配套合同校验。

## 已验证模型与启动

已验证的12486候选模型：

`reports/v5_speed_batches_20260922/artifacts/train/stage_00_flat/block_016/policy.onnx`

SHA-256：`ae58b862be5547195d8c4b3e71aa9be37b147792ebc903c68f032f341d92be6d`。
该模型导出通过，但整个flat评测未全过；界面明确标记为candidate。

在仓库根目录运行，输出目录须为新目录：

```bash
OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
TMPDIR="$HOME/.cache/kit-tmp" LD_LIBRARY_PATH="$HOME/.local/lib/compat" \
"$HOME/isaacsim60-venv/bin/python" -B scripts/play_v5_grounded.py \
  --onnx reports/v5_speed_batches_20260922/artifacts/train/stage_00_flat/block_016/policy.onnx \
  --device cpu --floor-boxes --output reports/v5_play_new
```

CPU单环境物理＋GPU绘图更适合本机交互；ONNX使用CPUExecutionProvider。
`--floor-boxes`使用等高、同摩擦的平地拼块，避免当前SDK的CPU三角网格材质查询告警。
它改变了试玩地面的表示，因此该测试不等同于原GPU训练场景的逐位复现。
需要原GPU物理与mesh地面时，使用`--device cuda:0`并省略`--floor-boxes`。

## 操作

| 按键 | 功能 |
|---|---|
| W / S | 前进 / 后退；按住持续下发，松开按配置斜率回零 |
| A / D | 正 / 反向yaw |
| Q / E，或T / G | 升高 / 降低高度，范围0.29–0.32m |
| Space | 清零移动指令 |
| J / K | 请求6cm／10cm跳跃；要求配套合同含对应jump技能 |
| P | 应用层暂停/继续 |
| R | 复位到固定初态，并恢复0.305m高度目标 |

点击视口取得键盘焦点。面板可调整键盘速度上限，默认0.5m/s、1rad/s；当前范围至3m/s和2圈/s。
组合指令仍经过轮速与横向加速度包络。关闭“Keyboard commands”可使用面板直接下发指令。
Kit工具栏的Stop会结束此次播放；暂停和复位优先使用P/R。

## 卡顿与输入修复

- 字符事件`CHAR`的payload是文本，不能访问`.name`；只处理KEY_PRESS/RELEASE/REPEAT。
- 主循环检查Kit时间线状态，避免在停止状态进入阻塞的物理步等待。
- 图形更新约30Hz、力矩箭头/文本约10Hz；物理与策略时基仍为200/50Hz。
- 单帧观测在新命令写入后刷新，避免无意增加一帧指令延迟。
- 播放移除了训练episode时长上限，真实越界/机械终止仍会暂停以保留现场。

实际系统键盘事件已验证W/S、A/D、Space、Q与R的命令路径。该交互检查不是各技能稳定性验收。
画面展示的膝角、气簧力和接触力属于仿真诊断，不意味着actor使用了这些不可直接测量的实机信号。

## 记录

图表写入输出目录的TensorBoard event，tag为`Playback/*`：目标/实际高度、vx、yaw、推理耗时、闭链残差。
`runtime.json`是原子更新的实时状态；`report.json`记录模型身份与退出结果；`viewport.png`是视口截图。
发生停顿时先查`runtime.json`的时间和`timeline_playing`，再查对应用户服务日志。

## 2026-09-26：13875更新快照

已回收`models/v59_candidate_13875_20260926/`：mixed_robust段1250更新，累计13875。
使用`--asset-directory model/纯底盘_v5_232mm/urdf`定位同SHA的机械归档，不修改原ONNX合同字节。
本快照高度范围为0.23–0.43m，以面板实际合同显示为准。

J/K锁存现有jump上下文，并从当前物理姿态启动任务时钟；不施加助跳力、不写入速度或位姿。
面板显示真实phase、轮间隙和腾空时间。一次请求结束／超时后会暂停，R复位后可重试。
当前GUI服务：`v59-latest-wasd-13875.service`；回执目录：`reports/v59_policy_13875_wasd_20260926/`。

单次headless 6cm请求记录：`reports/v59_policy_13875_jump06_20260926/report.json`。
请求确实进入actor与TAKEOFF，但未有效腾空，最终takeoff_timeout；这不是已学会跳跃的候选。
