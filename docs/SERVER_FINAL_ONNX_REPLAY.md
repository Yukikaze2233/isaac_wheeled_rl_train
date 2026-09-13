# 服务器最终 ONNX：本机 Sim 6 原生回放

## 诊断修正与固定高度 headless 评估（2026-09-12）

旧回放的 `diagnostic_frames` 错把任务 `termination_flags` 当成原始诊断。
v2 对非轮接触、膝限位、低高度等终止原因置零，因此**旧 summary 的诊断全零不能证明这些物理异常没有发生**。
下文历史计数保留为历史记录；应以原始遥测重审，不能把旧计数用于新验收。

### Snapshot API

`env.get_evaluation_snapshot()` 返回上一 policy tick 的 **pre-reset 深拷贝**；
新增 `diagnostic_flags_version=1`、`diagnostic_flags` 和 `sustained_failure_ticks`。
原有 `schema_version=1`、`termination_flags`、`terminated`、`timeout` 含义保持不变。

| `diagnostic_flags` 字段 | 含义 |
|---|---|
| `nonfinite` | 当前 root pose/velocity、关节位置/速度、gravity、接触力、力矩、clearance 非有限，或本次动作已标无效；不借用可能过期的 `_finite_state` |
| `non_wheel_contact` | 非轮刚体净力 history 峰值超过合同阈值；**不是 ground pair** |
| `knee_limit` | 实测膝位置超出硬界及原 tolerance |
| `instantaneous_tilt` | gravity 推算的瞬时倾角超过合同 `max_tilt_deg`（当前35°） |
| `low_height` | base-link 高度低于合同阈值 |
| `base_visual_bounds_ground` | base 视觉 AABB 保守净空≤0；不是精确网格接触 |
| `failure_gravity` | v2 当前 `gravity_z > failure_gravity_z`（当前-0.1）；尚未要求持续时间 |
| `sustained_failure` | v2 **实际累计计数**已超过 `failure_seconds/policy_dt`；当前超过100步才为真 |

v1 没有持续失稳机制，最后两项为 false、计数为0；v1 `termination_flags.tilt` 是即时条件，
v2 `termination_flags.tilt` 仍是持续失稳。读取 snapshot 不推进失稳计数，不改变奖励或终止逻辑。

`capture_evaluation_initial_snapshot()` 继续返回 `sample_kind=initial`、新增 `state_phase=post_reset`；
在 evaluation reset 后从当前状态重新计算诊断，`terminated/timeout=false`、`termination_flags={}`，
因为这不是已执行 transition 的终止决策。它不替换缓存的 terminal pre-reset snapshot。
相同 tick 再次调用 `_get_dones()` 也不会覆盖已有 pre-reset 快照。接触字段是当时 sensor buffer 的读数；
post-reset 若传感器尚未刷新，不应将该初始读数当成新回合已经发生的碰撞。

### Summary / CSV

- `summary.diagnostic_frames` 逐项累计真实 `diagnostic_flags`，`diagnostic_frame_count` 为计入的 pre-reset policy帧数。
- `summary.termination_flags` 只累计 terminated 帧上的原任务原因；timeout另计。两组计数不互代。
- `telemetry.csv` 添加 `diagnostic_<字段>`、`termination_<原原因>`、`sample_kind`、`episode_step`、
  `episode_time_s`、`sustained_failure_ticks`、`non_wheel_net_force_max_n`。
- 旧 `non_wheel_contact_n` 列保留为 `non_wheel_net_force_max_n` 的同值兼容列，两者都不表示对地pair力。
  连续多个frame可能来自同一接触及history保留，不等于独立碰撞次数。
- 遇到 nonfinite 物理帧，先记录该帧诊断/CSV，再退出失败；summary标记 `all_numeric_finite=false`，
  非有限latest position/velocity写为null以保持合法JSON，原始CSV保留NaN/Inf证据。

### Headless 固定高度命令

工作目录为本工作树。每次使用不存在的新 report 路径，父目录须已存在：

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
  TMPDIR=/home/yukikaze/.cache/kit-tmp \
  LD_LIBRARY_PATH=/home/yukikaze/.local/lib/compat \
  /home/yukikaze/isaacsim60-venv/bin/python -u scripts/play_v40_onnx.py \
  --research --headless --num-envs 1 --seed 42 --device cuda:0 \
  --onnx ../reports/current/cqa1_round2_20260912/final/policy.onnx \
  --contract ../reports/current/cqa1_round2_20260912/final/contract.json \
  --command 0 0 0.28 --max-steps 6000 --max-wall-seconds 580 \
  --report-dir "../reports/server_final_headless_h028_$(date +%Y%m%dT%H%M%S)"
```

分别替换为0.30/0.32及新目录即可做中/高位诊断。非键盘固定命令路径在所有自动reset后保留同一命令；
观测无噪声、reset根速度0、原始确定性ONNX输出保持已有行为。
`--headless --keyboard`仍拒绝；headless无viewport导入/键盘订阅/GUI pacing，
ready标记为 `SERVER_FINAL_ONNX_HEADLESS_READY`，截图状态为 `disabled_headless`。
原合同20秒timeout不变，6000步表示总计60仿真秒，**不是无reset连续站立60秒**。
本入口沿用本工作树Sim6 preflight；远端应在主agent确认的目标栈上运行，不绕过版本/资产检查。

本次只运行CPU相关测试：`tests/test_onnx_replay.py`、`tests/v40/test_snapshot_diagnostics.py`、
`tests/v40/test_round2.py`、`tests/v40/test_unified_commands.py`，共 **83 passed**，3条既有弃用提示。
覆盖真实小型ORT图＋CSV入口，以及从生产源码提取的snapshot/done/reset方法；未启动物理仿真。

## 可调速度上限：5 m/s / 120 rpm（探索模式）

新增 `--exploratory --max-linear-speed 5 --max-angular-rpm 120`。
**这设置的是可选上限，启动档位仍是 `1 m/s`、`1 rad/s ≈ 9.5493 rpm`，初始命令仍为零。**
`120 rpm = 120 × 2π / 60 = 12.566370614359172 rad/s`，表示机器人基座的目标 yaw 角速度，
不是电机或车轮转速。A/D 是转向；“平移”对应机体前后 `vx`，没有新增横移命令。

保存契约的训练域仍为 `vx ∈ [-2, 2] m/s`、`wz ∈ [-2, 2] rad/s`，高度仍限于 `[0.28, 0.32] m`。
超过训练速度范围是 **EXPLORATORY / OOD（训练分布外）命令探索**，不代表该策略已训练覆盖这些速度，
也不保证实际速度达到所选命令。模型、原始契约、资产身份检查和 ONNX 原始输出保持原样，
使用本地 Sim6 的跨版本回放，不能视为服务器原生环境复现。

### 调速键位

| 按键 | 动作 |
| --- | --- |
| `-` / `=`（`+` 所在键） | 线速度档位每次 −/+ `0.5 m/s`；小键盘 −/+ 同样有效 |
| `[` / `]` | 角速度档位每次 −/+ `10 rpm`（约 `1.0472 rad/s`） |
| WASD | 按当前档位前后移动／左右转向，允许组合 |
| Space | 清空 held keys，速度命令归零；保留速度档位和高度 |
| Q/E | 高度每次 ±0.01 m，原范围不变 |

调速键每次按下只改变一次，repeat 不累积；档位在零和已配置上限间限幅。
按住 WASD 时调速同样只排队新命令，在下一观测边界生效。
普通模式默认上限来自保存契约（此模型均为2），调速不会越界；普通模式指定更大上限会拒绝。
探索模式允许显式有限正上限；非有限值、零／负上限，以及超出原高度范围均拒绝。
启动 `--linear-speed` / `--angular-speed` 仍须在训练域和所选上限内，较高速度在窗口中选取。

窗口标题和 `TELEOP_STATUS` 控制台输出显示：模式、requested 命令的 m/s 与 rad/s/rpm、
当前档位／上限，以及 **requested OOD** 和档位 OOD 状态。
即使探索模式下当前请求为零，也明确显示 `EXPLORATORY` 与 `IN-TRAINING-RANGE`。

### 菜单调用的精确命令

工作目录可设为本 Sim6 工作树；菜单负责启动新窗口。每次使用新 report 路径：

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
  TMPDIR=/home/yukikaze/.cache/kit-tmp \
  LD_LIBRARY_PATH=/home/yukikaze/.local/lib/compat \
  /home/yukikaze/isaacsim60-venv/bin/python -u \
  /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train-60/scripts/play_v40_onnx.py \
  --research --keyboard --exploratory \
  --max-linear-speed 5 --max-angular-rpm 120 \
  --linear-speed 1 --angular-speed 1 --command 0 0 0.30 \
  --onnx /home/yukikaze/Documents/workspace/robot_rl/reports/current/cqa1_round2_20260912/final/policy.onnx \
  --contract /home/yukikaze/Documents/workspace/robot_rl/reports/current/cqa1_round2_20260912/final/contract.json \
  --report-dir "/home/yukikaze/Documents/workspace/robot_rl/reports/server_final_onnx_exploratory_$(date +%Y%m%dT%H%M%S)" \
  --max-steps 6000 --max-wall-seconds 580
```

上限／探索开关只适用于 `--keyboard`。附加 `--preflight-only` 可以不启动 Sim，
检查完整原模型身份及所选上限；已用实际服务器最终模型成功执行该检查。

### 探索日志与验证范围

- `summary.json` 增加 `exploratory`、`replay_domain`、`teleoperation` 和
  `out_of_training_domain_command_frames`。`teleoperation` 保存原训练域、配置上限、当前档位和单位。
- OOD 帧按**已完成物理步的 action/reward 命令**计数，不把键盘 repeat 或待应用请求计成物理帧。
- 新增 `command_changes.csv`：初始值、实际命令变化与正常结束行，包含 policy tick、m/s、rad/s、rpm、
  探索标记、下一命令 OOD 标记、截至该步的累计 OOD action 帧数。结束行给出最终累计值。
- 原有 `command_events.csv` 继续保留；summary 的事件详情包含探索／OOD标记和角速度rpm。
  全步 `telemetry.csv` 也包含 action/next 的 OOD 标记及累计计数。
- 本轮66项相关测试通过，其中本文件45项。覆盖120rpm换算、普通模式拒绝越域、有限上限、
  键盘限幅、低速启动、repeat、stop、原高度限制、原契约字节不变、历史／reward／自动重置对齐。
  4步CPU入口测试实际运行小型fixture ONNX、真实core方法与CSV输出，验证最后一个旧命令reward仍计入OOD帧。
- 高速120rpm的float32舍入差会超过旧的绝对 `1e-7` 容差；现在按原契约的float32缩放／clip编码
  精确比对 actor 最新命令字段，reward snapshot则与float32命令精确比对，没有放宽时序检查。
- **未执行5 m/s或120rpm的长时物理验证，没有高速策略质量结论。** 下方已有真实GUI移动数据属于原有低速验证。
- 本轮另运行一次训练域内短时GUI检查，证据为
  `/tmp/opencode/v40_speed_keys_gui_20260912T2031/` 与同前缀 `.log`：
  原生合成 `EQUAL` 事件将档位从1.0调至1.5 m/s；tick10应用W的1.5 m/s命令，
  repeat没有继续增速；tick20原生失焦清空held keys，tick21应用零命令。
  窗口标题／控制台实际更新了档位、命令和单位。检查在29步后因临时审计入口要求焦点而中止，
  summary如实保留 `failed`，并记录重置0、OOD帧0、数值有限、订阅成功清理。
  这不是完整125步通过记录，也没有角速度调节的本轮GUI通过结论；后者已通过原生回调CPU测试。
  此次没有启动新的常驻窗口，正式窗口由菜单主流程启动。

## 平地键盘操作

`scripts/play_v40_onnx.py --keyboard` 已支持原生 Kit 键盘控制。点击**本回放窗口的 viewport** 后：

| 按键 | 策略命令 |
| --- | --- |
| 按住 W / S | 前进 / 后退，默认 `vx = ±1 m/s` |
| 按住 A / D | 左转 / 右转，默认 `wz = ±1 rad/s`，不是横向平移 |
| 松开按键 | 对应方向归零；其他仍按住的方向继续有效 |
| Space | 清空所有 held keys，速度命令归零，保留高度 |
| Q / E | 每次按下升高 / 降低目标高度 `0.01 m`，限制在 `[0.28, 0.32] m` |

W+A 等组合可用；相反方向同时按住相互抵消。`KEY_REPEAT` 不累积速度或高度，
也不会在 Space 后重新激活旧按键。窗口失焦会清空 held keys。
默认起始命令 `(0, 0, 0.30)`；键盘模式要求初始线速度、角速度为零。
`--linear-speed` / `--angular-speed` 配置启动档位，正反方向均须满足保存契约。
普通模式当前 locomotion 上限均为 `2`；探索模式及运行中调速见上节。
`--keyboard --headless` 在启动 Sim 前拒绝。

这是独立的服务器最终 ONNX 策略回放：按键改变策略输入，物理移动由六维 ONNX 动作经原有执行器产生。
Space 是**命令急停**，不是瞬间制动或位置锁定；该模型在零速命令下仍有漂移。

### 时序与原生 API

- `KeyboardController` 仅保存纯 Python held-key 状态与不可变 requested tuple，不接触物理状态。
- 初始化仍为 `env.set_evaluation_command(initial)` 后 `env.reset()`，保持 evaluation 的零根速度、关闭观测噪声。
- `ObservationCommandBridge` 只包装本 viewer 实例的 `_sample_commands`，不修改训练类。
  `_get_observations` 即使传空 ids 也会调用此入口；先前动作的 reward 计算完成后，
  新 tick 才更新 `_evaluation_command_override` 和 `commands`，再调用原 sampler。
- 同 tick 重复读取观测不切换命令；普通命令变化不调用 reset、不清空五帧历史。
  真实环境的终止／超时重置仍单独累计，并恢复最近手动命令。
- 每步核对 actor 最新帧命令、动作所用命令和 pre-reset snapshot 命令的对应关系。
- 原生订阅照已安装 Lab 3 的 `devices/keyboard/se2_keyboard.py` 使用
  `carb.input.acquire_input_interface().subscribe_to_keyboard_events(window.get_keyboard(), callback)`。
  焦点使用 Sim 6 `IAppWindow.get_window_focus_event_stream()` 与 `is_focused()`，
  事件回调和观测边界都检查失焦。关闭时显式释放 keyboard 与 focus 订阅。
  作用域为本 Kit 顶层窗口，不是系统全局键盘钩子；在同窗口文本编辑框里输入时也会收到这些按键，操作请点击 viewport。

### 输出

`telemetry.csv` 现在记录**全部步骤**：物理位置、线／角速度、动作范数、终止／超时，
以及 action / reward / next-observation / requested 四组命令和 policy tick。
`command_events.csv` 与 `summary.json.command_events` 保存初始命令、原生按键事件、焦点清空和实际应用边界。
`viewport_policy_step100.png` 与首个运动命令后100步的 `viewport_after_command_change.png`
由官方 `capture_viewport_to_file` 获取真实 viewport。

## 已执行验证（2026-09-12）

### 键盘 → ONNX → 真实物理

主验证目录：`../reports/server_final_onnx_teleop_audit_20260912T2005/`。
PID `707387` 已完成退出，运行143.643秒 wall time，2100 policy steps / 21仿真秒。
使用临时审计入口 `/tmp/opencode/v40_onnx_teleop_audit.py` 包装正式入口：
只向本进程 `window.get_keyboard()` 通过原生
`InputProvider.buffer_keyboard_key_event` / `update_keyboard` / `distribute_buffered_events`
注入测试事件。记录中的来源为 `synthetic_carb_provider_to_native_subscription`，
**不是声称人类实际按下键盘**；没有发送系统全局按键，也没有直接设置运动学位姿或速度。

观测样本（速度为真实 pre-reset 物理 snapshot，命令格式 `vx, wz, height`）：

| policy tick | 当步动作／reward 命令 | 下一观测命令 | 实测 vx (m/s) | 实测 wz (rad/s) |
| --- | --- | --- | ---: | ---: |
| 120，W 按下 | `(0, 0, .30)` | `(1, 0, .30)` | -0.1910 | -0.0003 |
| 200，W 保持 | `(1, 0, .30)` | `(1, 0, .30)` | 0.9700 | -0.0019 |
| 320，W 松开 | `(1, 0, .30)` | `(0, 0, .30)` | 0.9454 | -0.0069 |
| 350，松开后 | `(0, 0, .30)` | `(0, 0, .30)` | -0.0354 | -0.0121 |
| 500，A 保持 | `(0, 1, .30)` | `(0, 1, .30)` | -0.0965 | 0.9806 |
| 650，W+A 后仅松 W | `(1, 1, .30)` | `(0, 1, .30)` | 1.1113 | 0.9820 |
| 700，Space | `(0, 1, .30)` | `(0, 0, .30)` | -0.4447 | 0.9776 |
| 850，S 保持 | `(-1, 0, .30)` | `(-1, 0, .30)` | -1.3345 | -0.0294 |
| 950，D 保持 | `(0, -1, .30)` | `(0, -1, .30)` | -0.0382 | -1.0426 |

- 42条事件记录，17次实际命令变化；全部2100步 action/reward 命令匹配，
  前一步 next 命令与后一步 action 命令连续匹配。
- W repeat 不增速，Space 后 A repeat 不恢复旧命令；Q+repeat+release 只从 `.30` 增至 `.31`，E 回到 `.30`。
- 命令变化导致的 reset **0**；public `env.reset()` 仅初始化调用1次。
  终止重置0，环境原生超时重置1（tick1999），tick2000/2001 继续 `(0, 1, .30)`。
- 全部检查的数值有限；退出时两类原生订阅均成功释放。
- 两张1280×720真实截图已打开核实机器人、地面可见：
  `viewport_policy_step100.png` 和 `viewport_after_command_change.png`（tick220）。
- `verification.json` 为 `/tmp/opencode/verify_v40_onnx_teleop_audit.py` 对完整CSV与审计记录运行断言后的结果；
  `synthetic_input_audit.json` 保存注入事件和真实物理样本；完整日志位于同级同名前缀 `.log`。

这证明本地策略能够接收平地前后移动／转向命令并产生实际响应，不代表精确速度跟踪或零速位置保持。
几何校准仍有既有问题：审计指出的膝轴外观件合并在原始 `L_link1` 中；本次没有轴修正，
运动学／物理资产、URDF、环境与奖励保持原样。

焦点事件的独立证据位于 `../reports/server_final_onnx_teleop_focus_20260912T2003/`：
先给本窗口注入 W，然后仅把桌面焦点切回已核实的终端窗口，没有向终端发送按键。
原生 `FOCUS_LOST` 在 tick149 请求 `(0, 0, .30)`，tick150 在旧 `(1, 0, .30)` reward 后应用零命令；
tick151/160 保持零命令，且无 reset。这轮审计随后因 Kit 的重新聚焦请求未获窗口管理器接受，
在 tick190 被临时审计入口的 focused 断言中止，summary 如实保留 `failed`，不计入上面的2100步完整通过记录。
后续焦点复测还出现测试计划以外的 W/S 输入与焦点变化，因此最终交付直接使用无注入的正式入口。
窗口焦点 API 确实可用；自动重新聚焦不保证成功，用户点击 viewport 即可。

## 正式交互窗口启动命令

tmux：`v40-server-final-onnx-teleop`。以下命令已在该独立 session 中执行；
重新启动时必须换一个新的 report 路径与日志名：

```bash
timeout --signal=TERM --kill-after=10s 590s env \
  OMNI_KIT_ACCEPT_EULA=YES \
  TMPDIR=/home/yukikaze/.cache/kit-tmp \
  LD_LIBRARY_PATH=/home/yukikaze/.local/lib/compat \
  /home/yukikaze/isaacsim60-venv/bin/python -u scripts/play_v40_onnx.py \
  --research --keyboard \
  --onnx /home/yukikaze/Documents/workspace/robot_rl/reports/current/cqa1_round2_20260912/final/policy.onnx \
  --contract /home/yukikaze/Documents/workspace/robot_rl/reports/current/cqa1_round2_20260912/final/contract.json \
  --report-dir /home/yukikaze/Documents/workspace/robot_rl/reports/server_final_onnx_teleop_20260912T2013 \
  --command 0 0 0.30 --max-steps 6000 --max-wall-seconds 580 \
  > /home/yukikaze/Documents/workspace/robot_rl/reports/server_final_onnx_teleop_20260912T2013.log 2>&1
```

单环境，窗口标题含 `SERVER FINAL ONNX TELEOP`、PID 和控制提示。
到6000步或580秒wall预算时正常结束，外层最迟600秒退出；也可直接关闭本窗口。
该正式入口没有审计事件注入。旧固定 viewer 已自然退出，没有停止其他 GUI 或训练进程。

正式窗口交付时的独立检查记录在该目录 `handoff_verification.json`：

- PID **732407**，niri window ID **102**；核实时窗口在前台，进程运行中。
- 已核对前 **1375** 步，重置0、数值全部有限，1375步的 action/reward/next 命令连续匹配。
- 正式窗口确实收到 `native_carb_keyboard` 事件：tick19 的 W press 在tick20生效，
  tick21 的 W release 在tick22归零；该运行没有注入测试事件。
- 持续同一命令至少26步后的真实样本：W 在tick327测得 `vx=0.69038 m/s`；
  零命令tick448测得 `vx=-0.05801 m/s`；A 在tick832测得 `wz=0.96504 rad/s`；
  S 在tick149测得 `vx=-0.81771 m/s`；D 在tick738测得 `wz=-1.08476 rad/s`。
- 正式窗口两张原生截图均已生成并打开核实：`viewport_policy_step100.png`、
  `viewport_after_command_change.png`。这些是运行中的时间点记录，后续状态以实时summary为准。

## 此前固定命令回放（历史证据）

使用 `../reports/current/cqa1_round2_20260912/final/policy.onnx`，SHA256：
`7d4a93f0d0692f914dc769fb94c88550de62e44a0b3553d95b20d80d406b700e`。
它与服务器最终 completion 中的 ONNX SHA 一致。入口强制核对 sidecar、原始
run_manifest 文件 SHA、sidecar 内嵌 manifest、契约摘要、资产 SHA、观测/动作维度和 stage。
运行阶段仅使用 ONNX Runtime CPU provider 的原始确定性输出，不读取 PyTorch checkpoint。

源环境 Sim 5.1 / Lab 2.3 / RSL 3；目标环境 Sim 6.0.0.1 / Lab 3 beta.patch1。
入口复用当前分支的版本与资产 preflight，以及 `launch_app` / `make_env`，不要求两端版本相等。
这是跨仿真版本的真实物理回放，不等于服务器上的策略质量验收。

此前固定命令回放的首次观察记录（历史记录，旧进程现已退出）：

- PID `687690`；tmux `v40-server-final-onnx-replay`。
- 525 个真实 policy steps / 5.25 仿真秒，约60.20秒 wall time（含启动）。
- 固定命令 `(0, 0, 0.30)`；初始根速度0、无观测噪声；1环境。
- 终止重置0、超时重置0；全部检查的观测/动作/reward/物理张量有限。
- 最大原始动作 L2 范数1.12727；该时刻根位置约 `(-1.002, -0.003, 0.307)` m。
- 该窗口内非轮接触、膝限位、倾倒、低高度及基座视觉边界触地诊断计数均0。
- 零速度命令下已有明显平移漂移，不能宣称实现世界坐标位置保持。
- 第100步实际 viewport PNG：1280×720、1,334,566字节，已打开核实机器人与地面可见。

证据目录（相对本工作树）：
`../reports/server_final_onnx_replay_20260912T192951/`。

- `summary.json`：运行时身份、PID、精确 argv、模型/契约/首帧原始输入 SHA、步数、重置、诊断累计及截图状态，每25步刷新。
- `telemetry.csv`：前500步，以及后续终止/超时帧的动作范数、根位置/速度、重力方向、接触及净空。
- `viewport_policy_step100.png`：本回放进程实际 Kit viewport 截图。
- 同级 `server_final_onnx_replay_20260912T192951.log`：完整标准输出与错误日志。

## 此前固定命令启动命令

工作目录为 `isaac_wheeled_rl_train-60`。以下为 tmux 内实际执行的命令；再次执行时必须替换成新的 report 路径及日志名：

```bash
timeout --signal=TERM --kill-after=10s 590s env \
  OMNI_KIT_ACCEPT_EULA=YES \
  TMPDIR=/home/yukikaze/.cache/kit-tmp \
  LD_LIBRARY_PATH=/home/yukikaze/.local/lib/compat \
  /home/yukikaze/isaacsim60-venv/bin/python3.12 -u scripts/play_v40_onnx.py \
  --research \
  --onnx /home/yukikaze/Documents/workspace/robot_rl/reports/current/cqa1_round2_20260912/final/policy.onnx \
  --contract /home/yukikaze/Documents/workspace/robot_rl/reports/current/cqa1_round2_20260912/final/contract.json \
  --report-dir /home/yukikaze/Documents/workspace/robot_rl/reports/server_final_onnx_replay_20260912T192951 \
  --command 0 0 0.30 --max-wall-seconds 580 \
  > /home/yukikaze/Documents/workspace/robot_rl/reports/server_final_onnx_replay_20260912T192951.log 2>&1
```

默认 GUI，原生相机通过官方 `update_view_to_asset_root("robot")` 跟随机器人。
可直接关闭本回放窗口。省略 `--keyboard` 时仍为固定命令模式，
`--command VX WZ HEIGHT` 必须满足保存的 locomotion 契约范围。
DirectRLEnv 的终止/超时自动重置继续使用同一 evaluation override，并分别计数，不隐藏失败。
默认6000步（60仿真秒）、600秒wall预算；此次wall预算580秒，外层最迟600秒结束。
只有运行快于物理时间时才 pacing，不为显示速度跳过物理步。

## 代码与测试

新增 `scripts/play_v40_onnx.py`、`tests/test_onnx_replay.py` 和本说明。
原有未提交兼容修改、资产、环境、奖励与执行器均保留。

```bash
/home/yukikaze/isaacsim60-venv/bin/python -m pytest \
  tests/test_onnx_replay.py tests/v40/test_unified_commands.py tests/v40/test_round2.py -q
git diff --check
```

47项相关测试通过（本文件26项），3条既有 ProxyArray／ONNX export 弃用提示。
覆盖真实小型ONNX图加载/推理、契约/资产/维度/stage错配、损坏模型、
manifest字节变更、图签名变更、缺失sidecar、输出目录拒绝覆盖，
以及 held keys、相反方向、Space、repeat、高度步进/限幅、CLI拒绝、焦点和订阅清理。
命令边界测试复用现有 source-method loader，直接执行实际 `_get_rewards`、`_get_observations`、
`_sample_commands` 和真实 `NoisyHistoryStack`；不复制生产方法，也不启动 Isaac。
CPU测试与下述真实 GUI 证据分别记录。

收尾时工作树的完整 `git status --short` 清单（目录按 Git 汇总展示，全部未提交）：

```text
 M README.md
 M scripts/check_v40_env.py
 M scripts/train_v40.py
 M src/wheeled_algo/v40_export.py
 M src/wheeled_algo/v40_job.py
 M src/wheeled_tasks/direct/v40_serial/env.py
 M src/wheeled_world/assets/v40.py
 M tests/v40/test_joint_import_limits.py
 M tests/v40/test_launch.py
 M tests/v40/test_round2.py
 M tests/v40/test_unified_commands.py
 M v40_train_local.sh
?? docs/SERVER_FINAL_ONNX_REPLAY.md
?? docs/SIM60_LOCAL_CAPACITY.md
?? docs/SIM60_LOCAL_VALIDATION.md
?? reports/
?? runs_v40/
?? scripts/play_v40_onnx.py
?? src/wheeled_tasks/direct/v40_serial/contact_sensor.py
?? tests/test_onnx_replay.py
?? tests/v40/test_sim60_compat.py
```
