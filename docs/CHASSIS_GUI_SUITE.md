# 本机 REC ONNX 场景回放

在项目目录执行 `scripts/play_chassis_suite.sh` 打开平地场景。执行
`scripts/play_chassis_suite.sh --menu` 打开场景菜单。先关闭当前回放窗口，
再从菜单启动新场景；入口持有同一 GUI 锁，重复启动会被拒绝。

所有场景默认使用
`models/v6_recovery_candidate_38000_20261008/policy.onnx`，不会按功能替换策略。
回放是 `interactive_unscored`，`qualification: null`。当前候选未授予
高速、台阶、跳跃或硬件部署资格。ONNX 只做确定性 actor 推理，未构建优化器，
未执行 PPO。环境直接复用 FixedCaseEnv、原资产和 native REC FSM。

点击 viewport 获得键盘焦点。W/S 前后，A/D 旋转，Q/E 或 T/G 改高度，
SPACE 请求停车，P 暂停，R 重置当前场景的原始出生姿态。平地可以用
速度上限 slider 测试到 3 m/s 的请求、启停、前后与正反旋转；高度沿源合同
0.23–0.43 m 范围请求。请求 vx/yaw 经过原合同的 1.5 m/s²、4 rad/s²
斜率约束，在本次 actor 观察刷新前执行一次。GUI 中头顶箭头、姿态、实际速度、
六电机力矩、弹簧、轮接触力和闭链间隙沿用原 PolicyView。

菜单提供 18 个独立原生场景：

| 入口参数 | 场景 |
| --- | --- |
| `flat` | 普通平地，WASD / 高度 / 启停 / 高速请求 |
| `rough` | 20 mm 粗糙起伏 |
| `slope_up`, `slope_down`, `cross_slope` | 上坡、下坡、侧坡，各 15° |
| `stairs_up_05mm`, `stairs_down_05mm` | 三个 5 mm 台阶，上 / 下 |
| `stairs_up_10mm`, `stairs_down_10mm` | 三个 10 mm 台阶，上 / 下 |
| `stairs_up_20mm`, `stairs_down_20mm` | 三个 20 mm 台阶，上 / 下 |
| `stairs_up_40mm`, `stairs_down_40mm` | 三个 40 mm 台阶，上 / 下 |
| `fallen_left`, `fallen_right` | 已定义左右侧倒 90° 姿态 |
| `fallen_belly`, `fallen_back` | 已定义趴卧 / 仰卧姿态 |
| `jump_challenge` | 未训练挑战：J 请求 6 cm，K 请求 10 cm 跳跃 |

小阶踏面各 0.6 m，实际累计台面为 rise、2×rise、3×rise；使用普通人工
运动命令，不触发 STEP 或 JUMP。场景只在启动时构建；R 走原生 episode
reset，不在 episode 中改机械状态来冒充完成任务。倒地场景由原 native 自起脚本
执行，再进行实际力矩混合与 RL 接管。平移和旋转等待原 native 连续稳定放行门。
附加窗口分别显示 script、script_actor_blend、rl_hold、rl_unlocked，
以及 FSM phase、pure RL、学习间隔 mask、稳定时长、实际速度和 closure。

每次启动在 `reports/chassis_suite_<scenario>_<UTC>` 留下独立回放 profile、
CPU 预检、TensorBoard event、report、runtime 和首次可见 viewport 截图。
TensorBoard event 只记录此次回放诊断；入口不会启动本机 TensorBoard 服务。
先认证候选源合同与冻结 full-role 设计，再分别从源运行配置和独立设计期望投影
回放场景，并执行真实 playback design preflight。50 Hz 策略 / 200 Hz 仿真、
PD 160/2.5、64/32 solver 和 3 mm closure guard 保持。

指定场景也可直接运行，例如：

```bash
scripts/play_chassis_suite.sh stairs_up_10mm
scripts/play_chassis_suite.sh fallen_belly
scripts/play_chassis_suite.sh jump_challenge
```

CPU 准备验证（不会启动原生场景）与本地命令时序测试：

```bash
/home/yukikaze/isaacsim60-venv/bin/python -B scripts/play_chassis_suite.py --preflight-only --scenario flat
/home/yukikaze/isaacsim60-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_chassis_suite_playback.py
```
