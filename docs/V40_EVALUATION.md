# V40 独立策略评估（工程研究目标，未运行真实评估）

当前优先事项仍是机械/部件归属。髋/膝链传到同轴嵌套输出这一信息**不改变 serial 运动学**，也**不证明 motor→joint 独立直驱映射**；链比例、角参照、内外输出归属与执行器映射尚未确认。本实现不修改该物理逻辑、contract、资产或任何准入门禁。

本次交付只有纯 CPU 指标/假数据回归与只读官方源码核对。没有运行训练、Isaac、MuJoCo 物理仿真、SSH、安装、3080 请求或录制视频；**没有策略效果/随机化/真机能力的通过结论**。

## 使用边界

- 默认（包括不传任何开关）为 --preflight-only 语义：只读检查，不创建输出、不导入 Isaac、不启动 Sim。
- 只有显式 --apply --research，且 train/play 的 contract、asset、collision、传动、版本等现有门全部通过，才可能进入实际评估。没有手工 passed 开关，research 不能豁免失败证据。
- 复用 scripts/train_v40.py 的 preflight / make_manifest / checked_checkpoint / make_env 和 play_v40.py 的官方 RslRlVecEnvWrapper 路径；不修改训练入口、不实现自制 PPO。
- checked_checkpoint 复用 stock RSL-RL v3.0.1 权重布局验证（包括 actor/critic 与配置身份、weights_only 加载、字节 hash）。执行的是批准 mean actor 的 eval()/inference_mode() 前向，不调用探索采样 act()/distribution.sample()。
- checkpoint 邻接 run_manifest.json 的 stage 必须等于 --stage。缺字段/不同 stage 一律拒绝，不能因命令范围相交就宣称 checkpoint 已训练另一个 stage。
- v1速度域为vx ∈ [-.5,.5] m/s、wz ∈ [-1,1] rad/s；v2速度域读取实际stage，默认locomotion为两者±2。高度保留 [.28,.32] m 域。启用站立混合时，`vx=wz=0` 是单独的合法分量，即使正常运动区间不含0；其他移动命令仍须满足运动区间。
- 每个 case 只有一个确定性 reset + 固定命令；不把不同 seed 当真实扰动/鲁棒性证据。默认 locomotion 只是有限采样点，不覆盖完整连续域或所有组合。
- 12.57 rad/s 高转、world-XY 定点保持、真实扰动/随机化均 not_tested；低速零 vx 旋转时的漂移指标不是世界位置控制器验收。

## CLI（下列只是用法，未执行实际评估）

只读 CPU 预检（在当前不满足门禁/目标版本的本机预期退出 2）：

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../.rl_deps_rsl23   ../.venv_mj314/bin/python scripts/evaluate_v40.py --preflight-only --research
~~~

未来经授权且所有门禁通过的目标服务器上，显式有界评估示例：

~~~bash
python scripts/evaluate_v40.py --apply --research --headless   --stage stand --checkpoint /absolute/run/model_final.pt   --output-dir /absolute/new-evaluation-directory
~~~

--num-envs 固定为 1；每 case 最多 10s（默认 1000 policy ticks，.01s/tick，加一条 initial anchor），绝不延长原 episode timeout。任何失败/终止/超时立即停止该 case，当前 suite 也 fail-fast；尚未执行的 case 显式 not_tested，不能被聚合成通过。

可通过 --command VX_M_S WZ_RAD_S HEIGHT_M 指定**一个固定命令**。不提供则 stand 1 点、height .28/.30/.32 三点、locomotion (vx,wz)=(0,0),(+.5,0),(-.5,0),(0,+1),(0,-1)，均 height=.30。若具体 contract 缩小范围，默认只保留合法点；保存实际 case 清单，不宣称缺失点已经验证。

当前v2统一策略还增加站立高度区间两端的固定命令用例。检查其静止能力时，仍以
`--stage locomotion --command 0 0 0.32` 运行同一模型；不要切成独立stand身份。
固定指令评估不会触发训练时的10%随机站立采样，也不证明真实起停切换已经验收。

--thresholds path.json 可在执行前配置工程目标，未知字段/NaN/负数拒绝。--duration-s 可覆盖时长，但必须是 .01s 整倍数且 <=10；若时长减到 2s 以下，也必须在 thresholds 中把 steady_start_s 调低。

默认阈值不是用户实测规格：

| 指标 | 工程研究目标 |
|---|---|
| 有效观测 horizon | 10s（不能用短 episode 假通过） |
| 稳态窗口 | reset 后 2s 起至 case 结束 |
| 零 vx 站立/低速旋转最大、最终轮轴漂移 | 各 <= .05m，锚点始终为 reset 初始值 |
| 高度 RMSE / 绝对稳态偏差 | <= .03m / .01m |
| vx RMSE / 绝对稳态偏差 | <= .15m/s / .05m/s |
| wz RMSE / 绝对稳态偏差 | <= .30rad/s / .10rad/s |
| 倾角峰值 | <= 20° |
| 仿真 effort / 常数 effort prior 峰值 | <= 1.00001（浮点容差；不是额定电流认证） |

物理终止接触阈值、膝界限、最低高度/倾角门槛仍读取原 contract；工程阈值绝不修改 env 的 dones/reward/action/reset/collision 行为。

## env API 与 auto-reset 防泄漏

新增 public API：

1. set_evaluation_command((vx, wz, height))：仅配置评估 override；默认 None，不改变训练 RNG 或 sampler。
2. 随后必须 reset()；reset 的原有 _sample_commands 在 override 下固定复制目标，不调用 rand。到期 observation 采样也保持该目标。设置后尚未 reset 就取 obs 会直接报错，不在同一 tick 重写/重复推入 history。
3. capture_evaluation_initial_snapshot()：reset 后、执行动作前取得初始真实轮 link 轴原点中点；不推进 physics/history。每个 case collector 只取一次。不是升速结束后重新设锚。
4. get_evaluation_snapshot()：返回最近 policy tick 的独立拥有副本。_get_dones 内已有终止判据算完后、奖励和 auto-reset 前缓存；同 tick 最多缓存一次，_reset_idx 不清空/替换该缓存。
5. set_evaluation_command(None) 后 reset() 才恢复训练 sampler；无 mid-episode ramp/扰动接口。

IsaacLab DirectRLEnv.step 在推进两个 physics substeps 后，先增加 policy/episode 计数，再 _get_dones→_get_rewards→_reset_idx→_get_observations。因此 step 返回的 obs 和 robot.data **可能已经属于下一 episode**。collector 不从它们提取指标；只消费 pre_reset snapshot，并在 terminated/timeout 时立即停止，避免被 reset 后 upright 姿态骗过。

快照在创建与返回时 deepcopy tensors（独立 storage，数值未 nan_to_num）；字段不存在就抛错，绝不用可疑默认 0。原 HistoryStack 的 tick 去重实现未改；旧 env CPU 回归仍保留。

## 字段、单位与源码核对

只读核对本机 ../.deployment_sources/IsaacLab-v2.3.0/source/isaaclab/isaaclab，**不是运行验证**：

| 输出字段 | v2.3.0 属性/语义 |
|---|---|
| root_link_pos_w_m / root_link_quat_wxyz | ArticulationData.root_link_pos_w / root_link_quat_w；link actor 原点与 wxyz 姿态 |
| root_com_lin_vel_b_m_s / root_com_ang_vel_b_rad_s | 训练使用的 root_lin_vel_b / root_ang_vel_b 在 2.3.0 是 root_com_*_vel_b 的别名；COM 速度在 root link 坐标表达 |
| projected_gravity_b | canonical body gravity；upright [0,0,-1] |
| joint_pos_rad / joint_vel_rad_s | joint_pos / joint_vel，按 contract action_order 显式重排 |
| sim_joint_effort_nm | applied_torque：显式 actuator clipping 后送入仿真的 effort；**不是 self.torques 指令、solver reaction，也不是真机电流** |
| wheel_axis_midpoint_w_m | 按 robot.body_names 查 L_link3/R_link3 的 body_link_pos_w，再取中点；明确不是 body_com_pos_w |
| wheel/non_wheel_net_force_max_n | contact_sensor.net_forces_w_history 的每向量模，先对 2 samples 取 max，再按轮/非轮分类 |
| base_visual_clearance_lower_bound_m | 原 env rotated base_visual_bbox 八角点的保守净空下界；不是精确 mesh 最近距离 |
| policy_tick / physics_steps / time_s / sim_time_s | 原 common_step_counter / _sim_step_counter，分别乘 .01/.005 秒；不是 wallclock 或毫秒 |
| episode_step / episode_time_s | 终止前 episode_length_buf 及乘 policy_dt 的秒数，reset 初始为 0 |
| terminated / timeout / termination_flags | 原有 dones 的 bool 与真实原因，二者同时发生也保留 |

官方源码关键位置：

- [DirectRLEnv v2.3.0](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/envs/direct_rl_env.py#L349-L397)：step 顺序、policy/physics 时钟。
- [ArticulationData](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/assets/articulation/articulation_data.py)：323-328 applied_torque，816-885 link pose/origin，1013-1020 root velocity aliases。
- [Articulation](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/assets/articulation/articulation.py#L1844-L1869)：actuator compute/effort 写入，216 set_dof_actuation_forces。
- [SensorBase.update](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/sensors/sensor_base.py#L183-L191)：history_length>0 时每 scene.update 刷新；ContactSensor 346-360 用 physics_dt 获取 net forces 并 roll history。

本地上述四个源文件 SHA256（依次 DirectRLEnv、ArticulationData、Articulation、SensorBase）：

~~~text
52f16e81cbde14abfffcb3946b612ba9f78b3d6394e54c6f5a13af67976f6bba
bfe53518d1c511455b3538dcee487845ccfe1b62ddbd9c2cd92794137d390994
9cc03b85642c36c801ff9683e94b8ccc3fbef1178761338974b433dacc78ef75
152c586a85da5eab67897a1849405a935fbd98d987c1a0de36171d85397ba5f9
~~~

实际目标服务器仍需验证 importer 的 joint/body 名称与序号、显式 actuator 输出 buffer、ContactSensor 子步采样、终止快照顺序/时钟与 bbox world frame。缺失/不兼容字段 fail，不产生假 0。接触是 **policy tick 内 2 个 physics samples 的最大值**，不是完整每物理步精确冲击统计，不保证能区分 ground/self-contact pairs。effort 只采该 tick 最后一个 physics substep 的值，prior >=99% 比例是已采样点比例，不是全速饱和占空比。

## 纯指标及失败规则

v40_metrics.py 只依赖 Python 标准库。evaluate_trajectory 接受初始 anchor + 连续 pre_reset records：

- 必需字段、长度、有限数、单位、整数 tick、连续步序、episode 时钟、固定命令均严格校验。NaN/Inf/缺字段/倒退/跳步/auto-reset 后记录直接 InvalidTrajectory，绝不 nan_to_num。
- 持续有效时长为首次硬失败之前的保守连续下界：剔除第一个失败 policy interval；另保存观测时长和首次失败时间，不把 reset episode 拼长。
- terminated / timeout / 同时二者 / horizon_reached / incomplete 分开；即使 timeout 恰好在目标 horizon，仍不是通过。
- vx、height、wz(yaw-rate，不是 heading angle) 的 RMSE 使用 post-action 等步长记录，稳态偏差保留正负号（阈值比较其绝对值）；不使用 reward 判成功。
- 零 vx case 的最大及最终 XY 漂移始终相对第一条 wheel-axis midpoint。非零 vx case 仍报告漂移，但不把应有前进运动当定点失败。
- 记录非轮接触 policy ticks/峰力、双轮峰力、倾角峰值、bbox 净空下界、各 joint q/dq 峰值、膝 hard-range 利用率/最小 margin、连续髋 soft-deviation 利用率。
- effort 除以常数仿真 effort prior，报告峰值及 >=99% sample fraction；不冒充 motor-side 标定或真机电流/温升/全 torque-speed envelope 认证。
- aggregate_cases 使用逐 case 全通过；一次摔倒不会被其他成功 case 或平均 reward 冲淡。没执行的 case 为 not_tested，不准作为通过。

## 输出、保护与视频

成功进入 apply 后先独占创建新 output-dir 并写入 evaluation_config.json / thresholds.json；旧目录（包括空目录）拒绝，文件使用 x 模式，禁止覆盖。

配置保存实际 cases、阈值、seed、stage、target versions/preflight、contract/asset_manifest/checkpoint/run_manifest hash、完整 run_manifest/provenance 与配置 hash。启动后再次核对环境身份和 checkpoint 字节，变化则失败。

每个已执行 case 保存 *.records.json（有界原始轨迹）与 *.summary.json；最终 summary.json 保存整体失败列表、阈值与 hash。若非有限原始记录导致失败，原始数值以 {"invalid_numeric":"nan/inf"} 的显式诊断标签保存，summary 为失败；这不是指标输入修复，重算时仍应拒绝这些记录。没有采样成功则不给模拟指标假值。

视频始终 video=not_recorded。本机未核对目标 render/capture API，未实现或声称录制成功。未来可在门禁通过后另行验证 IsaacLab render_mode=rgb_array、官方 RecordVideo wrapper 或 camera 的实际输出/帧时钟与设备兼容性，并把帧与快照 policy_tick 对齐；在验证前不要改成 recorded。

## CPU 回归

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../.rl_deps_rsl23   /home/yukikaze/Documents/workspace/robot_rl/.venv_mj314/bin/python   -m pytest -q -p no:cacheprovider tests/v40/test_metrics.py tests/v40/test_env_contract_static.py
~~~

本次该命令 91 passed（新增 54 个 case + 原 env 37 个）。所有新增数据都是 CPU 受控假轨迹/假 clock 或从源中隔离的 tensor method，不是物理仿真/真实策略测评；覆盖 auto-reset 泄漏、克隆所有权、初始漂移锚定、NaN/缺字段/短 episode、秒/毫秒错误、terminal/timeout 分类、单次失败不被均值冲淡、域外/stage 拒绝、固定命令 sampler 不耗 RNG、pending reset/history 保护及输出覆盖保护。
