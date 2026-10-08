# 普通大阶渐进课程候选

这是基于 `e5bd9c1367b875d7f229b9a1930647013d023372` 的隔离实现，默认关闭。
当前 source029、冻结 v26 合同和远端训练均未修改。本文件不是部署或能力验收回执。

## 生成入口与份额

`ordinary_large_stair.sampling_candidate(native_terrain_config, enabled=True,
origin_actor_update=..., actor_allowance=...)` 返回源配置的副本，不写合同。
输入使用已有 native terrain v23 小阶源；不直接补丁已认证的 composed v26。
`enabled=False` 完整保留源配置。

| 源池 | 原份额 | 候选份额 |
|---|---:|---:|
| 5/10/20/40 mm 小楼梯 | 42% | 42% |
| surface_traversal，含 rough/slope 等 | 10% | 2% |
| surface_height_joint | 4% | 2% |
| 新普通大阶，八行等额 | 0 | 10% |
| 其他静止、运动、停车等复习 | 44% | 44% |

donor 池内部所有旧行和原有命令/几何保持，只同比降低份额。rough/slope 的复习量
因此下降，这是明确取舍。之后 composed 域的实际份额还须通过现有组合器重新计算。

## 几何与时钟

全部八行在场景构建时用原生 `terrain_surfaces` 创建，训练过程中不移动平台。
普通上阶仍是 `body_climb`、mode 0、向前速度驱动，`StepClimbTask` 只做物理奖励
与评估，不生成命令、跳跃请求或状态机触发。

| 行 | 相对 terrain actor 放行区间 | 升高量 | 踏面 |
|---|---:|---|---|
| 单级50 mm | 1000–1200 | 50 mm | 800 mm |
| 单级100 mm | 1400–1600 | 100 mm | 800 mm |
| 单级150 mm | 1800–2000 | 150 mm | 800 mm |
| 单级200 mm | 2200–2400 | 200 mm | 800 mm |
| 双级50＋62.5 mm | 2600–2800 | 顶面112.5 mm | 800＋600 mm |
| 双级100＋125 mm | 3000–3200 | 顶面225 mm | 800＋600 mm |
| 双级150＋187.5 mm | 3400–3600 | 顶面337.5 mm | 800＋600 mm |
| 双级200＋250 mm | 3800–4000 | **顶面450 mm** | **800＋600 mm** |

每行显式 `ordinary_large_stair_curriculum` 存储 schema、enabled、clock、origin、
actor_allowance、row_index、activation_start/end。读取现有
`integrated_runtime.component_actor_updates(env, "terrain")`，不会用总消费、critic
适应次数或 episode 重置次数推进；后续 jump/stepjump 复习保留 terrain 组件时钟。

以上为 allowance 足够的时钟。较短预算按同一比例缩放，末级结束不晚于
`min(4000, floor(0.35 * actor_allowance))`，至少保留65%的 actor allowance 作全难度
巩固。allowance 必须由正式入场账本计算，排除已消费和 critic-only 更新；不得由此
增加预算或退款。小于1000 actor 的 allowance 拒绝使用这个八行候选。

## 预教、奖励与命令所有权

每个未激活大阶行，在同一真实接近垫上连续训练五秒周期：前进0.1 m/s一秒、后退
0.1 m/s一秒、零速三秒。episode 相位随机化，目标高度305 mm；不另加零力矩保持、
脚本控制、传送或 PPO mask。实际动作仍可偏离指令，越界、摔倒、碰撞等机械原因
继续计入普通训练结果，不把理想指令积分当作已经完成了接近任务。

放行后仅给向前命令，线性由0.1提高至0.3 m/s；现有 command slew 保留加速度约束。
带课程元数据的行禁止第二套 planar/heading/sequence/高度命令来源，防止后来覆盖
课程目标。exact evaluation 绕过训练课程，直接执行固定向前普通上阶用例。

`StepClimbTask.enabled` 保留静态几何归属，`assessing` 表示已实际启用的大阶评估。
预教期间禁用大阶平台奖励、完成 credit、无支撑 flight 统计、攀阶高度豁免与台面
接触豁免；普通速度、驻停、姿态奖励和机械保护保留。mask 在物理 transition 开始
前同步，采用实际起点轮坐标：启用时已经站顶不伪造跨沿，首个真实跨沿不会被吞掉。

新课程指标复用原 `extras["log"]`，包括 `/curriculum/ordinary_large_stair/*`。
诊断区分 `ordinary_large_stair_pretraining`、`ordinary_large_stair_assessing` 与
`body_climb_enabled`；只有 active 分母可以用于普通大阶专项成绩，不能把预教 timeout
描述成一次已请求的大阶失败。

## 已验证与剩余接线

CPU：70项通过，覆盖源池/复习不变、原生几何与物理评估一致、最终预算内放行、
组件时钟与 critic 计费分离、后续复习时钟保留、前进/后退/停车曝光、无跳跃请求、
禁止命令双所有者、exact 绕过、未激活专项零支付以及启用边界的正反例。
未启动 Simulator、GPU、PPO 或远端工作进程。

正式下一版还必须在已有 authoring 路径绑定：新 native terrain source 与 source pins、
SCUT 复用时排除这八行的 planar/heading 改写、重新组合的 row_provenance/row_ownership、
冻结的 terrain actor origin/实际 allowance、新 recipe 与 migration/入场回执。
须完成实际 startup 读回、命令曝光和普通上阶旁路仿真后再部署；不能把目前候选直接
当作 v26 pinned graph 内已启用的课程，也不能把 CPU 通过当作上阶能力通过。
