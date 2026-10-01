# V6 静止支付与毫秒延迟核对

本页记录用户2026-10-02“漂移严重、轮速/静止惩罚是否未生效”和扩大毫秒延迟覆盖的要求。当前v5源码07be0f6、4096环境，实际部署身份以launch.json为准。原始曲线来自Kaiser TensorBoard attempt_005，保存为 reports/v6_task_campaign_20261001/learning_audit/stationary_reward_audit_v5.json；没有复制event或在本机启动TB。

## 已确认的支付与指标边界

- 实际启用dense_tracking。静止广域核为未截断pseudo-Huber（尺度0.10m/s，权重1.5），精核尺度0.04m/s、权重0.75；零vx/yaw且普通接地时两项均生效。旧precision_tracking的0.03宽度已被dense替换，不能沿用旧“6cm/s以上完全饱和”的归因。
- 观测到前99轮：230mm静止两项最近10轮合计−0.1882 reward/s，305mm−0.0907，430mm−0.1379；wheel_velocity同窗分别约−4.66e−5、−5.81e−5、−7.06e−5/s。均非零；普通静止落地高度/速度bonus均为0。轮速项是−1e−5×两轮轴速平方的小成本，与SCUT相同量级，不承担位置驻留的全部责任。
- 第100轮累计行为：230/305/430mm平面速率均值2.17/1.24/1.40cm/s，最大出生点位移1.11/0.292/0.353m。方向偏置分别+0.381/+0.054/+0.572cm/s。确有剩余漂移，430mm偏置未消除，不能从总reward较高认定站稳。
- Behavior/stand_drift_max_m是每组所有随机策略环境、全部历史回合的最大值；新attempt100轮对应48s单环境仿真，不是固定10s确定性评测。EpisodeMetrics origin从各回合第一步取值，max跨回合保留；observed_sim_seconds是环境时间求和，不能当单车时长。动作仍随机探索，轮侧std约0.16；不得与确定性回放混记。
- 合同没有stationary_tracking位置anchor。瞬时速度成本已支付，但同样的小速度偏置积分后的位移没有单独定价。这是目标覆盖差异，尚不能证明为全部漂移的因果根源；原B臂静止权重×2无益的结论继续有效，不盲目翻倍或增加强制零轮速项。没有位置/历史输入的35D策略也不能保证漂远后找回出生点。
- LearningAudit/continuous_zero_10s还要求高度命令逐帧不变，当前为0；它不是“零速驻留没有采样”的证明。新增独立的2s进入后固定10s位移窗口，显式记录样本数、mean/max和速度，保持旧累计最大指标以便追溯。

## 华南虎逐项纠正

核对本地及GitHub冻结源码b8ff79f3，实际rough_dash/2026-07-19参数：

- use_obs_delay/use_act_delay均开启，四传感器组分别1–4、腿/轮动作1–3；缓冲在_apply_action物理子步推进。dt=5ms时是观测5–20ms、动作5–15ms，不是按50Hz策略帧算成20–80/20–60ms，也不是10–40/10–30ms。
- flat_orientation_x/y_v罚的是速度命令调制的投影重力分量平方，不是pitch/roll角速度。我们pitch_velocity、roll_velocity已经都有；horizontal_angular_velocity为−0.05×pitch/roll角速度平方，与SCUT ang_vel_xy对应。
- SCUT wheel_vel=−1e−5与我方已有项同量级；该rough_dash实际stand_still_deadzone_enabled=false、stand_still=−0.0，不能称“实际激活了强静止专项罚”。他们仍通过零速命令的全局速度跟踪付费。
- 我方mass/COM并非“几乎相同”：当前底盘质量0.9–1.1、COM x±10mm/y±5mm/z0；SCUT归档x±40mm/y/z±20mm。放宽COM和前置推扰均应独立审核，轻推不会给无位置输入的策略凭空增加位置感。本轮不夹带改力学随机化或reward。

## 延迟补强方案

已有command_transport覆盖USB下行0.04–0.16ms、抖动和1kHz丢包；signal_perturbations只开噪声、max_delay_steps=0。确实缺少较宽的策略传感器年龄和设定值执行滞后。

新增独立的物理时钟延迟层，复用Isaac Lab原生CircularBuffer：观测只延迟IMU角速/投影重力/主动轴角度和速度，不延迟当前命令、模式、时钟或previous-action；critic真实状态及底层PD反馈保持当前物理子步。动作延迟腿角设定值与轮速设定值，再用新反馈PD，避免将“反馈延迟”和“保持旧力矩”混同。USB短丢包仍由既有链路层处理，不重复注入。

拟覆盖：25%无额外延迟；其余环境按60/35/5%选常规/宽域/压力档。常规obs0–20ms、动作0–15ms；宽域obs20–40ms、动作15–30ms；压力obs40–80ms、动作30–60ms。各传感器/腿轮组独立抽取回合固定基延迟，另加±5ms抖动；压力是未测工程长尾，不能称实测USB延迟。采用200Hz网格，误差最多半个5ms子步，日志同时给请求范围和实际lag毫秒。强度按本域actor时钟渐增，flat从1500至2500，其他域0至1000；零延迟样本始终保留。

仅准备v6冻结迁移，测试通过前v5持续PPO；不热改正在训练的源码/合同。迁移继承完整actor/critic/Adam、所有消费和课程时钟，并计入一次10更新critic适应；其后常驻PPO，每500封存、非阻断review、原20k每域上限不变。全部入口断言新模块启用与参数，真实物理推进后核验非零lag和fresh PD时钟，再更新实际部署回执。未qualified腿bag仍未接入。

来源：[SCUT物理子步与reward源码](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py)、[实际rough_dash参数](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml)。
