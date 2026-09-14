# Round4：旧七刚体结构的平地完整鲁棒/机动首段

本段使用现有七body串联等效研究资产，不使用尚未完成机械的新结构，不修改质量、闭链、旧URDF/STL或惯量。
按用户决定，从零训练1024env、30000次成功PPO更新；没有从旧策略warm-start，也不再安排独立GPU/PPO smoke。

**“完整”指本首段同时训练多高度、起停、直行、旋转、组合机动、摩擦DR和渐进推扰；本段不是terrain/jump训练。**
粗糙地形与计划跳跃属于约100k更新长期研究预算中的后续独立阶段，参考华南虎Rough路线再设计。reward≥91不是硬门槛或自动晋级条件。

## 1. 合同与兼容边界

文件：[contracts/own_v40_round4_full.json](../contracts/own_v40_round4_full.json)。

```text
contract_id: own-v40-jointspace-h5-v2
round3.stage: B1             # 已有材质/基础命令组件身份
round4.profile: flat_full
round4.initialization: scratch
num_envs: 1024
planned_updates: 30000
rollout_steps_per_env: 48
semantic SHA256:
f701f2e391fd4e48d65902632b5a166df63e5967d38a65731fa84caf5cd7d592
```

保持125 actor / 29 critic / 6 actions、5帧历史、MLP `[256,128,64]` ELU，观测尺度与动作/PD/轮电机先验均不变。
膝内角35–80°、20s episode及原v2持续失稳终止保持。高度仍0.29–0.32m；不因新机械设想放宽到0.28以下或更高。

`validate_round4`只允许本profile明确列出的命令域/比例和两项reward差异；还原这些字段并剥离`round4`后，复用严格B1验证。
未知字段、非finite/boolean冒充数值、其它课程或奖励/PD变化被拒绝。旧A/B1合同文件和默认行为未改。

## 2. 四类命令与PPO更新课程

每次速度重采样（仍为3s）重新选择：

| 桶 | 概率 | 内容 |
|---|---:|---|
| 站立 | 30% | vx=wz=0 |
| 原地旋转 | 20% | vx=0，wz按当前yaw ceiling采样 |
| 纯直行 | 30% | wz=0，vx按当前vx ceiling采样 |
| 组合 | 20% | 同时采样vx/wz，再应用下面的可行性先验 |

最终合同最大域为vx±3m/s、wz±6rad/s。**初期不是直接使用最终域**：

| 已完成PPO更新数 | vx ceiling m/s | yaw ceiling rad/s |
|---:|---:|---:|
| 0 | 0.5 | 1.0 |
| 6000 | 2.0 | 2.0 |
| 25000 | 3.0 | 6.0 |
| 25000以后 | 3.0 | 6.0 |

区间内线性插值；调用`env.set_training_iteration(n)`同时推进push和command curriculum，`n`为**已成功完成的PPO更新数**：初始0、首个成功更新后1。不是env总steps、不是乘以env数的transitions，也不是恢复checkpoint中未经换算的0-based标签。
新ceiling在下次速度重采样生效，setter本身不改command、不清history；高度仍独立5–8s采样，两端各15%，其余70%连续均匀。

### 组合命令不是上限笛卡尔积

从启动时B1实际PhysX材质读回取每env动态系数，经地面0.5/average换算得到`mu_eff_dynamic`；不拿配置期望桶冒充实际值。
以旧研究资产名义轮距b=0.4373m、R=0.06m计算两个采样约束：

```text
abs(vx*wz) <= 0.7 * mu_eff_dynamic * 9.81
max(abs(vx-wz*b/2), abs(vx+wz*b/2)) <= 0.8 * R * motor_zero_torque_speed / gear_ratio
```

当前右式约 **4.35955m/s**。先在当前ceiling内采样requested，再共同缩放vx和wz：

`scale = min(1, sqrt(traction_budget/abs(vx*wz)), wheel_speed_budget/(abs(vx)+abs(wz)*b/2))`。

实现有零分母保护；共同缩放保留符号和曲率，不增加关节或物理速度硬限位。当前配置的轮速ceiling本身已经保守，附着约束主要裁剪高vx×yaw组合。
**这是命令采样先验，不是接触支持、真实滑移、抗倾覆或电机实测能力保证**。半径/轮距/线性电机曲线仍是现有研究资产及先验。

训练日志包含四桶比例、current caps、requested/executed绝对最大值、traction/wheel-speed裁剪比例；
`training_curriculum_state.command_curriculum`保存当前caps、实际μ、requested/executed命令及裁剪mask。

## 3. 两项指定抑振reward

原upright、height、PD、L1零vx平移速度惩罚和termination不变。仅full新增以下rate，最终统一乘一次policy_dt=0.01：

1. `body_angular_rate = -0.05 * (wx² + wy²)`：使用实际body-frame根角速度，**不惩罚wz**。
2. `wheel_quiet = -0.02 * mean((relu(abs(0.06*dq_wheel)-0.018)/0.1)²)`：仅当`abs(vx_cmd)<0.05 AND abs(wz_cmd)<0.05`。

轮速项使用实际dq、合同中的两轮索引，不用policy output或wheel target替代。低于0.018m/s不处罚；正常纯旋转（yaw命令不在0.05死区内）不触发该项。
极低yaw命令若也落入全零死区，则按定义处理。原零vx L1仍独立保留，约束纯旋转中的机体平移。

`compute_reward_terms(..., contract, *, joint_vel6=None)`保留旧调用兼容；只有full必须传入finite实际dq，env已经接入。
不宣称抑振项无瞬态副作用：高速停止时实际轮速可能很大，正式训练应观察该项贡献、停止过程与扭矩饱和，而不是再暗加reward课程。

## 4. 已有B1摩擦DR与推扰保留

### 摩擦

startup64桶、每env两轮同桶、约30%nominal；effective静摩擦[0.5,0.7]、动摩擦[0.4,0.6]后裁到≤静态；地面0.5/average、恢复系数0。
使用现有实际PhysX读回和`round3_material_report`，reset不重采材质。非轮材质、质量、PD、延迟不作新随机化。

### 推扰

- root COM世界XY速度 **`v_xy += delta_xy`**，不覆盖原速度，不改vz、角速度或pose。
- 50%episode无额外随机push；每次reset重采episode mask。
- 首次push为2s保护期再加独立3–5s等待，即episode约5–7s；后续间隔3–5s。
- 独立`torch.Generator`，seed2044，不消耗command/noise RNG。
- 幅值方向：世界平面均匀方向、[0,current max]均匀幅值。
- 最大幅值：update0为0.1m/s，5000为0.25，10000为0.5，区间线性插值，此后保持0.5。

物理hook在旧reward/终止/reset完成后、新观测生成前执行；每policy tick最多处理一次，绝不放进每个physics substep都会执行的`_apply_action`。
保留pre-reset snapshot；固定evaluation override默认关闭随机push，留出评估可显式调用同一`apply_velocity_impulse(delta_xy, env_ids)`。

该接口使用Lab3明确的`write_root_com_velocity_to_sim_index`，前后通过`root_view`新读回验证delta、vz/角速度/pose保持；仅选中发生push的env才做报告。
固定SDK的setter未失效化body-frame线速度缓存，env局部失效化该缓存，保证next critic看到push后的速度，而旧reward仍描述旧transition。
输入NaN、重复/越界ID和dtype错误在写物理前拒绝；真实读回不匹配会明确报错，不伪称成功。

日志：`Push/events_count`（该观测边界的env事件数）、`Push/max_delta_v`（课程上限）、requested/realized最大幅值、读回误差和completed PPO updates。
RSL对rollout日志可能取平均，不能将平均事件数误读成单次整数计数。

## 5. 状态保存、地面资源与正式启动内验证

- `env.training_curriculum_state`包含push进度/mask/timer和command curriculum；`set_training_iteration`可恢复课程幅值/ceiling进度。
- **不声称精确trajectory resume**：push RNG/mask/timer需完整状态恢复才可作此承诺，现实现明确标false；B1材质映射继续按既有完整report恢复机制处理。
- `first_push_report`初始为None，只有正式运行实际写入/读回后才形成证据。CPU fake-backend测试不是物理通过证明。
- `cfg.ground_usd_path`支持官方Grid USD本地缓存，env构造及spawn前再核对固定SHA：
  `78e9a1e72a8838a13d0f65c49cd487ab92e89233cd128b057730b5b5b4ca2164`。
  只改变文件来源位置，不改变地面物理内容；已有play的上下文缓存入口仍兼容。

用户已明确不再独立快测：**本任务没有启动GPU、PPO smoke或远端probe**。正式startup的B1材质读回、首次push真实读回，以及后续评估由主agent正式运行采集。

## 6. CPU验证与主agent集成

新测试：[tests/v40/test_round4.py](../tests/v40/test_round4.py)。包含：

- 0/5k/6k/10k/25k课程边界及非法计数；PPO进度不从env tick推算。
- 四桶概率、actual per-env μ顺序、附着/轮速域约束和采样前后值。
- 实际dq与policy target区分、全零/纯旋转门控、原A/B1 reward逐值兼容。
- 独立push RNG、保护期、同tick不二推、部分reset、evaluation不随机推。
- CPU真实张量写入adapter检查COM增量、非选中env、vz/角速度/pose保持；错误输入不写入、错误读回失败。
- oldreward→nextobs、critic缓存更新、命令课程只在重采样应用且不清history；地面cache错误字节拒绝。

运行：

```bash
/home/yukikaze/isaacsim60-venv/bin/python -m pytest \
  tests/v40/test_round4.py tests/v40/test_round3.py tests/v40/test_round3_b1.py \
  tests/v40/test_round2.py tests/v40/test_core.py -q
```

本次上述CPU回归结果：**118 passed**，2条既有ONNX export弃用提示；`git diff --check`通过。未生成本轮GPU/物理通过报告。

主agent需同步CLI profile/feature flags并完成其全量CPU验证，然后直接正式launch：

1. 固定旧七body资产、`contracts/own_v40_round4_full.json`、`--stage locomotion`、1024env、30000更新。
2. 新轮从零初始化，禁止在新轮入口沿用A/B1 warm-start/stage-transfer；既有Round4中断后的恢复属于另一明确语义。
3. 每次成功PPO update后调用统一`env.set_training_iteration(completed_updates)`；不要用0-based checkpoint标签代替计数。
4. 设置已验证`cfg.ground_usd_path`，记录真实B1 material report、curriculum state和首个push report；将`round4.py`纳入provenance源码清单。
5. 首段是平地完整鲁棒/机动训练，后续Rough/jump按独立评估与约100k长期研究计划推进，不用reward阈值自动宣称通过。

本任务未改CLI/job/export/play或旧资产；保留其它agent的dirty B1、stage-transfer及部署代码，没有commit/push。
