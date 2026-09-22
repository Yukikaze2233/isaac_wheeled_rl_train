# V5.2：奖励审计后的完整训练与部署计划

日期：2026-09-22。执行计划：`contracts/v5_scut35_repair_v52.json`，继承V5.1修复分布和完整七阶段队列。
来源：[华南虎原文核对](SCUT_STUDY_20260919.md)、[当前机制核对](V5_CURRICULUM_AUDIT_20260922.md)、用户9/22的R项与C1/C2分析。

## 1. 证据与本轮决策

旧mixed、locomotion与当前SCUT35不是同一reward合同，不能合并成“同一L1死区造成三轮失败”的因果证据。
当前SCUT35高度项为 `exp(-clip(e, ±0.15)^2 / 0.001) - (10e)^2`，`height_l1_weight=0`。
3cm处导数约−30.39 /m；指数核弱远场仍有平方代价。奖励对误差的导数不等于PPO策略梯度或物理推力。
多任务高度平台、漂移和启停失败支持继续检查奖励形状、分布与优化冲突，尚不能排除训练量与状态分布因素。

最新基准采用12486次更新的候选，固定27案例通过18项，27项均完成存活时域；不是已验收策略。
来源：Kaiser `v5-speed-resume-20260922/train/stage_00_flat/evaluation_016/candidate_00.json`。
checkpoint SHA-256：`e6307dd7419c052a96285e58ce6a6624b8832e6c5986aadf568287bfcfc37f81`。

| 实测短板 | 基准数据 | 本轮对应动作 |
|---|---|---|
| 低高度驻留 | 高度MAE 3.03mm，但漂移335mm | 固定端点训练、驻留锚点、零速惩罚 |
| 高高度驻留 | 高度MAE 6.29mm，漂移132mm | 同上，窄核精调 |
| backward_05、forward_1、backward_1 | vx MAE分别0.106/0.120/0.142m/s | 提高这些固定目标的采样覆盖 |
| 反向4、2π、8、4π rad/s自旋 | 高度MAE分别12.27/34.81/50.76/92.93mm | 方向分窗、转速门控、宽→窄高度核 |
| 可恢复姿态瞬态 | 历史问题待终止原因/轨迹对照；上述候选失败案例均无fall终止 | 训练100ms持续确认，固定评测仍单帧判定 |

## 2. R项与C项落地

### 高度“定性→定量”课程（R1／核覆盖）

每有符号技能组保留最近64个真实完成episode；双轮支撑帧占比至少80%，无失败/出界，才计为合格完成。
统计原始高度、vx、yaw误差，合格完成率需≥98%；两次变化至少间隔200个4096环境参考更新。

| 等级 | 1/e核宽 | 高度项倍率 | 晋级高度MAE | 退回初期高度MAE | 近目标二次损失系数 |
|---|---:|---:|---:|---:|---:|
| 定性 | 0.15m | 2.0 | ≤15mm | >40mm（已在初期） | 288.89 /m² |
| 巩固 | 0.05m | 1.5 | ≤6mm | >30mm | 750 /m² |
| 精调 | sqrt(0.001)m≈0.03162m | 1.0 | 已到末级 | >15mm | 1100 /m² |

初期宽核降低近场尖锐度，远场保留平方项；逐级提高近场分辨率。每级晋级/退级阈值有滞回，避免一越界即来回切换。
宽度与倍率在各环境reset时锁存；提高自旋难度时恢复初期等级。后续阶段沿用默认最终核，按前序案例回归验收。
窗口、等级、时钟随checkpoint保存，恢复时未完成物理episode重新开始。

### 辅助力（R2／R3中的辅助力条目）

本轮零外部竖直辅助力。公开SCUT存在性能门控组件，但已核对V14／预训练配置未启用它，
其计数也不是简单的“500个独立episode”。辅助力仅作为后续有独立对照、零辅助验收的实验候选。

### 奖励形状和权重（R3／R4）

`scripts/analyze_chassis_rewards.py`直接调用训练reward函数，输出高度、速度、yaw、姿态、
电机速度/加速度/力矩、动作变化及轮叉差的TensorBoard切片；驻留曲线调用实际`StationaryAnchor`。
训练增加`/reward/working_margin`和`/reward/dense_jump`，补齐这两项的实际贡献观测。

```bash
/home/yukikaze/isaacsim60-venv/bin/python -B scripts/analyze_chassis_rewards.py \
  --plan contracts/v5_scut35_repair_v52.json --output reports/v52_reward_shapes_new
```

审计值是相同状态下每秒奖励密度相对目标的损失，不能当作可达轨迹的等价交换率：

| 项 | 定性 | 巩固 | 精调 |
|---|---:|---:|---:|
| 高度误差1cm | 0.02887 | 0.07382 | 0.10516 |
| 高度误差5cm | 0.71032 | 1.32318 | 1.16792 |
| 高度1cm／速度0.05m/s跟踪损失比 | 5.61 | 14.35 | 20.44 |

速度0.05m/s的跟踪损失为0.005144/s；静止组另有0.15/s平移惩罚；驻留偏移5cm损失0.22120/s。
因此本轮明确加强有限驻留，同时逐步提高高度精度，保留已有努力与平滑代价。
实际价值取舍还须看各技能reward贡献、误差与力矩饱和，不能以系数数字跨500倍直接判错。

### sim2real先验（R5）

当前已有动作率、动作二阶差、电机加速度、力矩、轮功率、机械工作余量等正则，它们是工程先验，
不是延迟或摩擦误差的精确补偿。基础修复不叠加随机化以便对照；最终`mixed_robust`显式启用：

- 70%环境信号扰动，观测/动作各0–1个policy步延迟（0–20ms），指定传感器噪声；
- reset时电机强度0.85–1.0、气簧强度0.9–1.1；
- 5%材质切换场景及已有地形、推扰、前序技能混合；
- 同时验收干净案例和`*_perturbed`案例。

这些幅度是研究先验，硬件时延、摩擦和执行器辨识仍待实测校准。不能写成“全无域随机化”，也不能写成已完成sim2real。

### 持续确认与驻留（C1／C2）

- 训练fall软判据：倾斜超过60°，或非飞行高度<0.15m，连续100ms才终止，50Hz下为5个连续步。
- 单步恢复即清零计时；reset清零。翻倒≥90°、非飞行高度<0.10m仍立即判fall。
- 非轮接触、机械越界、闭链失效维持原判据；固定评测强制0ms，保持历史验收口径。
- TensorBoard分别记录fall候选比例、确认比例、非轮接触比例；时长增加不能替代姿态/漂移达标。
- 静止组沿用V5.1有界驻留锚点，位置仅用于reward，actor仍是编码器/IMU/命令35D。
- 跳跃继续使用相位/时间参考高度、竖直速度和收腿稠密目标。

## 3. 完整队列、迁移与验收

actor从上述12486候选迁移，critic／优化器新建，动作std下限0.08；前50次实际更新只训练critic，
学习率固定5e-5，之后恢复KL-adaptive PPO。所有训练阶段共享一个actor。

| 阶段 | 4096环境参考更新 | 6144环境实际上限 |
|---|---:|---:|
| flat_repair | 6000 | 4000 |
| landing | 3000 | 2000 |
| rough | 5000 | 3334 |
| spin_translate | 5000 | 3334 |
| high_speed | 5000 | 3334 |
| jump | 8000 | 5334 |
| mixed_robust | 6000 | 4000 |
| 合计 | 38000 | 25336 |

物理200Hz，策略50Hz，rollout24，6144环境每次实际更新147456 transitions。
每500实际更新独立固定评测；每100更新封存，首次第10更新验证回收；阶段最少训练量和双种子确认沿用V5.1。
基础仍为27个原案例，后续完整回归集逐阶段扩大。预算耗尽而未过门时停止于`stage_gate_pending`，保留候选与已验收模型的区别。

本轮是一组有审计依据的联合修复，不是单因素消融；若未改善，应按相同起点拆分核宽、驻留和采样因素复测。
重点比较同一固定suite的低/高高度、漂移、正反旋转高度、启停瞬态和终止原因，禁止从reward上升直接宣称修复成功。

## 4. 提交、冻结与部署

每次训练迭代先commit代码/合同/计划，再部署该commit，训练身份以commit、源码归档SHA和合同SHA绑定。
本轮部署不用未提交overlay。远端独立tmux、自动封存与TensorBoard继续沿用现有可靠性链路。

1. CPU回归、6144环境合同展开、奖励形状审计。
2. 提交训练版本；从该commit运行64环境60次PPO短测，验证critic预热、actor更新、ONNX与归档。
3. 同一commit部署6144环境完整队列，最长7天；检查实际worker、增长的更新计数及第10次checkpoint回收。
4. TensorBoard独立展示上一轮、本轮及奖励切片；本地关机不影响远端训练。
5. 实测启动结果和运行身份写入单独部署回执并commit。

```bash
python3 -B scripts/launch_v5_remote.py --commit <verified-commit> \
  --contract contracts/v5_scut35_repair_v52.json --num-envs 6144 \
  --transfer /home/kaiser/robot-rl-sim60/experiments/v5-speed-resume-20260922/train/stage_00_flat/block_016/model_final.pt \
  --start-stage flat_repair --max-runtime-seconds 604800 \
  --output reports/v52_formal_20260922 --execute
```

部署后使用回执查询与回收：

```bash
python3 -B scripts/check_chassis_remote.py reports/v52_formal_20260922/launch.json --brief
python3 -B scripts/sync_chassis_batches.py reports/v52_formal_20260922/launch.json \
  --output reports/v52_batches_20260922 --once
```
