# source021／v19：恢复终局激励窄审计

**仅凭“继续 hold 收五个峰，提前成功少奖励”，不能解释当前恢复停滞。实际成功奖励是 30，并非函数默认的 5。** 在 γ=0.99、50 Hz 下，理想五峰 5 reward/s 的无限折扣和只有 10；同完成成本、同成功状态的路径中，推迟成功降低收益。保留正峰是已证实的机制，尚未证实为当前失败原因。

本审计只读取冻结 d0dce618／v19 和既有本机原生证据；0 SSH、0 GPU、0 PPO、0 生产或合同变更。逐输入 SHA 与标量计算回执见同名 JSON（16 个绑定输入）。

## 1. 实际物化与实际付款

- `v6_task_campaign_recovery_worker_v19.json` 明确 `success_event_bonus=30`、`termination_event_cost=200`、`policy_dt=.02`。`episode_event_terms` 的 `get(...,5)` 仅是缺字段默认；成功事件加 30、失败事件减 200，均不乘 dt。
- 冻结 PPO 配置 γ=.99、λ=.95；训练入口没有覆盖 γ。此处绑定的是配置和入口源，未另读远端运行中的 `alg.gamma`。
- `apply_dense_tracking` 在普通恢复中保留 `track_lin_vel`、`track_yaw`、`track_height` 各至多 1 reward/s；pitch/roll exp 各至多 1。负成本相加后，整个 density 字典只乘一次 .02。
- `recovery_hold_cost` 已激活：whole actor interval 且 begin-policy 尚未 motion release 时替换腿／轮／gyro 三项成本，hold 的两项 dense yaw 成本归零，五个正峰保留。初次 motion-release interval 仍按开始时的 hold 记账。
- `RECOVERY_TAKEOVER_V1` 和去峰函数存在，但 **v19 没有 `recovery_takeover` 字段、实例 settings=None，不执行去峰**。付款观察器真实构造亦断言这条边界；不能将函数存在当作已经接入。enabled takeover 与 hold-cost 还受互斥检查。

实际付款使用已完成的 REC3076 mean／SIDE1.5／4096×600 策略（2400 物理子步）诊断。它共享冻结 021／v19 奖励生产函数，但控制使用显式 SIDE 反事实，**不是当前 Kaiser 探索 PPO 的奖励分布**。原生付款残差为 0；无 optimizer。首出生、actor whole-owned、尚 hold 的 11 个非空 lesson×goal bin 共 1,114,029 lane-interval：

| 五峰与成本 | 按 lane-interval 加权的实际 density（reward/s） |
|---|---:|
| track_lin_vel / track_yaw | 1.000 / 1.000 |
| track_height | 0.99977 |
| pitch_exp / roll_exp | 0.88731 / 0.99150 |
| 五峰合计 | 4.87858 |
| hold 腿／轮／gyro 成本 | −0.12755 / −0.23794 / −0.59625 |
| 全部 42 项合计净 density | **2.96561** |

各 bin 正 density 4.8113–4.9535/s、净 density 2.7616–3.6281/s；hold air/route 正项实际最大值为 0，两项 dense yaw 成本亦全部为 0。成功事件的原生最大单次付款为 **30**。这里仅相加 `density_components` 的 42 个键；同一 `components` 映射还含离散事件、返回奖励和状态字段，不能把这些也相加当 density。

## 2. 最小收益反事实

设保持的常数 density 为 ρ，r=ρ·dt；共同成功路径／成本相同，只额外推迟 m 帧。比较点是“本可按同一代价完成的时刻”，不取消原本必需的稳定及跟随时间。

\[
G_{\rm delay}=r\frac{1-\gamma^m}{1-\gamma}+\gamma^m B,
\qquad G_{\rm now}=B
\]
\[
\Delta=(1-\gamma^m)\left(\frac{r}{1-\gamma}-B\right).
\]

取理想峰上限 ρ=5、dt=.02，r/(1−γ)=10。标准 Python 直接有限求和与闭式在 1／50／100／250／600 帧逐值核对（误差 <1e−11）；这只是标量反事实，不是原生学习实验。

| 额外推迟 | 实际 B=30 的 Δ | 假设缺字段默认 B=5 的 Δ |
|---|---:|---:|
| 20 ms | −0.2000 | +0.0500 |
| 1 s | −7.8999 | +1.9750 |
| 2 s | −12.6794 | +3.1698 |
| 5 s | −18.3788 | +4.5947 |
| 12 s | −19.9519 | +4.9880 |

因此“默认 5＋继续 hold”确有理想拖延激励，但不是本 v19 的配置。实际 B=30 时，纯常数五峰不能胜过同成本完成。扣除当前 hold 成本只进一步减小 hold 的折扣和。

**条件仍然可能改变比较。** 若完成路径比保持路径新增折扣成本 C，或成功概率低而失败付 −200，则有效完成价值会变低。相同 density、需 m 帧完成时，相对无限 hold 的理想收益余量是 γ^m(B−10)：需 1s 时仅 12.1001，需 2s 时 7.3206；新增成本超过余量可反转偏好。再如同一事件时刻的理想 success／failure 彩票 S=30p−200(1−p)，S>10 需 p>0.91304。**均非实际概率或成本估计，不能据此认定当前 reward 是根因。** γ=.99 每 20ms 的折扣半衰期约 1.379s，延迟事件信号确实迅速减弱。

## 3. 终止、截断与 actor 归属

- 原 `RecoveryTraining.outcome` 仍要求实际接触／姿态／平台与稳定证据、传感器 motion release 和原目标连续跟随 1s。true success 直接成为恢复 success；本合同恢复 specs 无 sequence。成功 done、`time_outs=False`；这帧 actor-owned 时正常学习，GAE 在 done 截断，没有成功后新出生奖励接续。
- 机械／脚本失败优先于 success，物理终止 −200。actor 导致的终止保留 whole-interval 学习归属；前段／blend／release 的任一子步会排除整个 interval。
- 20s horizon 或 collection boundary 属 `time_outs`，无物理失败时不付 −200。**原生 RSL 先在存储 reward 加 γ×当前 transition critic value**；mask adapter 保留此加项。随后 done 使 GAE 不跨回合延续。
- `policy_bootstrap_mask` 在 autoreset 后计算，约束返回观测是否可接续；不能把它说成会否决原生 timeout 的当前 V 加项。终止帧 `learning_mask` 已在 reset 前复制。critic 预测、有效样本分割和归一化会影响优势；本标量收益上界没有证明真实 PPO 梯度方向。
- 五峰常量可以抬高 critic baseline／训练 reward，却不是误差修正梯度；误差与物理成本仍在 actor-owned interval 实付。熵 loss 和原生采样概率不是该 density 账本中的支付，不能混量纲作因果推断。

## 4. 下一条最小证据与边界

利用已授权的 2216 逐 lane 捕获，在每个实际 50Hz interval 保留五峰、全部负项、release／followup／success／done／timeout、whole mask；结合既有 200Hz 原生八门连续驻留。若能读取实际 PPO 存储，再分别对账 timeout 增广的 reward、critic V、return／advantage。先回答“不能产生连续稳定条件／完成动作”，还是“可完成动作但真实收益更差”，然后才谈激励改动。

目前没有 2216 的成功替代路径及逐 lane 实际付款／V／return。12s 观察短于 20s episode，首出生有右删失；既有聚合付款不能代替因果配对。此审计不建议改权重、不激活旧 takeover、不改变 1s／3mm 门、不停止正式 PPO，也不赋予机械或实车资格。

实际输入关键 SHA：worker `7918e785…aea95`；rewards `b6f402f2…93c2d`；env `ea40933b…96c0e`；原生 payment `c6e7c6f6…6ff62`；原生 report `e45e2b97…71888`；SIDE review `00d5ea58…c8d94`。完整值与全部输入路径在 JSON。
