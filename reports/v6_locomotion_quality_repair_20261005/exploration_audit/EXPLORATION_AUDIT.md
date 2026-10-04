# 逐轴探索与恢复来源：CPU 只读审核

本页基于三个已封存本车 PT、本地 SCUT `b8ff79f3df855faf9dc92f4a282bd80c42649466` 的公开 PT/YAML，以及既有原生恢复回执。**未发现 v14 将探索 std 重置、恢复漏项或非有限参数的证据。探索扩大已在旧 speed 阶段形成；它可能影响随机训练行为，但不能据此解释均值策略的全部退化。**

审核未启动 Isaac、GPU、PPO、SSH 或 TensorBoard。原 PT 的前后 SHA 一致；未修改源码、合同或运行任务。全部数值和来源 SHA 在 [exploration_audit.json](exploration_audit.json)，复核工具为 [audit_exploration_cpu.py](audit_exploration_cpu.py)。本审核不授予训练、部署或硬件 qualification。

## 1. 实际保存值：六轴不是共享一个 scalar

原生 RSL-RL 5.5.1 `GaussianDistribution` 使用 `distribution.std_param`，形状 `[6]`、`std_type=scalar`。这里 scalar 指直接在 std 空间参数化，每轴分别可学习；不是所有轴共享同一标量。它是**不随观测变化**的对角高斯探索噪声，均值由 MLP 产生。

轴顺序为 **L_joint1、LL_joint1、R_joint1、RR_joint1、L_joint3、R_joint3**。控制器内部顺序是左右各三轴，使用 `CONTROL_FROM_POLICY=(0,1,4,2,3,5)` 转换；本表保持 policy 顺序。

| 封存来源 | L hip | L knee 驱动轴 | R hip | R knee 驱动轴 | L wheel | R wheel | `mean_std` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 flat 14020／flat actor 13000 | .043575 | .042453 | .043324 | .038914 | .125889 | .106967 | .066854 |
| 旧 speed 19917／speed actor 3563 | .277495 | .293520 | .261213 | .267515 | .807045 | .750431 | .442870 |
| v14 immutable 24000／speed actor 3685 | .273837 | .283228 | .252977 | .261445 | .824454 | .761369 | .442885 |

三个输入的 SHA 分别为 `7e3644f…8891f`、`123841b…11da`、`430c1c3…b3c3e0`。旧 speed 目录名 **19837 是 actor 消费，不是 19917 学习更新，也不是该域 3563 actor 更新**。v14 的 24000 同样不是新训练了 24000 次：它只比此旧 speed 父模型多 **122 次 actor 更新**，域时钟为 3563→3685。消费时钟、学习谱系、域 actor 时钟在 JSON 中分列。

旧 speed 相对旧 flat 各轴 std 为 **6.03–7.02 倍**。v14 相对旧 speed 的四腿轴为 **0.965–0.987 倍**，两轮为 **1.015–1.022 倍**。因此 `.44287→.44288` 这个均值不能表达每轴变化，也没有显示本次 v14 全部重置为 `1.0`。

## 2. 动作尺度和上限：std 不是车体速度误差

本车三份合同的尺度与 ABI 相同：35D actor、81D critic、6D action、50Hz policy、200Hz feedback/physics。四腿动作先限于 ±3，再乘 **0.25 rad** 并相对名义角解码；轮动作先限于 ±9，再乘 **10 rad/s**。腿目标还进行就近角度环绕，实际力矩继续经过 PD、包络、响应与运输。

| 来源 | 四腿目标的未饱和 σ（rad） | 两轮速度目标的未饱和 σ（rad/s） |
|---|---|---|
| 旧 flat | .010894 / .010613 / .010831 / .009729 | 1.258892 / 1.069674 |
| 旧 speed | .069374 / .073380 / .065303 / .066879 | 8.070449 / 7.504309 |
| v14 | .068459 / .070807 / .063244 / .065361 | 8.244537 / 7.613686 |

这些是 **动作目标扰动尺度**。它们不等于实际关节误差、轮速误差、力矩 std 或车体平移误差；均值靠近动作／力矩上限时也不能继续用高斯线性换算。未在本审核测量物理饱和率。

Runner `clip_actions=null`；控制器执行自己的动作限幅。PPO 保存未裁剪采样及其高斯 likelihood，实际环境使用裁剪后的命令。这个链路在三来源一致，未发现 v14 专属尺度或裁剪漂移。它仍有动作饱和时的信息损失，不能只凭 `clip_actions=null` 判断输出无约束。

原生分布默认有效 std 范围为 **[1e-6, 1e6]**，通过前向 clamp 计算；本次三个模型所有 raw std 均为正数、处在范围内部。没有碰到下限／上限，也没有由 clamp 掩盖这三个输入的负 std。actor、critic 和 Adam 所有 tensor 均 finite。

## 3. 恢复和训练所有权：没有找到漏载 std 或熵绕过 mask

CPU 使用原生 `MLPModel` 严格加载两个网络，并独立加载原生 Adam 状态，三个输入逐值复核均一致；没有调用 optimizer step。配置中的 `initial_noise_std=1.0` 只用于构造新 actor；加载时它被 `distribution.std_param` 覆盖，不能把保存的 `agent_config.resume=false` 当作实际命令未恢复：训练入口使用命令行 `--resume`。

既有原生回执还提供了实际训练入口证据：

- `v9_resume_verification_010.json`：speed 的初始 actor／critic／Adam 都来自旧 flat `7e3644f…`，并逐值一致；critic 适配时 actor 不变。
- `mixed_resume_verification_011.json`：旧 speed 1684→混采续训，父 `08642e0…`；actor／critic／Adam 全部逐值恢复。
- `first_PPO_verification_014.json`：v14 父来源是旧 speed 真封存 `123841b…`，实际参数为 `--resume --resume-task-change`，actor／critic／Adam 逐值恢复；不是 actor-only transfer，也不是重开 critic。

当前 CPU 重载可独立验证三个输入自身完整性；这些历史入口的原始 initial-state PT 没有全部留在本机，历史逐值结论标明为**既有原生回执证据**。

std 对应的 Adam step 为：

| 起点→终点 | Adam std step 差 | 对应 actor 更新 | 核对 |
|---|---:|---:|---|
| flat 306630→旧 speed 377890 | 71260 | 3563 | 3563×5 epochs×4 minibatches = 71260 |
| 旧 speed 377890→v14 380330 | 2440 | 122 | 122×5×4 = 2440 |

这与 actor 的实际优化暴露相符，没有显示 std Adam 步数归零或额外推进；不能把此 minibatch 计数混作 PPO iteration。critic 适配阶段通过 `actor.requires_grad_(false)` 冻结含 std 的全部 actor 参数；恢复后打开其梯度，不重新赋值 std。

唯一显式 std floor 在 `--transfer` 的可选 `transfer_noise_floor` 路径。三份合同未设置它，实际 v14 是 `--resume`，不经过此路径。

`RecoveryRolloutMask` 用整个 20ms 区间的真实 actor 所有权决定 learning mask，先分割 GAE，再把有效人口压成 storage，交原生 PPO 更新。**actor surrogate、critic loss、entropy 使用同一个压缩有效人口**；没有发现 script／零力矩 hold／BLEND 被单独留下来领取 entropy 优化的路径。`policy_motion_hold` 并不自动等于脚本所有权：纯 RL 仍控制力矩的 hold 可继续学习，实际以原生 mask 为准。

## 4. entropy、TensorBoard 与均值回放的口径

原生 loss 为 `surrogate + value_coef*value_loss - 0.005*entropy.mean()`。entropy 是六轴高斯**微分熵之和**，促进探索；它不是任务成功率。三个输入的理论当前熵为 **−8.4720、2.8434、2.7752 nats**；连续分布微分熵可以为负，不是非有限或实现错误。

`Policy/mean_std` 写的是 `output_std.mean()`。这是最后一个 minibatch 前向时缓存的分布；checkpoint 保存最后 optimizer step 后参数，因此 event 与 PT 小量差异可能有一 minibatch 的更新偏移。它还把轮、腿六维混为一个标量，不能代替逐轴记录。当前没有读取新的 TensorBoard event 或远端日志；本页不声称已新增逐轴 TB tag。

训练用 `actor(..., stochastic_output=True)` 采样；ONNX 导出与已有物理探针用 `as_onnx()` 的 deterministic output，取 **MLP 均值**，不用探索 std。本机 CPU 复核取样前后 deterministic output 一致。

既有 `native_003/report.json` 同条件均值策略回放，在 305mm、先 2s settle 后 **13s stand** 的窗口为：

| 同一均值策略短探针 | 平面速度均值（m/s） |
|---|---:|
| 旧 flat | .005926 |
| 旧 speed | .049831 |
| v14 immutable 24000 | .034265 |

旧 speed 与 v14 std 几乎相同，均值回放结果仍不同。这三个速度差异来自均值网络及其闭环行为，**不能解释成回放采到了不同 std**。训练中约 0.13m/s 的随机人口指标包含探索、随机化、出生、终止／重采和窗口选择，与这条均值短探针并不匹配；不能直接用 0.13→0.034 判断训练恶化或探索单因，也不能据此宣称均值策略已达到旧 flat 水平。

## 5. SCUT 对照：保存的 std 也会增长，要比较物理尺度

SCUT 实际 `rough_dash/2026-07-19_19-15-07/params/agent.yaml` 配置 `ActorCritic / scalar / state_dependent_std=false / init_noise_std=1.0 / entropy_coef=0.005 / clip_actions=null`。其 `launch_checkpoint` 指向已有 flat `model_5500.pt`、`resume_training=false`；所以 `model_0.pt` 的名称不代表六轴 std 必须都是 1。

| SCUT 保存来源 | 四腿 std | 两轮 std | `mean_std` |
|---|---|---|---:|
| flat-and-rotation 17500 | .100668 / .115158 / .104266 / .117082 | .371466 / .361702 | .195057 |
| rough-dash 0 | .147781 / .168826 / .143737 / .171414 | .520813 / .524843 | .279569 |
| rough-dash 3500 | .188840 / .232753 / .192491 / .218938 | .705462 / .704093 | .373763 |

其实际 env 配置腿尺度是 **0.5 rad**、轮速度尺度 **10 rad/s**；本车腿尺度为 0.25。因此 rough3500 的腿目标 σ 为 **0.0944–0.1164rad**、两轮约 **7.05rad/s**；本车 v14 腿目标 σ 为 **0.0632–0.0708rad**，两轮 **7.61–8.24rad/s**。本车 normalized std 更大并不等于所有物理动作探索更激烈，std 增长本身也不是 SCUT 没有而本车独有的 bug。

SCUT 的历史 stock `ActorCritic` 安装实现没有完整打包在这份源码内；本页只认证实际 YAML 和 `model_state_dict.std` 的保存值，**不借用本车 RSL5 的 clamp 范围来声称他们历史 runtime 有同样上下限**。SCUT 数据不是本车机械、速度覆盖或真实效果的对照组。

## 6. 下一步最小判因建议，尚未执行

保持同一冻结 PT、资产、出生 cohort、目标轨迹与物理反馈，做单次 305mm 静止／0.5m/s 双向／释放停车的 **mean 与原生 stochastic** 配对回放。先在无噪声无延迟投影下看探索贡献，再单独加入当前延迟／随机化；记录逐轴采样减均值、目标／力矩饱和、动作变化、速度与姿态。两种 actor 输出不同是受控变量，不能把它描述为完全同 RNG 的环境暴露，因为不同动作会改变终止和重采时刻。

这一诊断无需 PPO。若它确认当前 std 对训练误差有大贡献，才讨论冻结后的 exploration 配置候选；减少 std 仍不会自动修复已经改变的均值网络，也不能只按训练 reward 来选择。当前没有足够证据直接热改 std、重新初始化或归零 Adam。

验证：审核工具正常完成，8 项 CPU 失败路径／参数化测试通过；其中显式测试了非有限状态拒绝、错误 SHA 在加载前拒绝、逐值树比较及 raw std 与前向 clamp 的区分。未执行物理消融或新训练。
