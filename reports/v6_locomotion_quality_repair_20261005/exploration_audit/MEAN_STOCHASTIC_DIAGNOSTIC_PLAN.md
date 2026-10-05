# 同一 PT 的均值／随机策略：最小物理配对诊断

状态：**implemented_cpu_verified / physical_pair_pending**。独立入口、fixture、原生采样和首cohort记账已完成，74项CPU测试通过。此文绑定24500封存输入，替代早期24000草案；尚未执行两臂物理诊断，没有修改core、训练合同、std、Adam或当前课程。`qualification=null`，`hardware_deployment_ready=false`。机器可读设计及源码 SHA、CPU 采样接口检查在 [mean_stochastic_design.json](mean_stochastic_design.json)。

先做 **clean mean / clean stochastic 两臂**，不一次启动噪声、延迟等六臂。目标是分离同一均值网络上探索的物理贡献，不能用它将已有均值策略退步归因于 std，也不据此直接修改 std。

## 固定输入与场景

第一对输入为 `probe/v14_update24500_input/model.pt`，SHA `78dc24240038910b3844cc609a26d259075239b137c008ee038de7a76b4963e7`。seal 的学习消费是24500、学习谱系17613、全局actor谱系17553、speed域actor时钟4185、v14新增622actor，不能把24500 解释成 v14 新增训练轮数。源合同 SHA `f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3`。

两臂各自使用新进程、新原生 `FixedCaseEnv` 场景，同一个 PT、case 顺序、资产、控制、出生 seed、population 和时序。每 case 首 cohort **N4**，3 case 合计 **N12/arm**。环境 seed 为 190619，动作采样私有 seed 为 93001；这是一组短诊断，不声称四个 lane 是四次独立训练或多种子显著性证明。默认 source 是 speed，但物理场景为平地普通任务，不触发跳跃。

保持源合同原有 **35D/81D/6D、50Hz 策略、200Hz 新反馈物理/PD、64/32 求解器迭代、轮侧响应、重力、气弹簧和闭链资产**。腿 PD 为 160/2.5，动作尺度腿 .25rad、轮 10rad/s，腿动作 clip±3、轮 clip±9。资产 SHA `875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba`；控制 SHA `8073540d1cec0b32c6216699863b3ef53064a2d385b7110abe291d9da0e9b1ac`。未 qualified 的腿侧响应继续不声称已辨识接入。

显式构造诊断专用的 source-role 投影及独立设计文件，枚举每个修改字段，在 `AppLauncher` 导入/初始化之前进行 CPU preflight。实际模块、材料、质量/COM 和种子出生值在 runtime 再核对。所有输入和 effective fixture 写 SHA 回执；在 finally 再核对 PT、合同和核心文件不变。诊断源码也须冻结后再执行，不接受启动时任意工作树。

clean 投影关闭观测噪声、`PhysicalSignalDelay`、旧帧延迟、USB/command transport 的延迟/抖动/丢包、动力学随机化、接触域随机化、push、降级、非标准出生和恢复 FSM。保留原生名义机构、轮摩擦/惯性响应和标称接触。`contact_domain` 关闭会切回 native `average` 合成；必须读取实际轮/地材料证明其**有效名义摩擦**，不能只以模块关闭推断接触相同。两臂使用同一投影，不能把结果与保留 transport/DR 的旧 `native_003` 直接当同条件对照。

普通出生仍使用 `FixedCaseEnv` 的原生小姿态 jitter、相同 `reset_seed_key=quality_common_reset` 和每 case 4 replicas；不自己写静态站姿或复原 contact state。出生事务结束后不再写 root、joint position/velocity 或 reset。两臂 public birth 的18DOF和 root 状态要逐值/公差检查，场地/材料和排布要逐项相同。这不声称复原了 PhysX contact/warm-start 内部状态；使用 fresh process 可避免顺序复用旧求解器缓存，但并不保证跨运行轨迹逐位相同。

场地可直接沿用既有 quality probe 的 `evaluation_long_corridors=true`、半长44m、原生拼块宽度和 `flat_triangle_mesh=false`。新投影优先使用 `terrain=flat`、constant skill、mode0/1；源 `env.py` 已支持 flat 与 jump 使用相同 corridor surface 路径，不能为了复用旧名字宣称已训练跳跃。保留所有原生终止项和场地边界。

## 三条可见命令轨迹

复用 `probe_locomotion_quality.trajectory`、`advance_playback_command`，冻结 `vx_m_s2=1.5 / yaw_rad_s2=4`。driver 每20ms更新 requested command，slew 后的实际 command 同时写入 `commands` 与 `command_target`，避免 native 再走一次 slope；记录每 tick requested/visible command 验证这一点。yaw 请求为0，height请求 .305m。

| case | primary phases | 完整20s剩余时间 |
|---|---|---|
| stand_305mm | settle 2s → stand 13s | post_probe_hold 5s，单列 |
| forward_0p5 | settle 2s → acceleration .34s → steady 4s → braking .34s → stop_warmup 2s → stop 10s | post_probe_hold 1.32s，单列 |
| backward_0p5 | 同上，速度命令 −.5m/s | post_probe_hold 1.32s，单列 |

**release 是松杆后请求速度归零**，发生于 braking 首 tick；车体参考按1.5m/s²斜率降为0。它不是 DM 失能、零力矩 release，也不是非标准出生的200ms hold。停车10s窗口使用自身起点位置 anchor；不能把前段合法行进位移当停车漂移。primary stand 13s和stop 10s不混入尾段 hold，以保留明确历史比较窗口；尾段仅另报。

每臂执行1000策略 ticks、4000物理 ticks，请求 horizon20s。fixture 明确 `episode_seconds=20`、每 case20s、`auto_reset=false`；不复用旧脚本把 episode limit 设为 `2**40` 的逻辑。原生20s done 的 timeout 记为 timeout；只有进程/探针先退出且 lane 尚未 native done 才记 right-censored，`censor_time_s=actual_physics_tick*.005`。

## 原生 PT actor 与采样接口

独立入口为 `scripts/probe_locomotion_exploration.py`，**实现与CPU检查完成，物理执行待完成**。不用旧 `load_actor()` 的最终返回值：该 helper 最后执行 `.as_onnx()`，只保留均值而移除分布。复用其 full-PT/合同/ABI 认证原则，直接构造同 native `MLPModel` 并 `strict=True` 加载保存的 `actor_state_dict`，包括 `distribution.std_param`；不 transfer、不重新初始化、不替换 std、不构造 PPO/optimizer。

agent_config 只作网络/分布 factory，不能将其中历史 `resume=false` 当实际训练入口。认证 PT seal、infos 内 source contract/asset/control、架构及35/81/6；saved actor 全 finite 且六维 raw std 有效；输出 SHA 与原 checkpoint 匹配。可复用 `recovery_prefix_support.validate_checkpoint_payload` 的纯 CPU检查，但不直接复用其专为 V8/PREPARE 的 `materialize/core_sources` allowlist来接受任意新 probe；本入口需独立 source identity 与 projection。

每 tick 在同一次 actor 前向取得 `mu=actor(obs, stochastic_output=False)`，然后**显式** `actor.distribution.update(mu)`。native deterministic forward 不刷新分布；不先 update 就可能采到上一帧 mu。记录 `actor.output_std` 与 raw `std_param`，有效 clamp 仅来自原生分布。本 PT clamp 未激活，仍需每次实际读，不能用自写 clamp 替换。

采样调用原生 `actor.distribution.sample()`，用 `torch.random.fork_rng` 暂时安装私有 Generator 的 RNG state，取样后保存该 state，并由上下文恢复环境的全局 RNG。CPU 使用 `devices=[]`；CUDA使用明确整数 device index列表和对应 CUDA RNG API。每臂是单线程采样，不跨线程交错 fork。示意如下（独立 probe 内的新 helper，不修改 RSL）：

```python
mu = actor(observations, stochastic_output=False)
actor.distribution.update(mu)
with torch.random.fork_rng(devices=[cuda_index] if is_cuda else []):
    # Device-specific set/get RNG state; private generator is not env.generator.
    install_private_state(action_generator.get_state())
    sampled = actor.distribution.sample()
    action_generator.set_state(read_private_state())
raw = mu if arm == "mean" else sampled
```

mean 臂也每 tick 消费同形状 native draw，只把 mu 发给环境；其 `potential_sample_minus_mean` 与 **实际 injected=0** 分列。这保持动作私有流的 state/count配对，不能将潜在样本写成 mean 臂实际执行。两臂样本均值会随各自真实观测而不同；不能要求 `sampled-mu` 浮点逐位一致来判断 exogenous stream相同。用同 private seed、每 tick原生 sampler前后 state SHA及总 draw-count证明流对齐。

不自行生成 Normal、不切到 ONNX模拟随机、不将 std 设为0。不使用 epsilon tape 作本轮主实现；数学等价 Gaussian 的实现也不能直接称为 native sample逐位一致。

已完成小型 **CPU合成分布接口检查**：原生 sample精确重放、private state一致、全局CPU RNG不变、原生 std clamp仍适用；没有加载物理场景或 CUDA context。探针已实现；CPU检查不等于 GPU sampler已验收，首次物理入口仍须核验 fork恢复及 draw-count。

## 终止、固定分母与20s执行

每 case 四个 birth slots 固定，不 auto-reset、替补、退款或把终止后0值当完整窗口。某 slot 的 **首次 native done** 关闭其首 cohort统计，记录原始 terminated/timeouts/success/reasons及真实 tick；同 lane 后续反复 done 不重复记账。所有 lane仍执行原定20s可见命令/actor全形状 native sample调用，以保持固定人口的随机流和时间计划；终止后帧显式标 `after_first_terminal=true`，不进入任务/窗口指标。没有额外 dead-lane action门、扭矩失能或状态恢复。

这是诊断所需的 continued execution，不授予死后恢复/成功信用，不代替 native episode lifecycle。若某 lane出现 NaN/数值崩溃引发原生全局失败，不 mask/teleport救场；保留失败与未完成 lanes 的实际曝光/右删失 artifact，不能报告整臂 completed。native success 提早终止也按首次事件冻结；独立 stop窗口指标不覆盖原 success定义。

每 case汇总 native outcome分区和 `complete_primary_window_count/N4`，即使0也显式写0/4。均速/角速度/姿态等连续指标说明有效曝光秒数，不将缺失/partial窗口按0平均。mean与stochastic比较同时给完整窗子集和共同存活时间，避免只比较 surviving lanes的幸存者偏差；若完整窗率不同，主要结论先看固定N4 outcome和曝光差，再解释条件均值。最多将配对变坏描述为“此fixture同PT的探索贡献”，不能推出整个训练失败都由std造成。

## 实读记录与最小实现改动

只新增独立 probe、纯CPU采样/记账 helper/tests，复用 native `FixedCaseEnv.step`；不改 env/reward/控制/合同/RSL。不要重复调用 `decode`、`motor_efforts`、delay/noise模块或传感器读来造审计值：用只读 wrapper在**原生调用一次的输入/输出**捕获，return 原对象，退出时移除。

50Hz policy记录：case/replica/birth ID、phase、requested/visible command、actor实际35D input或其完整可追溯文件、mu、raw/effective std、potential draw、实际 raw action、实际 sample-minus-mean、policy/control轴序、action clip命中及native decoded腿目标/轮速度目标、私有 RNG前后 state SHA与drawcount、原生learning/control ownership mask（诊断不优化）。

200Hz substep记录：q/dq六主动轴、root真实姿态/平移速度、pitch/roll/yaw角速度、native PD计算的**限幅前** requested_motor_effort、current_motor_bounds、PD限幅后、post mix/transport/response与实际 applied effort、bounds ratio/命中次数。四腿/两轮分别记，不只有wheel饱和；每一力矩字段有pre/post物理时间标记。`motor_efforts()` 在 env返回后的tensor可能被乘 motor_strength或别的路径原位修改，必须在wrapper中立即 clone/读取，不事后把缓存当限幅前值。

腿目标就近环绕后的目标和raw尺度预测分开；轮目标10rad/s尺度扰动不是实际车体速度误差。无 delay角色的effective target也明记 fresh，而不是缺字段等同0。姿态velocity要区分roll/pitch角速度与yaw角速度，vx泄漏也不能写成yaw误差。只报告已实读字段；任何未接到的控制层记`not_covered`，不能写伪完整状态。

输出小 summary JSON、来源/投影 SHA及 native JSONL trace；指标进入独立 TensorBoard event，stable tag标`ExplorationProbe/<role>/<arm>/<case>/...`及单位。**不启动本机 TensorBoard服务**，以后若要查看由 root按既有Kaiser6006导入；6007不动。不能让logger/ stdout中断毁掉完整trace，文件写完后stdout故障单独记录，但不自动把failed改completed。

实现验收最小CPU测试：错误 PT/合同/SHA/轴序拒绝；strict加载保存std而非init1；确定性前向需explicit distribution.update；私有native sampler exact replay与环境全局RNG复原；native std clamp未替换；1000/4000 ticks和单次slew；first native terminal锁闭/N4显式0和partial无零填；实际tick右删失；异常/finally输入不变核验。GPU future firstrun只做工程核验，无PPO/资格。

## 后续隔离角色，均未实现／未运行

clean结果审查后再选择一个下一对；不把它们计入本轮执行计划：

- `observation_noise_only`：与clean相同名义物理，只开native `V5SignalPerturbations`，`max_delay_steps=0`，显式cases `perturbed=true`；否则 `FixedCaseEnv.reset`会关闭噪声。observations只按native每tick一次读/缓存，wrapper记录原生innovation；无delay/transport/DR/push。
- `pipeline_age_only`：只开源 `PhysicalSignalDelay`，固定source域actor时钟4185以保持全强度，fresh200Hz PD反馈。保留native .75enabled fraction、5ms jitter和profile概率/范围；不用自己设置lag或替换buffer。noise/command transport/USB/DR/push仍关。记录实际每lane enabled/profile/age/lag jitter和真source tick。N12可能没抽到长尾80ms，不声称全面覆盖。

后续每个角色仍是一对fresh scene、同角色源码/出生/seed、独立动作RNG；角色之间改变初始化模块和generator消费顺序，因此不能宣称噪声角色、延迟角色和clean跨角色共享完全相同exogenous tape。只在各对内核验实际噪声或lag stream及共同曝光。USB毫秒链路/丢包作为另一个后续独立角色，不混作pipeline age。观察噪声、延迟、探索各自差异后才讨论下一份冻结训练候选，当前不热改探索参数。

首次物理启动 `mean_physical_001` 在动作前因未计入原生构造器的四项cfg覆盖而拒绝；输入/冻结源码finally核验通过，0PPO，未产生任务评分。诊断fixture已修复为独立冻结JSON加明确四字段的严格expected，成功/失败保存逐路径差异。此修复不改环境核心或正式训练；两臂须从同一新冻结源码fresh运行。
