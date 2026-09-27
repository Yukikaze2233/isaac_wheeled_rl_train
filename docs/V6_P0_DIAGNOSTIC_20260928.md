# P0补救诊断：高度扫描与去饱和A/B

计划：`contracts/v6_p0_diagnostic_v1.json`。本次是独立、有限的诊断研究；正式课程消费保持750次。

## 实验边界

- 两臂均从第750次完整checkpoint恢复actor、critic、Adam和RNG。
- 源checkpoint SHA：`f02018ec3e10d10ea1509390e2d31f649e59557f0628cc76e2b42c240d0692dd`。
- 源合同SHA：`346846638dccf9159a20e84e4864bb61126000dda9180e12f9268f5713bc5879`。
- 对照扫描使用保留的第100次checkpoint，SHA为`f09470588aa42ef4be960564cbf26dbfdfb071ca922de13ea48338b6ff9f00eb`。
- 35D/81D、200Hz物理、50Hz策略、原气簧/机械/力矩限制；16384环境，每次rollout24步。
- 每臂100次优化，包括10次critic适应与90次actor更新；LR固定为源checkpoint的7.5e-6。
- A/B分别从同一个源开始，B不继承A的训练结果。合计200次诊断更新、78,643,200个transition，单独记账。
- 每臂TensorBoard横轴从0开始，Adam内部step和动量不清零；源正式计数750、学习谱系250单独保留。

## 唯一治疗变量

| 参数 | A_control | B_wide |
|---|---:|---:|
| 新静止精度宽度，m/s | 0.03 | 0.15 |
| 新高度精度宽度，m | 0.01 | 0.03 |
| 原静止宽核，m/s | 0.10 | 0.10 |
| 移动速度精度宽度，m/s | 0.15 | 0.15 |

两臂其余奖励权重、控制、任务采样、种子完全相同。扩大高度核也会改变当前误差区的有效奖励强度；
若B优于A，只支持该改动组合有效，不自动证明“总奖励无梯度”是唯一根因。
A/B总reward数值也不能直接横比；应比较同定义的行为指标、有效误差分布和饱和占比。

## 训练分布

- 高度：0.26、0.28、0.305、0.32m；每个高度分别训练静止和0.5m/s前进，8个等额组。
- 回合20秒，前进命令斜率0.5m/s²；无非零yaw训练命令。
- 两臂使用相同名义机械/材质/传输通道，关闭附加随机化采样和观测扰动。
  这用于隔离名义条件下的可学习性，不是正式鲁棒能力验收。
- 环境机械边界、接触/跌倒终止仍生效。课程晋级、能力回滚和自动接续正式课程均关闭。

## 自动流程

1. seed190619、190620分别对100次与750次模型执行高度扫描。
2. 对750次源模型运行既有32项快速回归面板，作为本次同协议比较基线。
3. A_control完成100次，再做双seed高度扫描与32项回归。
4. B_wide从相同750源状态独立完成100次，再做相同评测。
5. 输出诊断汇总后正常结束，`deployment_checkpoint`始终为空。

高度扫描是独立协议`v6-height-response-diagnostic-v1`：4个高度×静止/前进，
每case4次、20秒回合、前5秒预热。扫描保存策略输入、四腿目标角、实际电机角、
请求/执行力矩及包络等轨迹。仅完整存活且有有效统计窗口的点参与闭环高度响应拟合；
拟合的增益和截距不作为自动标定修正。

固定回归面板沿用原case定义、canonical布局及阈值，yaw仅评测。正式最大位移门限没有被平均速度替代。
平均速率使用根link在世界水平面的速度模长，避免有符号速度抵消；其统计排除warmup，
原最大漂移仍从回合首帧累计，两个时间窗不能混用。

## TensorBoard诊断

- `Behavior/<group>/height_actual_mean_m`、`height_command_mean_m`、`height_bias_m`。
- `Behavior/<group>/planar_speed_mean_cm_s`及原`stand_drift_max_m`。
- `/diagnostic/<group>/reward/<term>`：分任务奖励贡献。
- `/diagnostic/<group>/saturation/<term>`：有效样本中误差大于约3倍核宽的比例。
- `/diagnostic/<group>/activation/<term>`：对应项实际生效的样本比例。
- `DiagnosticUpdate/<group>/kl`与动作均值变化：每轮更新前后，对同一组rollout观测的策略漂移。
  这不是所有PPO minibatch的KL，也不改变固定学习率。
- `DiagnosticUpdate/action_std_<axis>`：实际采样分布的各轴标准差。
- 扫描/回归指标写入同一远端TensorBoard目录树；JSON/NPZ只承担验收和复算证据。

分任务统计在GPU上按rollout累计，优化后批量回收标量，避免每个物理过程频繁同步。

## 判读规则

- A、B都改善：任务分布与继续适应本身可能有效。
- B明显优于A：支持扩大当前奖励工作区有帮助。
- 两臂都未通过：不足以直接归因为资产错误或网络容量不足，需要结合动作、力矩、观测与独立可达性证据。
- 高度达标而持续漂移：只能说明高度控制有进展，不等于站立任务通过。
- yaw掉点：说明负迁移/缺少复习，不能单独证明参数量不足。

## 验证

- 154项相关CPU测试通过，覆盖两臂变量隔离、同源顺序编排、独立预算、无自动晋级、诊断批次独立封存、
  平面速率不抵消、KL探针不消耗RNG，以及跨inference/optimizer阶段的诊断缓冲生命周期。
- 本地真实64环境短测验证完整750学习状态恢复，完成10次critic适应与2次actor更新，ONNX校验通过。
- 本地高度扫描验证了新增带符号指标与策略输入/关节/力矩轨迹输出。
- 工程证据：`reports/v6_p0_diag_probe_20260928/`；正式远端研究的部署身份另附回执。

## 2026-09-28部署与首批核验

- 冻结代码commit：`35f77f121d645adaf259ed9678c687381eb3fdae`，已推送GitHub main。
- 源码归档SHA：`51deae984b5df8293b63d8038e267f7fddf2c409bbb93625823302100faf854d`；无未提交overlay。
- Kaiser run：`/home/kaiser/robot-rl-sim60/experiments/v6-p0diag-20260927T172355Z-650f90`。
- tmux：`v6-p0diag-20260927T172355Z-650f90`，最大运行时间14400秒，独立于SSH连接。
- 启动回执：`reports/v6_p0_diag_formal_20260928/launch.json`。
- GitHub `tests`与`command-reference-tests`均通过，运行ID分别为`36336703089`、`36336703115`。
- TensorBoard：`http://127.0.0.1:6006`，A组训练选择`p0_diag/A_control/train`；
  基线扫描/回归位于`p0_diag/tensorboard/{retained100,source750}/`。

北京时间01:24启动，01:36完成两seed基线扫描和回归并启动A组子进程；
16384环境初始化约5分钟，随后执行10次critic适应。这些阶段不能只靠actor更新计数判断进程是否停止。

01:44:58已核验A组完成26/100次优化，其中10次critic适应、16次actor更新，
累计10,223,616个诊断transition。正式课程消费仍为750次；B组按计划排在A组训练及评测之后。
actor、critic、Adam恢复均逐张量一致，学习谱系从250接续，诊断计数从0单独记录。
恢复证明与该时刻progress已回收至启动回执目录：
`A_control_resume_verification.json`、`progress_first_actor_updates.json`。

TensorBoard已记录实际actor更新：step22的同观测KL探针约0.00615，LR为7.5e-6，
采样8.32秒、学习0.228秒；前10个critic-only step的KL为0符合预期。
第10次checkpoint已独立封存，进程、状态发布和event写入均在推进。

### 基线结果与解释边界

750次源模型的32项回归在两个seed均为6/32。下表为seed190619的静止高度扫描，
高度均值只统计5秒预热后的窗口，每点4次完整20秒回合：

| 指令高度，mm | 100次模型实际均值，mm | 750次模型实际均值，mm |
|---|---:|---:|
| 260 | 299.53 | 319.42 |
| 280 | 302.64 | 324.27 |
| 305 | 306.12 | 329.51 |
| 320 | 308.32 | 332.54 |

双seed的静止高度响应增益分别约0.1455–0.1456和0.2178–0.2179。
305mm指令下，seed190619的平均平面速率分别为2.63和12.34cm/s。
这些是源模型对照数据，尚不是A/B治疗效果，也不能解释为固定的几何零位偏置。

两个模型的前进扫描在两个seed均因`boundary`提前截断，各高度0/4完整回合，
所以响应拟合正确返回`insufficient_complete_episodes`。代表轨迹确认截断时横向位置约3.70m，
对应8m宽走廊扣除0.3m余量的边界；`done=true`、`terminated=false`，无跌倒或机械越限记录。
这是横向越界，不是训练程序退出；这些不完整轨迹不能冒充完整20秒前进高度响应。
机器可读基线索引已回收至`reports/v6_p0_diag_formal_20260928/baseline_summary.json`，
完整JSON与NPZ保留在远端`train/baseline_scan/seed_{190619,190620}/`。
