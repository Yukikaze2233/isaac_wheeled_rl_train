# 跳跃真实触地奖励修复与部署

本轮只修跳跃域的真实落地支付和请求生命周期。冻结合同 `v6_task_campaign_v12.json` 从正在运行的 v10 迁移；其余五个 worker、父模型树、采样、固定评测、预算及自动节点保持原值。部署回执以 `reports/v6_jump_contact_repair_20261005/deploy/` 和主 `launch.json` 为准；候选源码提交本身不代表部署。

## 证据与参考

原生 TensorBoard 最近 300 次 actor 更新、10 个普通跳跃组的六项落地奖励均为零。物理 LANDING/RECOVERY 存在，命令 LAND/RECOVER 相位为零；触发信号在真实恢复阶段释放，使预设轨迹相位条件失效。完整任务成功率约 12.6–14.8%，不是落地条件成功率；基础跟踪奖励仍在，不能称所有奖励缺失。

参考华南虎官方冻结源码 `b8ff79f3df855faf9dc92f4a282bd80c42649466` 的 [rough_dash 实际配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml) 和 `state_machines/airborne.py`：真实腾空/接触窗口、腾空轮力矩、触地前方向轮速、下降速度和单轮接触力约束。该配置的 landing_trajectory 是关闭的。这里增加的支撑相对站高吸收参考、3 秒成本窗口和参数是本机构工程候选，不声称其原样参数或已辨识参数。

## 生命周期

接受新的请求后锁存命令；观察到真实 FLIGHT、无轮接触、COM release 和至少 60 ms clear air，才允许首次物理触地开启落地成本。未验证腾空前取消请求会解除；验证腾空后释放触发信号不擦掉结果。普通站立和非请求被动接触没有这些成本。

首次触地启动不可续期的 3 秒成本窗口；最初 0.3 秒使用实际支撑相对站高与下降速度入口的有界 Hermite 参考，然后精修锁存站高、前进速度、横向速度、yaw 角速度、倾角和 roll/pitch 角速度。入口位置/速度限制有独立诊断；不是修改实际状态或注入力。反弹、持续请求、忙时重复不能重置时钟。

成本窗口截止和任务截止独立。原来 6 秒结果超时也被触发释放屏蔽，现在从接受请求的锁存时钟执行。现有 COM 顶点、空中净空、释放速度、落地区域、速度/姿态/双轮接触和连续 2 秒稳定的成功条件不变。成功、终止和 reset 清理生命周期；之后从地面接受的新请求可开启下一次跳跃。

## 稠密成本

每个轴用零中心广域与局部项：

\[
r_b(e)=-w_b\left(\sqrt{1+(e/s_b)^2}-1\right),\qquad
r_f(e)=w_f\left(\exp(-(e/s_f)^2)-1\right).
\]

广域成本持续区分大误差改善，局部项提高精度区分；理想状态为零，没有长期停留在恢复相位的正奖励。解析误差斜率不是 PPO 收敛保证，后续以行为和优势采样判断。各项是每秒密度，环境只乘一次 20 ms policy dt。

空中下降且轮接近地面时，方向轮速目标为锁存前进命令除以实际轮半径；使用本资产左右轴符号 `[+1,-1]`。轮力矩过大在该窗口付成本，静态跳没有正向转轮收益。触地冲击使用每策略步真实物理子步的每轮力峰值，单位 N；各轮持续接触至少 20 ms 后取单轮最大，再按实际随机化质量的体重归一化。不是两轮总力、不是冲量，参数不照搬另一种车的 200–400 N。

静态跳同时保留相对请求起点的位移成本；行进跳追踪原前进命令，不被要求停车。完整参数在 worker 的 `jump_landing` 字段，独立设计表逐项断言。

## 工程与训练

4096 环境，actor35D/critic81D，50 Hz 策略和 200 Hz 仿真新反馈 PD；关节160/2.5及轮侧响应、气簧、延迟和域随机化继承。新增奖励状态不进入 actor。实车 PC PD 1 kHz 不能仅凭仿真200 Hz称等价；未 qualified 的腿侧 bag 仍未接入。

先 CPU 生命周期/公式/独立合同测试和30角色预检；冻结 Git archive，远端逐文件验证；合法封存当前完整 actor/critic/Adam；原生4096环境、同一冻结 train role、真实actor确定性推理工程探针确认触地后实际成本支付且触发释放后持续。探针不计PPO、不授予技能资格。然后同账户一次接续，10次 critic-only 适应收费不推进 actor，后续连续 PPO；按原15000 actor节点进入后续课程，20k本域消费上限/121000总上限不变。不中断做45项晋级检查，不能退款历史工作。

成本与诊断在已有 TensorBoard event：`Reward/jump_*`、`LearningAudit/<group>/jump_contact/*`、`/jump_contact/*`。重点同时看非零支付、触发释放后窗口、入口裁剪、实际任务成功/失败/超时、静止/运动复习组和每更新 wall/env时间。推力奖励为零的早释放现象不在本轮声称解决；大场地/rough分布也没有在这个合同中扩展。

## 模型交接

新 ONNX 是对应不可变 checkpoint 的确定性 actor 导出；checker和PyTorch/ONNXRuntime数值核验不等于实车动态验收。当前RMCS跳跃上下文仍固定零，启用跳跃前需按冻结 `scut_observation.py`、`references.py` 和命令参数实现 command clock/apex/context。物理落地成本只在训练端，不能把仿真接触或COM真值塞给部署actor。

模型必须附原checkpoint/合同SHA、轴顺序/符号、160/2.5与速度环参数、策略/反馈控制/发送频率区分、上下文ABI和验证结果。未经同条件回放和硬件接口核查，不标“比旧模型强”或hardware qualified。

## 实际部署结果

2026-10-05 01:55（UTC+8），冻结源码 `2b3734aeb37a4c1d91b555a1cbec74e72ce02f14` 已在原账户、tmux r12、jump attempt_001 接续。原 attempt_000 于 actor2571／总消费23694合法封存，PT SHA `1dcd85bfb94c5b79e1e32da6e46368df425b394c15c508a101f9a4a4a6747156`；完整actor/critic/Adam逐值恢复，fixed LR1e-4。10次critic适应计费、不推进actor；首总消费23705／jump actor2572的9个actor张量改变。02:02原生读数已到2591／总23724，仍连续PPO，未授予跳跃资格。

同冻结train role的4096环境原生工程探针完成400策略步／1600物理反馈PD步：观察3952次真实触地、触发释放后的510737个落地环境帧，新增成本非零且非正，六项旧落地支付为零。首次探针遗漏现有 `sim60-runtime.sh` 引导，CUDA上下文未建立；失败报告保留，同一源码正确引导后通过，未改物理或放宽检查。正式PPO的原生event也确认新成本非零、旧jump_03落地项为零，运行配置预检通过。工程探针的664次完整任务事件不是确定性成功率，不能用来证明修复后质量提升。

证据为 `reports/v6_jump_contact_repair_20261005/deploy/verification_summary.json`、`jump_contact_first_actor_verification_012.json`、`native_probe_012b.summary.json`、`native_PPO_payment_012.json`；主 `reports/v6_task_campaign_20261001/deploy/launch.json` 已更新。源码之后的结果文档提交不重启训练。

本轮供用户本机GUI及实车接口联调的主交付为已完成高速平地分支：`models/v6_speed_candidate_19837_20261005/policy.onnx`，SHA `294d80c8618fdc4baced6587c3c36edd8d97cfba3d99afd05b3639c681e8b738`。它是speed←flat，不包含本轮跳跃修复或独立recovery技能；35组ONNX数值向量与文件身份核验通过。交接见 `docs/V6_REAL_DEPLOYMENT_HANDOFF_20261005.md`。当前RMCS源码实际200Hz新反馈PD、1kHz持有重发，实际安装包未核实；不沿用“1kHz发送就是1kHz反馈闭环”的表述。
