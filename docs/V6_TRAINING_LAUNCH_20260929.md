# Kaiser V6 新资产完整课程启动回执

北京时间2026-09-29。本次已启动S1–S6完整仿真研究链，当前在S1；不是仅创建tmux，也不是全课程已经通过。

## 冻结身份

- 训练源码提交：`1f9a884368e772741c269567a1b4b3c21f371349`，分支 `dev/v6-asset-identification-20260929`，无未提交overlay。
- 源码归档SHA：`256304f5281f301c1affbe03d17c99e58ca2c5552e3192dcbc7dd5dede86538c`。
- 计划：`contracts/v6_new_asset_training_v1.json`，SHA `858bacb823ae372677df9ee67d8e70745857078c2faf7010f158e0c1dae68d51`。
- 独立设计表SHA：`b36d5ebb7e367a7bdef84df10143587623626030bc06362e6730373824eaf8c6`。
- 资产manifest：`875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba`。
- 远端目录：`/home/kaiser/robot-rl-sim60/experiments/v6-new-asset-20260928T190254Z-b07eff`。
- tmux：`v6-new-asset-20260928T190254Z-b07eff`，socket `/home/kaiser/robot-rl-sim60/tmux.sock`。

## 实际执行证据

[机器回执](../reports/v6_training_launch_20260929/verification.json)对应实读 **2次PPO更新、960,000条环境转移**，actor与critic均参与。第一更新40.57秒；第二更新采样39.051秒、优化0.287秒，约12,202转移/s。第二更新value/surrogate/entropy loss分别12389.4296/−0.0007/8.5324，均有限；这是从零学习初期，不表示功能已通过。

GPU采样13272MiB、利用率90%，系统可用RAM约6.8GiB；这些是首批运行采样，不是全课程峰值保证。20,000环境已实际带PPO运行。实际日志包含十项 `dense_*_broad/precision`，确认新的稠密基形进入计算。

运行构造实读：physics_dt=.001、policy_dt=.02、decimation20、PD160/2.5、wheel P.6、dense_tracking与command_reference启用；S1按设计关闭随机化和辅助。`precision_tracking=false`指被替换的旧精度项，新的dense精核实际启用。每策略步的SubstepControlAudit运行中断言20次PD/反馈刷新；构造期startup快照仍标pending_first_step，不能误称该旧快照已经含运行后计数，完整计数将在首块completion封存。

30种阶段角色的静态预检及CPU实际传输/随机化/课程探针已在**远端冻结源码**上通过，见[预检回执](../reports/v6_training_launch_20260929/remote_preflight.json)。CPU生产接口与新旧课程/恢复测试通过；最后更改预期表后单独重跑120项受影响检查通过。本机合并额外保留原网络研究改动，107项合并相关测试通过；这些额外网络改动未进入远端运行。

## 继续执行规则

S1首块40更新结束后，对真实0更新初始checkpoint建立双seed完整基线，再评候选。此后每40更新评测，阶段门通过后自动继续S2–S6；预算耗尽或保护回滚耗尽会保留best并停止，不自动跳过失败阶段。总预算8500，墙钟上限7天，回滚不返还消费。

当前使用标明来源的名义动力学与逐阶段随机化。打到止挡的数据保留分项辨识，未把尚未通过G1的六参数候选当真值注入。四主动轴连续，105°只约束机构相对膝内角；本轮不进行真机操作或自动硬件发布。

TensorBoard远端既有6006服务已增加 `tensorboard-v6-active/v6-new-asset-20260929`，读取此运行train目录。训练不依赖浏览器或SSH持续连接。

```bash
ssh -p 2222 -t kaiser@192.168.64.234 \
  tmux -S /home/kaiser/robot-rl-sim60/tmux.sock attach -t v6-new-asset-20260928T190254Z-b07eff
```

入口日志位于远端 `train.log`，首块细节在 `train/stage_00_S1/block_000.log`；总进度在 `train/progress.json`。本轮方案见[完整计划](V6_NEW_ASSET_TRAINING_MASTER_PLAN_20260928.md)。
