# V6原35D接续：Kaiser部署回执

## 冻结身份

- 启动时间：2026-09-26 22:37:11 UTC。
- 训练源码commit：`9651b0b93bbcb58cd7e0214c2449932dfc2880e4`。
- 源码归档SHA-256：`c76c56d6541525c3db5171912c362650f9b046f1cc9113abe33bf7d6de26457d`。
- 未提交overlay：空。
- 合同：`contracts/v6_scut35_continuation_v1.json`。
- 父模型：原V5.9累计14625，SHA-256 `db52492bd7af732255b010ffa4f6256904c910e0648367a2b481be44587e7871`。
- 正式run：`/home/kaiser/robot-rl-sim60/experiments/v5-scut35-20260926T223706Z-cd8a7d`。
- tmux：`v5-scut35-20260926T223706Z-cd8a7d`，socket `/home/kaiser/robot-rl-sim60/tmux.sock`。
- 本地启动回执：`reports/v6_scut35_formal_20260927/launch.json`。

## 正式配置与启动核验

已从远端物化合同及进程核验：actor35、critic81、1ms物理/主机反馈周期、20ms策略周期，
P0为16384环境、500次更新，统一评测335个固定用例。
七段全部16384环境，7000次实际更新上限、2,752,512,000 transitions，包含175次critic-only warmup。

初次核验状态为`command_repair / fixed_evaluation`：正式PPO更新为0，正在执行父模型双seed基线。
supervisor、worker、tmux、独立归档进程均存活。基线完成后编排器自动进入训练。
这里不把容量探针的8次更新记入正式更新数，也不把启动成功记为技能通过。

## 实测容量与验证

- 128项相关CPU测试通过；扩大并行后的合同/编排14项检查通过。
- 本地真实1kHz仿真：64环境PPO4次更新及恢复2次更新通过，35D ONNX数值校验通过。
- 真实场景：15/25cm台阶尺寸正确，31次partial reset检查通过，坡道出生前200ms有效。
- Kaiser容量：16384环境8次更新，其中7次actor更新；约10883 transition/s、36.13s/update。
- 峰值显存11879MiB；最小WSL可用RAM9175916KiB，最小Windows宿主可用RAM1938392KiB。
- 容量回执：`reports/v6_scut35_capacity_20260927/report.json`。

## 监控

- TensorBoard服务端已按用户要求迁到Kaiser，本机只运行SSH端口转发。
- 本机访问：`http://127.0.0.1:6006`；训练标签`v6_35d`。
- Kaiser服务：tmux会话`v6-35d-tensorboard`，使用上述同一tmux socket，监听`127.0.0.1:6006`。
- 远端日志入口：`/home/kaiser/robot-rl-sim60/tensorboard-v6-active`，其中`v6_35d`链接到本次正式run的`train`目录。
- 服务参数：`--load_fast=true --samples_per_plugin=scalars=1000 --reload_interval=30`，使用Rust数据加载后端。
- 本机转发服务：`v6-35d-tensorboard-tunnel.service`，将本机`127.0.0.1:6006`转发到Kaiser的同名地址；启用SSH保活及断线自动重连。
- 原本机TensorBoard服务`v6-35d-tensorboard.service`和event镜像服务`v6-35d-event-sync.service`已停止。
- 历史V5.9日志约732MiB，一次性加载全部历史时服务占用超过4GiB内存；常驻面板改为当前V6，历史原始event仍存于Kaiser原run目录。
- 迁移核验：本机6006监听进程为`ssh`；HTTP后端日志路径位于Kaiser，Rust后端与Web服务合计RSS约102MiB；当前训练run已可见。
- 迁移核验时，训练worker正常，正式更新数为60。图表服务和端口转发独立于训练生命周期。

状态查询：

```bash
python3 -B scripts/check_chassis_remote.py reports/v6_scut35_formal_20260927/launch.json --brief
```

训练计划及参数定义见`docs/V6_SCUT35_CONTINUATION_20260927.md`。

本run随后在第100次更新正常暂停，保留模型与优化器状态，转为200Hz继续训练。
新run及恢复证明见`docs/V6_200HZ_RESUME_20260927.md`；TensorBoard使用`v6_200hz`标签显示后续更新。
