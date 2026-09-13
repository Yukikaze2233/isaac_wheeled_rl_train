# V40 第二轮最终证据与上游参考审计

日期：2026-09-12。本文为可随仓库保存的简版；完整独立报告位于工作区
`reports/round3_20260912/upstream_and_round2_audit.md`（相对本工作树为 `../reports/...`）。
本次审计重读本地原始数据，不代表重新执行远端仿真或验收实物。

## 最终训练确已完成

证据目录：`../reports/current/cqa1_round2_20260912/`。

- `pilot/completion.json`：20次更新；`final/completion.json`：追加19980次；
  `final/local_receipt.json`：累计 **20000**，完成时间北京时间 **2026-09-12 13:48:15**。
- 最终8项产物SHA256重新验证全部匹配；14个source hashes与冻结代码
  `453cd1ca86fc76a54654e622972f47dce6e23394`逐项匹配。
- ONNX SHA256：`7d4a93f0d0692f914dc769fb94c88550de62e44a0b3553d95b20d80d406b700e`。
- Checkpoint SHA256：`f94919f8f7ae1cfee95e45b922a9939e21c9f6ad9dff5656ff6f29dc0f4e6360`。
- Contract摘要：`667865de8352e67cff6720af9a142d00ff9b980132b27e4993a25c8e09d3d22e`。
- 最终checkpoint的`iter=19998`与main标签19…19998一致；恢复标签重复pilot末尾19，
  update实际计数以完成收据为准。`policy_quality_verified=false`仍保留。

重读main原始TensorBoard event，reward/vx/wz/height/std/非轮净力及接触比例共7个标签与
`curves/train-scalars.csv`逐点一致。最后100个update均值：

| 指标 | 数值 |
|---|---:|
| reward | 88.9074 |
| height MAE | 4.248 mm |
| vx MAE | 0.19598 m/s |
| wz MAE | 0.15892 rad/s |
| episode length | 1998.29步 |
| mean std | 0.59160 |
| 非轮净接触诊断比例 | 16.835% |
| 非轮净力统计量 | 77.606 N |

约5432窗口的非轮比例/净力约6.514%/15.993N，后期没有持续下降。
这些是训练随机策略聚合统计，不是确定性固定场景性能；净力也不是ground-pair支撑力。
std增长可能受熵激励与动作/目标/力矩限幅影响，不能独立证明未收敛。
episode/timeout不能证明全域稳定；v2许多诊断并不参与termination。
旧约37k samples/s是这次Sim5.1/Lab2.3/RSL3训练的历史吞吐，不是下一轮新栈benchmark。

## 最终ONNX确有原生GUI回放，但覆盖有限

外部证据目录：

- `../reports/server_final_onnx_teleop_20260912T2013/`
- `../reports/server_final_onnx_exploratory_20260912T203531/`
- `../reports/server_final_onnx_exploratory_20260912T204509-dd2e79/`

各6000帧/60仿真秒，3次timeout、0次failure reset。使用同一最终ONNX，Sim5.1训练→Sim6回放。
**实际命令均未超过vx/wz±1，高度全为0.30m**；界面上限5m/s、120rpm不是已执行范围。
普通teleop的CSV有21帧非轮净力>1N，峰值612N；两份exploratory分别6/12帧，峰值505/1092N。
旧summary的diagnostic_frames错误地累计了被v2屏蔽的termination_flags，因此全零不可信。
修正后的API及headless命令见 [SERVER_FINAL_ONNX_REPLAY.md](SERVER_FINAL_ONNX_REPLAY.md)。

## 上游必须区分当前源码、继承覆盖与已发布run快照

### 华南虎

Commit **`b8ff79f3df855faf9dc92f4a282bd80c42649466`**：

- [V14 task注册](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/__init__.py#L17)：Flat-v0、落地v1、自旋平移v2及Rough分支不同。
- [V14配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py#L468)：普通flat高度0.20–0.42m、absolute root height、站立桶10%；DR和延迟实际开启，默认`curriculum=None`，并非所有课程全开。
- [高度与站立奖励实现](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py#L3743)：有效tight高度核分母0.001m²，另有高度平方项；零vx时抑制平移，不要求yaw同时为零。
- [发布flat agent快照](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/flat_and_rotation/2026-07-29_23-20-03/params/agent.yaml#L18)：实际记录[128,64,32]，不同于当前源码[256,128,64]。
- [发布rough迁移记录](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_rotation_stair/2026-07-30_09-16-27/params/agent.yaml#L51)：从flat model_8000初始化，`resume_training=false`；不能说完整恢复optimizer。

### 复旦

Commit **`8204e853dfd2ed06d85a322e1a998c3d20a3be2c`**：

- [plane基类配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot_config.py#L90)：height命令0.10–0.20m；DR开启，push和action-delay关闭。
- [最终资产子类覆盖](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py#L66)：受罚/终止contact body列表为空，self-collision关闭；collision有scale也不产生惩罚。
- [运行时terrain覆盖](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot.py#L1597)：plane强制terrain.curriculum=False；命令课程仍可扩vx，非高度课程。
- [ActorCriticSequence](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py#L39)：不是RNN；125历史→速度latent3，当前25+latent入actor，critic141。
- [jump配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/jump/wheel_legged_gym/envs/base/legged_robot_config.py#L158)：另有推扰/跳跃奖励，不能并入plane默认能力。

## 下一轮方向与尚缺证据

保持本机0.28/0.30/0.32m，先比较4200、约15k、final的同一确定性矩阵表现，再分阶段改善
高低站立/行驶转换、温和DR/延迟、抗扰动；网络容量目前没有不足证据，地形可另开一轮。
当前高度reward核已接近上游，不能仅因上游项多就叠加更多约束。

本机2R髋膝210mm、膝轮250mm、轮半径60mm，真实膝内角35–80°；已有0.36m候选姿态
超过80°，不得抄上游高度范围。宏观2R等效不保证STL零件归属、耳孔随动、惯量/碰撞或encoder映射真实。
待补：高低位最终回放、持续漂移、±2速度域、推扰恢复、具体接触pair、目标栈短benchmark。
