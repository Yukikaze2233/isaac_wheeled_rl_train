# V8 → 真实物理接管初态课程：只读可实施性审计

结论：旧库可以在 episode reset 采样静态姿态，**不能直接恢复 002 的动态终点或完整原生仿真状态**。最小候选改法是新增带速度的出生 sample，复用原生 reset 写入和控制链，从实际物理初态重新执行 SELECT/PREPARE、ready、BLEND、接稳与后续请求。不要把源轨迹完成记录载入新 episode。此处未修改训练 core/合同、未启动 GPU/Isaac、未执行 restore、未动态验证，也不授予资格。

当前身份以实读 `reports/v6_task_campaign_20261001/deploy/launch.json` 为准：v14、冻结源码 `b4ce80964d47c0127e0b3d9929ad35a620bbf204`、speed attempt_002；recovery 仍是 V8 worker 与独立 flat 父树。AGENTS 的旧 v10/v8条目只提供协作边界。审计脚本核对五个涉及模块当前字节与 b4 完全一致；main launch、旧 worker/姿态库、002原始事件与 PT 快照的 SHA 在 [evidence.json](evidence.json)。读取的是已完成的旧 flat actor诊断，不是 live speed actor。

## 现有入口与确证限制

| 入口 | 源码证据 | 实际能力/限制 |
|---|---|---|
| `RecoveryResetCurriculum` | `recovery_curriculum.py:81–89,117–146,188–210` | 只在 reset 采样离散 full18DOF姿态；`interpolation=linear`只插值 easy/medium/hard **概率**，没有插值机器人姿态。easy/medium按相同 terrain/平台高取样，hard用各 case 自己的原姿态。 |
| 初速度 | 同文件 `131–134` | 任一 root线/角速度或关节速度非零都拒绝。CPU 实际将002非零 COM速度/dq植入候选表，两个分支均按预期拒绝。 |
| `RecoveryTraining.initial_state` | `recovery_training.py:164–176` | 载入root/q、叠加目标 lane origin，强制 root COM速度为0；未提供dq输出。 |
| `ChassisEnv.reset` | `env.py:900–995` | `921–924`四个公开 API写root pose、COM世界速度、q18、dq18；dq18无条件写零。`925–990`重置响应、运输、actions/history、传感器、episode/命令时钟及FSM。 |
| 原生 FSM 出生 | `recovery.py:136–164`；`recovery_training.py:178–195` | active → SELECT，ready/稳定计数、失败与运动解锁清零；zero-release即可从真实近直立姿态重新走PREPARE。需要初始释放时 raw RL9仍是无使能占位。 |
| 已安装 IsaacLab PhysX setters | `articulation.py:445–488,680–722,855–922,1040–1088` | xyzw的root-link pose、**root COM世界**线/角速度及joint q/dq可写入。`reset:222–246`重置actuator/wrench；这些 API未恢复接触流形、solver impulse/warmstart。只读核对了实现与SHA，未调用。 |

## 002 快照能用于什么

`native_prefix_002/event_snapshots.pt`实际有68个全batch快照，与68事件索引一一对应。14个成功case相关快照经CPU核验必要字段有限、shape正确、四元数近单位、18关节集合与资产一致。root姿态位置必须减去**源origin**再加目标origin；速度使用 `physics.root_com_lin_vel_w` 与 `physics.root_com_ang_vel_w`，关节按 `report.native_joint_names`重排，不能把18列默认当另一个资产顺序。

| source event（lane2仰腹 / lane1右侧） | 原时刻 s | COM平面速度 m/s | gyro范数 rad/s | dq18最大绝对值* | 倾角 ° |
|---|---:|---:|---:|---:|---:|
| motion_unlock | 6.220 / 6.680 | .00862 / .00371 | .01944 / .02151 | .20962 / .04130 | .461 / .324 |
| first_pure_rl_post_physics | 5.190 / 5.615 | .07116 / .13646 | .15764 / .19970 | 4.86695 / 6.10941 | 3.066 / 4.231 |
| blend_last_mixed_post_physics | 5.185 / 5.610 | .07539 / .14065 | .11973 / .18687 | 5.02810 / 6.27651 | 3.035 / 4.186 |
| blend_entry_pre_effort | 4.985 / 5.410 | .15883 / .23777 | .27026 / .17327 | 2.31434 / 3.78296 | 5.965 / 6.890 |
| phase_PREPARE | 3.840 / 4.280 | .22345 / .52946 | .46233 / 6.86546 | 2.57116 / 20.17361 | 32.661 / 19.094 |

\* 18维混合转动/移动关节，数值仅描述原buffer，不赋予统一rad/s单位；使用完整关节名与joint类型。motion_unlock是已完成接稳后的物理初态来源，新lesson必须将source的 `true_stable_peak≈1.04/1.08s`、解锁/完成状态清零。first_pureRL的源 `learning_mask=false` 也直接证实“首个纯RL物理子步”不等于整个policy interval可学习。

可用的其他公开字段包括 encoders/winding、IMU/支撑探针、FSM、动作历史/35D81D消费观测、transport队列和response状态。它们用于审计来源和解释原轨迹，**不构成完整可复原状态**：

- PhysX接触、约束/求解器warmstart、Scene内部状态没有导出。
- 未导出episode/command/height/push时钟、raw_height、recovery handover_count/ever_blended/wheel_balance_active、FSM enabled及若干内部物理导数buffer。原报告的missing列表包含出生时尚未创建的v5 effort/bounds，后续快照里存在这两项；不能据missing列表把它们说成始终缺失。
- applied事件在physics/update后、接触读取/下一反馈追加前；reference-entry事件在torque前。snapshot中消费观测是本policy调用的输入，cache是对应子步的证据，不能不加边界定义就作为一个完整Markov continuation。
- transport `pending/valid/generated/pending_tick`的lane维是轴1，其他多数字段是轴0；tick和generator是全batch共享。不能用通用 `tensor[lane]`复制所有状态，也不能为一个reset lane回滚共享时钟/RNG影响其余lane。
- 普通200Hz frames只有root-link线速度和body gyro；实际 root-link 与 COM线速度差可达 .3484m/s（右侧PREPARE快照）。不能原样喂给COM世界6D setter。要在40个BLEND槽任意选初态，应新增相位明确的完整COM世界速度快照；不能插值相邻位置/速度来伪造新的物理终点。

## 最小实现接口（建议，尚未实施）

新增版本化 `RecoveryBirthSample`，CPU bank校验与GPU tensor gather分开。必要字段：`asset_manifest_sha256`、机械/solver/材料/响应/控制合同SHA、`joint_names[18]`、`root_link_pose_local_xyzw[7]`、`root_com_velocity_world[6]`、`joint_positions[18]`、`joint_velocities[18]`、terrain/平台高、`source_run/event/snapshot_index/physical_tick/sha256`及`entry_mode=physical_seed_replay_from_select`。样本的来源完成标志仅作provenance，永不当新episode统计。

1. `sample_birth(ids, domain_actor_update, generator) -> sample, profile_id, level_id`：仅reset lanes，用冻结绝对actor origin调整离散池概率，硬条件匹配资产/terrain；不得重写live lane。
2. `initial_state(ids, root, q, root_velocity, joint_velocity)`：延展当前出生事务的dq输出；写入root-link pose、COM世界速度和full18DOF q/dq一次。profile需要新schema，不能静默放宽V8冻结库。
3. `reset_software_from_birth(ids)`：保留当前native reset语义，运输与响应清空/按新episode条件重采，actions/history及encoder/IMU/probe从实际新物理初态建立。FSM SELECT，所有ready/stable/true_success/followup/elapsed/handover统计清零；不复制source的RL phase、elapsed、hold通过、已解锁或奖励记账。这是动力学初态重播，**不是精确snapshot resume**。
4. 仅原生env推进：READY真实.1s，BLEND真实40×5ms，sensor与truth各1s接稳，再执行新followup。新增seed provenance、实际重新交接与complete-window计数接入原生TB；full-chain from fallen birth 与 seed-lesson completion分开记，不给seed课补记被省略的自起前缀。

这样最小差异落在 bank/schema 与出生q/dq接口，而不要求提前强设FSM为RL。若另设“直接actor接稳”入口，需要单独命名lesson、真实reset适应期/whole-interval mask与独立结果分母；不能沿用当前full-chain success标签表示已经执行了完整脚本。

## 难度与验证不变量

先试源motion_unlock的近静止物理姿态，之后试first-pure/last-mixed残余速度，再逐步加入blend-entry和较早PREPARE样本；这只是按记录的速度/倾角描述候选梯度，难度顺序须动态测定。保留普通flat41组、原始跌倒初态、同平台池和旧fixed评测；两个成功flat/stand样本不能代表左侧、仰背、台阶或运动followup已可用。旧V8概率迁移不改已经消耗的actor/critic时钟或预算，不按wall/event step推进。

CPU/静态可验证：资产/joint映射/SHA/有限/shape/四元数、固定reset IDs和非reset不变、全18DOF与spring/闭链/碰撞检查、terrain/平台匹配、概率与恢复actor origin、reject nonzero velocity on旧V8 schema。已执行的检查见回执；新的bank/schema接口尚未实现。

之后必须原生验证：出生前/后读回 q18/dq18/rootlink/COMworld6与sample一致；只写一次且之后无teleport；20ms interval内4次fresh反馈PD；transport/response/reset实际条件可见；PhysX重建期间不能伪造相同后继；40真实BLEND子步、sensor/真值连续dwell及未提前解锁；第一批固定分母、失败/timeout/右删失与autoreset前状态齐全。脚本/混合/初始释放整个interval不进PPO，actor拥有的终止仍计惩罚；`RecoveryTraining.begin_policy/torque/bootstrap_mask`（206–212、315–322）与 `RecoveryRolloutMask`（71–125、143–150）保留分别判定elapsed interval学习资格和returned observation bootstrap，并在GAE边界隔离。报告若只做到静态预检应保留 `qualification=false` 与 `physical_restore_executed=false`。

复现只读CPU审计：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /home/yukikaze/isaacsim60-venv/bin/python reports/v6_locomotion_quality_repair_20261005/recovery_audit/state_seed_audit/audit_reset_seeds.py
```
