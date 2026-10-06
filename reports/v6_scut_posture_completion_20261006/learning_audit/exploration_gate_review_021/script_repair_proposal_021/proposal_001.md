# 前段自起脚本两项修复提案

本提案只冻结候选范围和迁移验证要求。正式 source021/v19、生产合同和运行 PPO 均未修改；本轮没有运行物理、SSH 或测试，也没有消费正式更新。原生证据来自既有 REC3076 mean 的 0 PPO 本机诊断，不能直接成为新源码或实车的资格。

## 最小候选

1. SIDE 的速度显式从 6 改为 1.5 rad/s。优先在新恢复 worker 的现有 `recovery_training.side_speed_rad_s` 字段指定 1.5，保留原生 DEFAULTS 的 6，使未修改消费者维持原配置。保持角度 π/2、共同进度比例、轴符号、80 ms 驻留和原 `2θ/v+0.4` 超时公式。单程时长 0.261799→1.047198 s；阶段超时 0.923599→2.494395 s，阶段超时不是四倍。全局 8 s 脚本预算不延长。
2. 原生 `CAPTURE→PREPARE` 的 `tilt<20° && support && estimated_height>0.22` 条件，再 AND `gyro.norm<handover_gyro_max`，复用既有严格 0.75 rad/s 上限。该门检查进入 PREPARE 时的当前传感器状态；它不替代八项连续稳定门。其他转移、原 ready 条件与 100 ms 驻留、200 ms 混合、RL 连续 1 s 接稳、原目标跟随 1 s、一次 reroute 和全局 8 s 都保留。

这是两项脚本修改；不叠加奖励、熵、std、Adam、PD、求解器、质量、质心、摩擦、出生池或命令采样修改。慢 SIDE 和多等待 CAPTURE 自然改变脚本用时、真实触地与后续经历，这些变化必须实测，不能用“参数更保守”替代验证。

## 已有证据及边界

| 实际观察 | 原 SIDE6 | SIDE1.5 | SIDE1.5＋gyro 候选 |
|---|---:|---:|---:|
| 1708 首出生闭链峰 | 49.484 mm | 0.761759 mm | 已有描述性轨迹，未重新授整体资格 |
| 2216 首出生闭链峰 | 23.048 mm | 0.440510 mm | 已有描述性轨迹，未重新授整体资格 |
| 全 4096 全窗闭链峰 | 未作为本行同域比较 | 27.620118 mm | 27.620118 mm |
| L2 original_chain 首出生成功 | 2/89 | 2/89 | 3/89，仅描述性计数 |

SIDE 诊断完整 600 策略步、2400 物理步，进程真实 exit0；两条选定轨迹 SIDE 前 180/177 帧逐位一致，耦合参考比例保持。降速消除了这两条的护栏触发，降低了所记录的轴速和接触冲量峰。它没有解决整个恢复域：1708 随后在 PREPARE 用完 8 s 脚本预算，2216 虽进入 pure RL，12 s 截止仍没有 motion release 或成功。

加入 gyro 条件的独立进程也实际 exit0，但原严格回读拒绝必须保留。首次全域介入前，两条所记轨迹的接触槽位在物理步 319/352 已置换；原 ordered-slot comparator 判为不一致。新窄描述 admission 核验完整关联 contact record 的类型和浮点位集相同、实际状态/控制/合力相同，没有加浮点容差或独立排序各字段；它只覆盖所记录两条步 0..530，以及已有出生、事件和峰记录。它不证明全 4096 未保存的动态历史，原 `strict_whole_reproduction=false`、`counterfactual_admission=false` 与 qualification=null 均保持。

该描述证据中，1708 已进入 pure RL，2216 仍在 pure RL；两条 12 s 都未 release、未完成原任务，均右删失。稳定峰分别约 180/155 ms，仍短于 1 s。3/89 对 2/89 的单次 L2 计数上升不是已认证功能增益，也不是 PPO 学习效果。全域 27.620118 mm 越界没有消失。

残留 1877 在 ORBIT 的物理步 411 已触发 3 mm 护栏，R_rocker_P 首越界 9.903078 mm；此前并未到 CAPTURE。追加 CAPTURE 的 gyro 条件不直接作用其首个故障。881/2121 的 PREPARE 越界也不能因两条选定轨迹改善就记为修复。接触、轴速、请求力矩、限幅与闭链误差的时序是诊断线索；尚不能宣布非法接触、唯一求解器问题或唯一目标跃迁原因。

## 共同源码的影响范围

冻结 v19 六 worker 均构造恢复模块，但只有恢复域选入 54 个 `recovery_pose` 组，共占 65%；另五域实际 `scene_groups` 的恢复组数均为 0，物理步有 `recovery.has_tasks` 分支。因此目前正式样本的直接脚本改动面是恢复域，恢复域内普通保留组的控制应逐值保持；普通非标准姿态出生使用独立模块，不能把它混称为这段脚本。

SIDE 修改只直接改变 SIDE 路线；随后转换、ORBIT/THRUST/CAPTURE、支撑估计、接管时刻及 shared reset RNG 消费可以间接受影响。gyro 谓词作用实际 CAPTURE→PREPARE，包括 ORBIT/THRUST 后的 CAPTURE；SELECT/PLANT 或 SIDE 转换直接进入 PREPARE 的路径不经该谓词，pure RL 接稳门本身也不变。源码是共享类；未来离线评测、新场景或其他消费者一旦选入恢复姿态也会受影响。故“其他五 worker 字节不变”不等于永久不受影响，须核实际场景投影并保留对应来源，不跨域拿恢复模型作为父模型。

## 后续真实源码迁移的最低验证集

1. **独立候选源码与合同。** 从 source021 冻结树建立新源码身份，只引入上述谓词；SIDE 用已有配置字段显式给新恢复合同 1.5。冻结新源码、资产、合同、预期和 migration 哈希，运行时核实际 FSM 1.5、原阈值及时间公式。旧 hook/旧 inventory 不冒充新源码。其他五合同、各域父树、预算、origin 不随修复修改。
2. **聚焦源码核对。** 重用已存在的 29 项源码和 19 项 hook 边界思路，验证默认旧路径、低/等于/高 gyro、同 tick phase 转移、一次 reroute、partial reset/异常，以及 θ/v 与耦合参考比例。做 native 源码与候选 hook 的等价核对；不用另一套控制器或把 observer 当生产实现。普通组、L0/L1 与其他转移应保持原语义。
3. **同父 0 PPO 原生复验。** 源码版与现候选使用同 REC3076 全 seal、4096 出生/private sampling RNG、50/200 Hz、solver64/32、原真实模型及 3 mm；完整 600/2400 覆盖，至少记录 1708、2216、881、1877、2121 的首出生逐 5 ms 状态/控制/六闭链 pin/实际接触，并保留全域峰的 lane、birth、时刻。未介入历史只能截至真正全域首次介入；其后共享 reset/RNG 可能分叉。接触槽位无序语义若成为新比较规范，须单独冻结并审查，只能比较完整关联 record，不改写既有拒绝回执。
4. **分开判定机械与任务。** 保留全部 guard、脚本失败、pure RL 接管、连续 1 s 八门、原跟随成功和右删失。不能用 selected 无 guard、进入 RL、12 s 工程覆盖或 L2 计数 +1 代替全域机械/功能通过。ORBIT1877、全域峰和 RL 接稳仍待独立解决；本提案不降低 3 mm/1 s，也不移除问题姿态。
5. **正式接续另行准入。** 待源码与原生复验可审后，按届时最新已终结正式恢复 seal 生成 migration，先验证完整 restore 和原合同 critic warmup/actor-clock 语义，再做已授权范围内的接续。REC3076 只用于诊断，不是应回滚到的正式训练父。普通五域的 startup 投影与父树核对可以 CPU 完成；无需为这项脚本修复重新训练它们。

## 全状态、预算与回滚

正式迁移必须在安全更新边界封存动态 latest parent，精确保持 actor、critic、Adam、六轴 std、环境/采样 RNG、stage/task/global/actor 时钟、全部账本和其他五域 seal。继承现有 `resume_critic_warmup_updates=10` 时，跨新合同迁移的十次 critic 更新全部计消费，actor 不前进；同版本恢复读取剩余 warmup，不能重做十次。若后续迁移合同另有设计，必须显式审查，不能由本提案隐式改写。

迁移失败可恢复旧冻结源与合法完整 seal；恢复学习谱系不退还已实际执行或预留消费，也不将全局消费时钟倒拨。保留失败 attempt、optimizer 实际步数、剩余预算与父来源；不覆盖旧结果，不热改 live 合同、不并行重复 worker、不自动重开已退休域。继续受各域 20k 与 campaign 总 121000 上限约束。坏状态不得伪封为成功 checkpoint；故障后的再接续需要新的、可审查的恢复回执。

本提案没有执行上述后续验证。文件输入哈希和精确证据路径见 [proposal_001.json](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_scut_posture_completion_20261006/learning_audit/exploration_gate_review_021/script_repair_proposal_021/proposal_001.json)。
