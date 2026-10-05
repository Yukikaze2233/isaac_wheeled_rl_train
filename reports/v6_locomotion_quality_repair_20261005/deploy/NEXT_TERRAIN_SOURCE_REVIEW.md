# 下一 terrain 父源与复习覆盖静态审阅

本记录只复核本地保存的冻结源码、合同和原生回执，不代表下一 terrain 已启动。未修改源码、合同或 launch；未运行 SSH、GPU、训练、评测或部署；未提交。未观测到遗忘，也不声称确定性/综合策略/硬件资格。

冻结源 `b4ce80964d47c0127e0b3d9929ad35a620bbf204`；本次逐字节核对以下本地源码/合同等于该 commit 的 Git blob。正式启动回执保留同源目录和父树：[launch.json:77](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_task_campaign_20261001/deploy/launch.json:77)。

**来源事实。** campaign 的 speed、terrain 均以 flat 为声明父域；调度先扫描各自域 `attempt_*/train/**/*.checkpoint.json`，优先采用本域 learning_updates 最大的封存候选。有旧 terrain 时继续它；没有时才用 flat 父域候选。完成 speed 仅推进调度顺序，不将 speed 新权重/critic/Adam 传给 rough。
证据：[v6_task_campaign_v14.json:1174](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/contracts/v6_task_campaign_v14.json:1174)；[run_task_campaign.py:424](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/scripts/run_task_campaign.py:424)；[run_task_campaign.py:578](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/scripts/run_task_campaign.py:578)。

现有已认证 terrain 来源回执：`/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train/terrain/attempt_000/train/model_final.pt`；PT SHA256 `fe429597ebb565644b2774747f558a8f70fbea53e033819096bbf9fe2a927134`，旧 v6 terrain 合同 SHA256 `4337842f9689ec2a4ecd0994dfb39ef821c95a9868a91f445de52f2a0afc456e`，learning_updates21113。该 SHA 来自已保存认证回执，本次未重新读取远端 PT。v14命名迁移保留旧 terrain actor1186。
[native_terrain_014.summary.json:14](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_locomotion_quality_repair_20261005/deploy/native_terrain_014.summary.json:14)；[v6_task_campaign_migration_v14.json:9](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/contracts/v6_task_campaign_migration_v14.json:9)。该原生探针强制时钟2186、10s、正式更新0，只有工程接线信用，不是正式 rough 接续或质量验收；实际新 worker 的来源须再看其 invocation/seal。

**复习覆盖。** 原 flat 有41组；terrain仅保留其中20组、缺21组。terrain总43组 = 17terrain + 20flat + 6release。

| 行为池 | 配额 |
| --- | ---: |
| 普通上下阶/攀爬（untriggered_steps） | 57% |
| 坡/粗糙/材料穿越 | 19% |
| 九档静站 | 8.55% |
| 连续高度静站 | 5.70% |
| ±0.5、start_stop_05、rotate_1及高度/反向变体 | 4.75% |
| translate/yaw/combined release及height变体（6组） | 5% |

平地普通复习合计19%，terrain76%，release5%（release可在混合地形上，不全计入平地）。[v6_task_campaign_terrain_worker_v14.json:16738](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/contracts/v6_task_campaign_terrain_worker_v14.json:16738)。

缺失的21个旧flat组：`backward_1`, `backward_1__height_train`, `backward_continuous_low`, `backward_continuous_low__height_train`, `curve_low`, `curve_low__height_train`, `curve_low__reverse_train`, `curve_low__reverse_train__height_train`, `forward_1`, `forward_1__height_train`, `forward_continuous_low`, `forward_continuous_low__height_train`, `height_scan`, `start_stop_1`, `start_stop_1__height_train`, `velocity_curve_1`, `velocity_curve_1__height_train`, `yaw_continuous_low`, `yaw_continuous_low__height_train`, `yaw_continuous_low__reverse_train`, `yaw_continuous_low__reverse_train__height_train`。没有 forward2/3/4/5、对应高速曲线或高速rps复习。actual `skill_specs` 的声明vx最高1.5m/s、yaw最高2rad/s（release组）；`stages.vx_max=5`、`yaw_max=3rev/s` 是宽上限，不能当实际训练命令覆盖。原生 `skills.sample()` 完全替代旧随机采样，并直接return。
[env.py:1002](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/src/wheeled_tasks/chassis/env.py:1002)；[v6_task_campaign_terrain_worker_v14.json:16152](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/contracts/v6_task_campaign_terrain_worker_v14.json:16152)；[v6_task_campaign_terrain_worker_v14.json:16746](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/contracts/v6_task_campaign_terrain_worker_v14.json:16746)。这些是未复习能力的结构风险，尚不能认定已遗忘或承诺不遗忘。

**机制与时钟。** speed/terrain v14 逐值相同：35D actor/81D critic/6 actions、机械与control SHA、wheel response、160/2.5腿PD、50Hz policy/200Hz physics、command slew1.5m/s²/4rad/s²、motion envelope、v14静止L1与姿态floor、PhysicalSignalDelay、command_transport和release timing。非标准出生配置仅origin不同。已有terrain原生runtime读回以上模块启用；`usb_transport=false`是legacy模块，command_transport=true；不得把工程pipeline age称为独立实测USB反馈延迟，也不得声称完整1kHz反馈。
[native_terrain_014.json:541605](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/reports/v6_locomotion_quality_repair_20261005/deploy/native_terrain_014.json:541605)。修复机制继承 ≠ speed新增学习权重继承。reference_reward.visible_reference_only=true；普通地形pitch支付目标为0，不能称保留加速度pitch补偿。姿态项是软成本。

下一terrain计划actor1186→2186，即新1000actor，critic适应另计消费。delay使用绝对本域actor0→1000，所以此接续一开始已全强度。出生以1186为origin，新1000actor终点按500→1500线性插值，nominal/gentle/wide = **70/20/10%**；未走到新1500actor的65/20/15%。
[v6_task_campaign_terrain_worker_v14.json:16842](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/contracts/v6_task_campaign_terrain_worker_v14.json:16842)；[v6_task_campaign_terrain_worker_v14.json:16848](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/contracts/v6_task_campaign_terrain_worker_v14.json:16848)；[nonstandard_start.py:91](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/src/wheeled_tasks/chassis/nonstandard_start.py:91)。

**下一命名迁移要求。** 当前父树符合多个独立专精任务的调度形式；用户授权多个任务，不等于当前已经累加成同一策略。若目标改为将speed新权重传入rough，应明确新父PT/seal/SHA与全学习状态恢复、terrain actor时钟/出生origin/消费账本映射以及复习覆盖，冻结并验证新的命名迁移后才能运行。不能只换`--resume`：当前跨域恢复严格要求terrain声明的旧flat源合同，speed v14源会被拒绝。
[task_campaign.py:515](/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule/src/wheeled_tasks/chassis/task_campaign.py:515)。本记录不实施迁移；下一正式启动回执仍需认证实际来源、合同、冻结源与恢复结果。

| 本次核对的冻结文件 | SHA256 |
| --- | --- |
| contracts/v6_task_campaign_v14.json | `41319993aa53a54e32efaab725ec8b0e269f4b8b3576bc101f23c6e525c4cf73` |
| contracts/v6_task_campaign_terrain_worker_v14.json | `35014b99f4c8ac7423df386bdddf809699c10fc75be56e6a73002f8f532be1d0` |
| contracts/v6_task_campaign_speed_worker_v14.json | `f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3` |
| contracts/v6_task_campaign_flat_worker_v6.json | `666f7bdbba8c305d4bedc464fd1a42ea8725245461c49d186b75ed37fe58aae7` |
| scripts/run_task_campaign.py | `ff6f7a480aed025c52e7a52db87bf4c5bb35475de73a72730b96258009546b12` |
| src/wheeled_tasks/chassis/task_campaign.py | `8879d1c4608a53aa9d275c7f92ac5dd9ec7ab53317a6f24801903b7211e3f576` |
| src/wheeled_tasks/chassis/env.py | `2623375260725f56e4cc5ee8b1ab44b76a3e699b43bba81762c630cd4910433c` |

旧terrain摘要回执 SHA256 `b60c731a7aac52503cb61ce0edf6e345619b8ebaabad946bbdb1b6edb1333b19`。
本次CPU只读断言通过：冻结blob一致、父域、合同SHA、41/43/20/21计数、机制逐值相同和原生engineering模块读回；未新增动态验证。
