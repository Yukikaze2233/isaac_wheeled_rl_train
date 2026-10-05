# Terrain v15 独立采样与投影审查

CPU物化与离线proposal逐值一致：85组/4096人，旧43个spec及scene非fraction字段全部保留，完整flat41常驻1393人。只改四采样字段及命名revision/design绑定；机械资产、奖励、控制、延迟、出生origin1186、其他五worker、父树、endpoint2186和消费上限保持。v14与真实旧v6→精确v14→v15两条同域构造均认证通过，乱序/改high地形/额外别名/改奖励、时钟、physics及出生origin均拒绝。

| 行为池 | 原生4096人数 |
| --- | ---: |
| 原12上/下阶 | 1716 |
| 原5surface | 410 |
| 新5surface-height | 165 |
| 九档静站 | 369（各41） |
| 连续静站 | 246 |
| 高度轨迹 | 103 |
| 基础平移/曲线 | 245 |
| 基础正反yaw | 184 |
| 加减速/停车轨迹 | 246 |
| 平地高速直行 | 208（16×13） |
| 原松杆组 | 204（6×34） |

新增high仅平地±2/3/4/5直行×名义/高度，未加高yaw、spin、high曲线或5m/s松杆，不能称完整135组保留。surface-height绑定各原坡/粗糙/材料；h26–36cm及vx.2–.8/yaw±.1在原声明/差速命令范围内，静态合法不证明支撑、力矩、气簧和姿态动态资格。terrain人口3118→2291，复习与地形学习存在曝光竞争，不能预测无遗忘或更快。

**口径更正建议：** `start_at_target_fraction=.5`只令目标高度指令直接从采样目标开始，普通height组`reset_pose`不把物理车体放在该高度。25%nominal仅保留平面目标，height仍独立采样。未新增group-scoped难度推进，不可用全局clock伪称terrain本域渐进。

审查发现并由root修复实际fixed-suite投影：原函数保留85个训练spec/membership，和独立旧43spec/28case期望不符。新`evaluation.py` v15分支只从SHA绑定terrainv14恢复`skill_specs`及`behavior_pool_membership`；对相同28case输入分别执行b4原函数与新函数，差异仅这两个metadata字段。evaluation case/scene、传感器、物理、PD、reward、delay及fixed角色投影全部不因该分支改变。五独立design role中train85组，四fixed角色仍旧43spec/28case及原runtime。

两项初测失败也已改正：pool浮点sum用容差；严格命名修订拒绝flat与原合法声明flat父树的通用跨域resume分开测试。`learning_source`最新terrain seal优先测试通过，不能把合法首次parent等同于允许替换已有terrain学习状态。当前focused **52 CPU tests通过**。**真实30roles尚待root的clean archive；实际GPU85人口、物理/学习效果和部署未由本审查执行或认证。**

部署建议：speed自然冻结终点后单次命名迁移；若terrain已起，先合法封存实际完整actor/critic/Adam与消费，再用新冻结源恢复实际v14seal；尚无v14训练时才走真实v6seal→精确构造。v15不reopen、不清时钟、不退款；U<2186只执行剩余actor及原生计费critic适应，U≥2186不自动加训练。

详见`independent_sampling_review.json`的源SHA、两前驱认证、实际分配、字段差异和限制。本次没有SSH、GPU、PPO、部署、core/合同改动或提交。
