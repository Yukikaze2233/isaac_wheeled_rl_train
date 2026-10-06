同source712、seal2579／SHA`7f91d355…dbcca`两臂各完成4096×600策略／2400物理／12s，0PPO。逐lane出生group/lesson/profile与root/q/qd/velocity/origin所有物理tensorSHA、environment RNG及private sampling首末stream完全一致；savedstd逐值保留。原始文件SHA与完整核验见JSON。两进程exit0由root实际工具观察；两者均缺postclose sidecar，不能声称App.close返回正常。

| lesson／首生人数 | mean成功/失败/边界/未完 | stochastic成功/失败/边界/未完 | mean/stoch放行 | mean/stoch actual hold≥1s |
|---|---|---|---|---|
| prefix_stand／2284 | 908/99/0/1277 | 0/273/6/2005 | 979/0 | 1600/0 |
| prefix_release／327 | 66/32/0/229 | 0/47/2/278 | 140/0 | 221/0 |
| original_chain／36 | 9/15/0/12 | 0/15/0/21 | 15/0 | 16/0 |

当前同actor对照直接显示保存的探索动作在此12s窗口无法连续站稳并阻断运动曝光；不能推成所有20s训练回合必失败或历史退步唯一原因。stochastic actual峰max L0/L1/L2为.320/.300/.180s，sensorpeakmax.070/.050/.035s；不是仅略差浮点门。native门为≥1−1e−5，mean24条.99999958正常达门。

L0原moving目标1185条全变站立，含mean放行502、prefix成功459条；L0所有979放行仍零命令。L1mean有actor-valid forward25108帧、yaw834帧，L2forward821帧；stochastic三lesson全部0。目标桥接是明确设计修复，但单独不能帮助从未解除hold的探索策略；先校准真实actor-owned连续settling信号更符合当前证据。

mean5条sensorrelease未达实际1s（L1两条／L2三条）全部code7失败、success=false/followup0，安全拦截保持有效。mean actual-only达门708条与5条sensor-only不能净差混算。stochastic recovery未完2304条均为600帧右删失，不算失败；失败共335条另记。

以后仍保持最后sensor/contact/followup三个连续1s标准、完整学习状态和保存的探索，不重置std或改mean-only PPO。逐门边际不相乘；原始firstcohort/后续reset不同分母，native原始付款没有被该冻结wrapper保存，不能据此声称某reward已测得最弱。初态和sampling stream相同，后续环境随机流可因动作导致不同reset而分岔；完整因果归因仍需限定于此工程比较。
