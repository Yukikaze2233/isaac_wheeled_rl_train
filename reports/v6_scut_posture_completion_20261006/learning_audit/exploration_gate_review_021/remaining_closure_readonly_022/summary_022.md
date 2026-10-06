# source022 剩余首出生闭链护栏：实际只读分析

结论：881／2121当前在 **CAPTURE**、1877在 **ORBIT** 超过3mm。新SIDE1.5和CAPTURE→PREPARE gyro<.75已实际执行；这些剩余事件不是两项修复未接入的证据。三条在超线及终止时均为脚本控制、blend=0、pure_rl=false、whole-actor mask=false，不能归为本窗PPO探索动作，也不能用脚本生存替代自起成功。

分析只读已完成的SOURCE_SCRIPT_001（source723732、v20、历史REC3076 mean），4096×600策略／2400物理步，PID230688真实exit0／410.175s。原生产r22的当前actor不是这个历史诊断权重。出生／私有采样RNG／std身份、完整contact record多重集对照、preclose恢复由原source native readback通过；旧ordered contact拒绝保持。3mm、64/32、PD160/2.5及资产／奖励／出生未改。

| lane／首出生实际课 | 当前首个5ms超3mm | 当前50Hz终止 | 当前阶段及先后事实 |
|---|---|---|---|
| 881／L2 front_down_90 | phys887／4.435s，L_spring_upper_mount15.3314mm | policy222／phys888，27.6201mm | CAPTURE／route2。phys886原生L_link1→terrain0法向记录合计4109.220N，物理后L髋dq−90.299rad/s；phys887用这份反馈，原请求250.842Nm，帽40Nm，下一物理后闭链超线。原腿参考最大约4rad/s持续前进。 |
| 1877／L2 step200_lip_near_shell | phys411／2.055s，R_rocker_P9.9031mm | policy104／phys416，9.0361mm | ORBIT／route3。phys410 RR_link1→terrain0法向记录合计31.4704N，物理后RRdq73.4759rad/s；phys411原请求−148.867Nm、帽40Nm。其后四帧闭链回到3mm内，phys416又超线，不能把首次瞬态超线当立即终止。参考四根仍±4rad/s。 |
| 2121／L1实际step200_upright_top | phys1003／5.015s，R_upper_E3.31498mm | policy269／phys1076，3.75980mm | CAPTURE／route3。phys993 base_link→terrain0法向记录合计15082.380N、gyro51.1043rad/s、腿dq峰79.5061rad/s，但当帧pin仅.13366mm。50ms后首次超线前20ms无terrain patch，腿反馈最高约33rad/s；髋原请求224.5→120.5Nm持续限40Nm。首次超线后69帧回到3mm内，终止更晚。 |

全域实测最大仍是881／birth1／phys888／pin2，27.620118mm。旧SIDE-only的881 phase7及2121首phys998／终phys1004不适用于source022；新gyro门使其留在CAPTURE，并改变2121的后续轨迹。1877在ORBIT阶段就故障，追加CAPTURE门不能直接修这个更早事件。全域首次源敏感改变在reference149／input physics148，之后共享reset/RNG可能分叉，因此旧晚期对照只作描述；没有冒称三条晚期逐位相同。

实际闭链gap由六个原生世界端点对计算并已捕获，不是FK高度代理。原生terrain patch、dq、请求、限幅、下一帧gap的先后关系是可证实事实。原请求超帽不表示执行超帽：真实40Nm腿帽／4.5Nm轮帽保留，setter／engine effort记录也保留。force×dt仅是派生冲量；这些时序不能独立分解接触、目标轨迹、闭链负载和求解器误差的因果份额，不能把body支撑接触称为非法碰撞或唯一根因。

下一最窄研究候选是只把实际 `capture_speed_rad_s=4` 改为 `1`，检验881／2121在CAPTURE继续收拢的速度是否增加落地后的闭链负载；它复用已有native配置键，不改目标几何、耦合比例、gyro门、1s稳定门或力矩帽。该值只是可检验假设，不是已证明修复。1877的ORBIT4保持，以免一个试验混入另一阶段的改变。若CAPTURE减速没有改善或只把失败拖到原8s超时，仍按负结果保留，不延长预算。

以后若单独研究1877，应先另开ORBIT4→2的窄候选：原 `orbit_angle += speed*dt` 和整圈判定自然计时，记录原8s预算下是否更早退出／超时。不能在当前CAPTURE试验同时改ORBIT、轮平衡、PD、资产、solver或奖励；速度降低也不能证明接触已被正确解决。

验证应保持同封存REC3076／同初态和私有采样流，复用唯一native600／2400原loop，不增加policy／decode／FSM／reward调用；比较全4096首次CAPTURE速度敏感改变前的完整state/control/contact record多重集，之后selected轨迹只作描述。881／2121首出生全子步≤3mm、实际帽保持、接管／release／success／失败／timeout／右删失独立记账；同时报1877及全域峰／guard，避免选定两条改善隐藏全域失败。只有脚本机械改善不授予1s稳定、原链成功或实车资格；需要后续正式学习窗口及独立功能门验证。

时钟：当帧原控制／FSM使用physics k−1的反馈，terrain contact／gyro／六pin是physics k后；FK/IMU/probe support缓存也是k−1，不能替代k后真实contact。跨phase或release重置的参考有限差分可能含reference重新初始化，不应当作物理关节速率；表中约4rad/s仅取故障附近同一CAPTURE／ORBIT连续段。

范围限制：没有独立连续编码器winding、真实随机mass/COM/friction draw、自碰撞pair identity或native overflow flag；旧stochastic730mm全峰未被定位到这三条。未改原报告／源码／合同／正式进程，也没有GPU／SSH／PPO／额外测试或新native。

数据：`actual_remaining_022.json` SHA `31dbdd10bbd3bccf098ff08a7dd91b2d6be0943844ee2ff52f1c5ed3112fc39a`；完整第一超线与终止窗口、source input SHA、当前phase统计保留。`actual_extreme_anchors_022.json` SHA `e8d182af08656948996a3f53e212a8a002222379e36efbe0c96b180716756c73` 记录三条实际contact／gyro／dq极值前后20ms，非独立第二次模拟。全部输入只读前后SHA一致。
