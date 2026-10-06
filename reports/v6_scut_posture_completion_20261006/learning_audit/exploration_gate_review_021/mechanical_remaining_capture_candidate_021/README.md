# 三条剩余闭链 lane 的测量候选

此目录只准备下一次 0PPO 观察；root 尚未运行它。复用冻结003 capture、原SIDE1.5 factory/lifecycle、同源021/同REC3076 mean600/4096以及原生出生/RNG/事件/付款。正式训练与合同不改。

唯一对 actual SIDE001 的区别是私有 observer 保存集合：881/L2/front_down_90、1877/L2/step200_lip_near_shell、2121/L1/step200_upright_top。保留3mm guard、64/32求解器、.005s物理/.02s策略和2400子步；首出生终止后停止该 lane 的详细保存。每条初态、原setter参考/目标/反馈、真实姿态/pin以及现有地形接触到首terminal均保存。

measurement_only.patch 是原003 helper的私有副本差异：LANES/PROFILES/LEVELS/GROUPS/TRACE与三lane遥测切片。2121 L1绑定没有豁免。没有新增控制器、FSM、物理循环、reset或随机抽样。原SIDE1.5 sole override来自未改原SideOverride；原make_side_observer负责pre-close恢复。新参考审计仅读原生FSM字段。

默认CPU模式复用原认证和preflight，不创建App；显式native只由root独占启动，外部记录真正exit后再readback。固定reader要求全4096首出生、全部原事件、全部付款/teaching/identity/override以及全域peak逐值重现SIDE001；主报告仅排除开始时间、观察墙钟、机械summary SHA。缺行、付款/事件变化、出生/身份变化、contact容量指示、hook失败或错level/profile一律拒绝，原数据保留。所选trace另须初态至原terminal每物理tick完整，预计881/1877/2121为889/417/1005行。

输出可说明哪里发生闭链误差及伴随接触，不能单靠argmax或同步力矩证明动力学原因。881峰在PREPARE/ORBIT_POSITIVE；1877 ORBIT/ORBIT_NEGATIVE；2121 PREPARE/ORBIT_NEGATIVE。现有地形过滤接触不含self-contact配对身份，overflow原生标志仍不可用，derived impulse=force×dt不能冒充独立测量。资格为空、消费0。没有宣称新控制修复或正式部署。
