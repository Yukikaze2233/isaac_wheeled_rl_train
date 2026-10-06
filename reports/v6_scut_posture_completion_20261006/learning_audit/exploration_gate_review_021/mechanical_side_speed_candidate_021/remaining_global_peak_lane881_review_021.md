# 剩余闭链终止的只读核对

实际 SIDE=1.5 rad/s 窗口中，lane881 首出生在 policy222（4.44s）被闭链门终止，最大误差27.620118mm来自 L_spring_upper_mount。global peak 保存了 physical887/888 的完整机构状态、两端点及已施加力矩；未保存该 lane 的初始状态、连续接触、setter参考和前后反馈。因此能定位发生时间和构型，暂不能证明接触、气簧或某条脚本参考是原因。

**phase7=PREPARE、route2=ORBIT_POSITIVE；这条峰值不是已知 SIDE 相位。** 保存的末态不能重建它是否更早进入过 SIDE。左三个约束在峰值处分别为13.797461、20.307258、27.620118mm，右侧均小于0.252mm；argmax 只定位误差，不能定位作用力来源。support/height 是 FK/IMU 可部署观察器的代理，非真实接触。

| lane | 实际 lesson / sampled profile | SIDE1.5首出生 | 基线6首出生 |
|---|---|---|---|
|881|L2 / front_down_90|222tick，PREPARE / ORBIT_POSITIVE，closure_gap；0接管|600tick右删失；1090 reference tick接管，未释放运动，未成功|
|1877|L2 / step200_lip_near_shell|104tick，ORBIT / ORBIT_NEGATIVE，closure_gap；0接管|400tick，recovery_script失败；0接管|
|2121|L1 / step200_upright_top|251tick，PREPARE / ORBIT_NEGATIVE，closure_gap；0接管|600tick右删失；1190 reference tick接管，未释放运动，未成功|

当前完整子步仅保存1708/2216，两 lane 的 layout、control、contact 和 reset 捕获均限定在该集合。全域 global peak 只是逐次打破全局最大值的快照；881在887已经15.331400mm，所以无法据此认定887是首次超过3mm。基线没有881全局峰值也不能替代其子步记录。

最小下一次测量保留原SIDE1.5控制、原4096出生、原600policy/2400physical、原source/PT/RNG/门，只将私有 observer 的保存集合换成881/1877/2121。2121必须精确绑定L1和实际sampled profile，不能用行为组名字推断姿态。记录各自原reset后初态到首terminal前的5ms参考/实际setter目标、反馈、pin状态、现有原生地形接触以及pre-autoreset快照。全4096首出生、原事件和付款必须逐值重现SIDE001；不可只看三条峰值。

同源同seal与初始出生hash支持全局匹配输入，但不足以归因这三条脚本轨迹的变化。大速度、40Nm附近力矩和气簧力属于同步状态证据，不能代替动力学因果证明。当前PPO=0、资格为空；未启动新场景，未改变正式训练。JSON保存全部输入前后SHA，实际字节均未变。
