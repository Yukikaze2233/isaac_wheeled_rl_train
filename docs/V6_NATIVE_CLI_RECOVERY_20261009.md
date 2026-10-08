# source028 启动参数的一次命名修复

2026-10-09（UTC+8）。继续执行已授权的同 actor 串行接续；本修复只更正外部启动参数，不修改冻结 production335542dd、v26合同、课程、reward、物理或预算。

REC已真实封存为actor14169，消费41374／actor消费41204；完整检查点SHA ed24fe4e5e62fd8c8e01f81427c0741ae46799a237dee2b0fbd4b288df4e1085，旧wrapper归档完成。原028 native startup exit1／10.39s在authenticate_invocation处拒绝，stdout为空、scene不存在、0PPO、未创建launch intent。实际argv缺--learning-rate-scale，parser给None而认证要求1.0；global offset/block原默认0，现也显式传0。

52项原边界/原生启动CPU回归通过；13项新增未知故障、已有scene/launch/owner等拒绝测试通过。独立AST/parser复核确认唯一改变实际解析值为LR scale None→1.0。测试不代替真实原生启动。

外部repair helper使用原共享锁、原manifest和五份CLI失败字节SHA，必须认证无原PPO或observer、完整归档、原selection/账本不变、source inventory、原授权/immutable origin。仅在新目录boundary_integrated_028_cli_repair001.native执行一次4096×1策略／4物理步骤检查；原boundary_integrated_028.native全部保留。新repair意图、native意图、结果都是exclusive，部分/未知结果禁止自动重提，不重发旧SIGTERM。

原生检查真实通过才复用原q.step()/native admission/launch流程，一次提交新wrapper。接续完整actor/critic/Adam/RNG，10critic计费，speed新域actor从0开始。首actor状态恢复与真实更新另核验。此文是执行前计划；真实结果另存本次审计目录，不能因本文或CPU测试称已开始PPO。qualification仍null，腿侧bag仍未qualified。
