# Source021 原生教学观察器

此入口只做工程观测，尚未由准备者运行原生场景。使用源 `d0dce6181df83b03fe166abcbadef3c81de2e57e` 的完整 v19 recovery train role、4096 环境、600 策略步和2400物理步；不改变物理、控制、奖励输出、命令、重置、标准差或学习状态。

本轮准入为 `engineering_flat_prefix`：使用正式 probe 认证的旧 flat 终点 PT、合同和逐文件 seal，只应用均值动作。实际父检查点 task/global 时钟为13000/13368，总消费14020；三个环境 actor 时钟独立强制为 prefix origin2294，不计消费，也不表示正式 recovery resume。正式学习父模型由部署程序另取最新 REC seal。沿用原55观察器的保存 CPU/env/CUDA RNG恢复后一次原生 reset；它不是旧正式 probe 的新 seed 出生样本。原生 Gaussian/std 仍加载且逐值保持，不构造旧 formal probe 的 PPO runner/optimizer。

观察器复用不可变55步进、51580事件付款观察器和a9e7生命周期组合，完整正式 probe authenticate 与 StructuralAudit 来自真实冻结021源码。每区间真实 hold producer、step和原event各调用一次。成本输入与实际返回逐项对照；非held普通/排除/已释放区间、原事件和正奖励保持原值，dict顺序和dt处理保持。mask须等于 enabled∩whole-interval-actor-owned∩begin-policy-hold。新记录仅为侧证，不创建reward键。

释放发生在区间内时，该区间仍使用原始零目标和hold费用；下一真实区间才记录原forward或yaw目标。三种L0后续目标分别统计实际pre-autoreset命令和完整actor区间，包括首出生与所有后续出生；不根据原目标标签推断实际曝光。

`report.json`、`payment_summary.json`与`teaching_summary.json`在App.close之前封存；异常也先落失败侧证，因为Kit可能exit0而不回到Python。外部 `review_output(output)` 必须通过：完整600/2400和2457600付款区间、实际held/ordinary/excluded覆盖、原事件逐值对账、完整角色工程检查、actor/std/学习时钟不变、全部sidecar身份匹配，且L0 forward、正yaw和负yaw均有真实非零曝光。只得到覆盖缺口不会通过collector；exit0不构成成功。

新增teaching、StructuralAudit和payment回调按安装前所有者恢复；继承55原观察器回调保持原App生命周期，不宣称所有原观察器回调都卸载。CPU44边界测试与真实冻结源/flat输入CPU预检已通过；这不能证明原生动力学、PPO收益、实车资格或Kaiser提速。独立原生运行由root独占执行，准入结果的qualification始终为null。

运行参数见 `native_observer_invocation_021.json`。准备者没有运行GPU、SSH或PPO。
