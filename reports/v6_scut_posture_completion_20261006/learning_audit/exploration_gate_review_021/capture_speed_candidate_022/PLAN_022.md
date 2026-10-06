# CAPTURE4→1：隔离原生诊断候选

仅原生构造完成后的实例 `RecoveryFSM.config.capture_speed_rad_s` 从4改为1。使用source022冻结723732／实际v20 canonical原配置，同历史REC3076 full seal、mean、4096×600策略／2400物理步。ORBIT4、THRUST4、PREPARE现路径、SIDE1.5、CAPTURE→PREPARE gyro<.75、原8s脚本预算及1s最终稳定门不变；3mm、64/32、50Hz／200Hz、PD160/2.5、40Nm腿帽／4.5Nm轮帽、资产／奖励／出生／delay／RNG／网络／std不变，零PPO。

已有实测源版三条剩余guard：881 CAPTURE phys887→terminal888，2121 CAPTURE phys1003→terminal1076，1877 ORBIT phys411→terminal416。此候选用于辨识881／2121的脚本收拢速度与落地后闭链负载关系，不尝试在同一变量中修1877。速度减小可能改善机械误差，也可能只是延迟失败或增加原8s超时；负结果保留，不放宽任何门。

复用冻结 `run_source_script_native_022.py` 的真实source inventory、canonical_worker、design_preflight、old-v19→v20 parent验证及完整actor／critic／Adam／RNG恢复；复用冻结Teaching002／55唯一原生loop、五lane机械／PD／付款observer。新文件仅配置override、调用计数、当前tick派生参考比较、入口和完成后readback。没有第二次policy／decode／delay／FSM／reward producer，也没有独立控制器／物理步／reset／求解器。CPU模式默认，不创建App或应用运行期override；native入口只能由root在本机独占资源、独立审查后启动。

原v20 startup preflight先按正式配置构造，override在原env／observer构造完成后、首policy前应用。原SourceObserver的`control_hooks_injected=false`只描述其自身冻结钩子；新run_plan／report明确 `diagnostic_configuration_override=true` 与实际capture1，并记录完整before／effective配置。这个工程配置不是正式worker或部署合同，不冒称正式训练采用capture1。

每policy和每原FSM调用前后严格检查同一个config对象及完整值，只允许该一个键改变。唯一原update返回后读实际ref1；派生ref4使用同一输入和原source的valid／same-tick reroute、aligned goal、hip-selected pair winding及共同fraction算式。它仅用于确定全4096首个真实returned-reference差异，不是旧物理轨迹。保留原tensor/tensor算式顺序，无容差；同输入ref1必须与实际native returned legs逐位相同。全部原输入和RNG不变。

新的全域first-sensitive cutoff是改变该参数首次影响实际CAPTURE参考的reference−1。prefix只要求五条selected当时全部state／control／完整联合contact record多重集逐位相同，body／filter／tick／重复记录／每记录字段不松绑。全域介入后共享env reset/RNG可能分叉，因此后续各lane比较只作描述。旧ordered contact及旧global payment aggregate严格拒绝保留；不以gzip头或跨运行GPU浮点归约尾差冒充逐lane物理变化。每个实际run自身原reward／events对账仍必须通过。

五条firstbirth持续捕获到首terminal或12s右删失；原六pin、关节／gyro／FSM、script ref、一次raw decode与delayed RL PD target、原请求／帽／setter／engine effort、完整terrain normal／friction records保留。source022原observer全域每5ms argmax保持2400点。脚本时raw／delayed RL目标不是实际脚本目标。故障前后20ms由完成后CPU读回现完整记录取得，不扩observer或重采样。

关闭前先撤新outer step／update钩子并恢复capture4，再让原SourceObserver及其五层／付款／机械钩子恢复。原finish后半失败也写新的preclose失败侧证并保留原异常；foreign config对象／值／callback不覆盖。App.close可能直接退出Python，close是否返回仍unknown；root需独立记录实际exit0／进程消失，之后唯一review模式才读raw，不盲重试。

工程完整性、机械结果与功能资格分层。readback必须有实际600／2400、旧seal／出生／私有采样／std绑定、完整目标／付款计数、config及钩子恢复、全域prefix及contact scope。guard消失、脚本生存或L2计数变化都不授RL接管／1s dwell／motion release／followup／成功；每项以及失败、timeout、右删失单列。全域峰／guard和1877结果必须一起呈现。该candidate qualification始终null，不部署、不学习。

有限可观测性：terrain filter records可识别body／surface；没有self-contact pair、native overflow flag或随机mass／COM／friction draw。FK/IMU/probe support是参考所用physics k−1的代理，pin/contact/gyro是physics k后，policy终止在四子步后；原瞬态超线与实际terminal要分开。旧stochastic730mm仍未定位。

artifact revision002只修跨run identity比较边界：原Teaching002会在 `observer_artifact_sha256` 加入本次admission.json绝对路径。每次run的该文件分别按identity保存SHA核实且JSON内容必须相同，然后仅移除此一个精确path键比较其余identity。没有忽略其他observer／source／parent／RNG字段。旧candidate001／App-free001成功但whole-identity伪拒边界回执原样保留，native从未启动。
