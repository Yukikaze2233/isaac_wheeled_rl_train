# 原生信号年龄：同一 PT 的第二组物理配对

状态：physical_pair_completed_verified。两臂完成20秒，原生延迟journal/actor/PD来源严格核验通过；详见PIPELINE_AGE_RESULT.md。正式PPO继续，0新增训练消费，资格为null，不作实车部署证明。

输入沿用已完成clean配对的update24500：PT SHA `78dc24240038910b3844cc609a26d259075239b137c008ee038de7a76b4963e7`，合同SHA `f2a950b6769d2d46ccae5e7631f2e66af609cef1de251b3c8360724c140588f3`。每臂fresh进程、3case×N4、20秒、1000策略/4000物理步。出生seed190619、动作私有seed93001，mean也消费同形状原生潜在随机样本，但仅发送均值。架构/PD/轮响应、±0.5m/s命令及slew、首cohort终止/曝光规则沿用已冻结clean设计。

`--fixture-role pipeline_age_only`只恢复源合同的`PhysicalSignalDelay`完整配置；不启用观测噪声、USB/command transport、动力学/接触随机化、push、降级或非标准出生。固定已认证seal的域actor时钟4185，无PPO/时钟推进；出生时strength=0且lag=0，第一次原生begin_policy后strength=1。不能把CLI传参当实际模块/强度启用证据，需运行时逐项实读。

原生模型保留75%启用、每策略tick的±5ms抖动，以及60%普通、35%宽域、5%长尾profile：观测0–20/20–40/40–80ms，动作0–15/15–30/30–60ms。N12的实际抽样不代表全部长尾覆盖。200Hz PD继续读当前物理q/dq；延迟作用于actor传感器18维和解码后的腿/轮目标，不能称为1kHz底层反馈延迟。

直接在原生调用一次的begin_policy/targets/observe处记录只读journal；不得再次read_groups、重放随机draw、改lag或替换buffer。原生随机连续jitter未保存在对象中，raw_jitter明确null/not_covered；量化lag与base之差包含clip、round、enabled mask，不能标成原始jitter。

原生`GroupedDelayBuffer`用`min(requested_lag, num_pushes-1)`读取有效历史，因此启动阶段真实年龄小于请求lag。target在物理tick k前append，输入采样时刻k-1、应用时刻k；sensor在物理tick k后append，来源时刻k。actor输入取最近一次sensor refresh，早于本策略tick的begin_policy；分别记录采样、刷新和消费时钟，不能用新lag重写旧feedback来源。postphysics物理行发生在native observe前，明确标出刷新后续发生。

比较器须从报告实际role判定、拒绝异role伪配对；认证角色内同配置、出生、每lane实际enabled/profile/base/bounds和量化lag流、完整历史输出来源、私有动作RNG、首cohort共同曝光。输出的传感器/目标数值会随动作不同，不能要求两臂轨迹相等。clean与pipeline的构造模块不同，会改变出生generator消费顺序；跨角色只能作注明出生差异的诊断比较，不能宣称唯一因果效应。

执行完成后保存输入/核心源码finally不变回执、实际profile/有效年龄范围、每case outcome/暴露与速度/姿态/停车指标。曲线输出原生TensorBoard event并校验上传现有Kaiser6006，只保留本机SSH转发，不启动本机TensorBoard，不碰6007/6009。运行中b4主线合同不热改，诊断结果提交不触发重新部署。

执行H3 87c174e78fbec4d6d0c894a4a2d673a7a214cb46、比较器H4 5f2805a562d527aa3faab17bf9beb098c17c81df；111入口/fixture检查与19比较器检查通过。mean004/stochastic005各12原生timeout、无failure；实际8/12延迟启用、最高观测40ms/目标30ms，无80ms长尾。Kaiser事件SHA回执pipeline_tensorboard_upload_receipt.json；原始失败启动pipeline_preflight_004因root误填源码revision被入口在认证前拒绝，0场景/0PPO，raw report保留；正确H3的preflight005通过。
