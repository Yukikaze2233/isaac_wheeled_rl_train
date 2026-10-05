# 同一封存模型 mean / stochastic clean 物理对照结果

两臂完成并通过配对校验；这是独立推理诊断，0PPO。正式Kaiser b4训练持续运行，未热改std、reward、合同或核心。

物理入口冻结：`13439727c365ef126f19d6da35c3f2f7291ae504`；比较器冻结：`e13d20f`；PT SHA：`78dc24240038910b3844cc609a26d259075239b137c008ee038de7a76b4963e7`。输入为消费24500、speed域actor4185、本轮新增622actor的完整状态，不能用本机诊断给当前790actor或整个域授予资格。

每case N4、每臂N12，fresh process/scene，20s、1000策略步、4000物理步。相同出生、材料、机构和私有采样流均核验一致。两臂12/12首次结局都是正常20s timeout，无failure/success/boundary/右删失；timeout只证明完整暴露，不等于功能达标。原生四腿/轮侧响应、闭链、重力、气簧保留；噪声、信号/通信延迟、接触/动力学随机化和非标准出生关闭。

| case / 阶段 | mean 均速 cm/s | stochastic 均速 cm/s | mean峰位移均值 cm | stochastic峰位移均值 cm |
|---|---:|---:|---:|---:|
| stand_305mm/stand | 2.59 | 10.35 | 32.84 | 47.86 |
| forward_0p5/stop | 2.76 | 10.29 | 27.21 | 20.82 |
| backward_0p5/stop | 2.32 | 9.95 | 22.91 | 20.47 |

stand是2s settle后的13s；stop是命令归零后2s warmup后的10s，自身phase起点anchor。合法移动位移没有算成停车漂移。所有窗口4/4完整，没有幸存筛选差异。随机动作使速度/姿态更抖，但停车净位移不是单调变大：mean前进停车峰位移均值27.21cm，stochastic20.82cm；不能把每帧速度抖动等同长期漂移。

mean静止均速2.59cm/s、pitch绝对值均值0.24°；stochastic10.35cm/s、3.76°。mean13s静止峰位移仍32.84cm，证明消除探索不能单独修好确定性静止精度。mean的前进steady vx MAE7.03cm/s、后退3.26cm/s；正向稳态仍有速度偏差。本轮只测±0.5m/s，不宣称高速加减速姿态已修好。

同共同暴露区间，mean腿侧限幅命中为0，两轮近0（右轮每case3/16000=.01875%）；stochastic左轮37.26–38.16%、右轮33.38–34.02%，四腿0.319–0.719%。定义是原生PD限幅前requested绝对值达到即时bounds，不是电机持续过载，也不是测得全部关节反力。轮std为0.88724/0.87292，乘训练轮目标尺度10rad/s是目标级约8.87/8.73rad/s扰动，不能当车体速度。

结论：探索对这个同PT clean fixture的抖动/角速度误差/轮侧限幅有明显物理贡献；确定性策略仍有低速偏差与长时间零速精度问题。不能由此认证训练全过程唯一病因、改变历史通过判定或直接降低/清零训练std。下一对优先只开启原生信号年龄，在同角色两臂内核验流和出生；随后才冻结训练候选，先证实延迟/噪声贡献及确定性偏差。未qualified腿bag仍不接入。

CPU入口/fixture/sampling/比较器共83测试通过；真实两臂各4000步完成，finally输入/源码SHA和actor所有保存张量不变。首个H1启动因遗漏原生构造器四配置覆盖在动作前失败；回执保留，严格修复后H2完整fresh两臂，未放宽任何实际机构字段。

TensorBoard只在Kaiser6006，事件别名 `v6-exploration-clean-mean-24500-20261005` / `v6-exploration-clean-stochastic-24500-20261005`。曲线为每4物理tick抽样的50Hz瞬时首cohort均值，step=物理tick、1tick=5ms，不是PPO步；共同暴露200Hz统计在机器JSON。未启动本机TB或重启远端服务。

机器比较文件SHA：`641d5a68a0a715f55e3fb5704d2f3c66bdc42a45d45d188dea001cbb2ee9f0ba`；所有原始物理/动作trace及runtime实际readback保留在两臂目录；`qualification=null`、`hardware_deployment_ready=false`。
