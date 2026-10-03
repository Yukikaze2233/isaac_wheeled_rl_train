# V6 普通运动常驻混采修订（2026-10-04）

## 1. 目标与部署状态

用户明确要求像华南虎那样混采静止、低速和高速。当前普通运动模型改用同一个常驻PPO worker、同一策略，在4096环境中同时练基础行为和扩展运动；历史flat模型继续封存，不再另开一条低速训练线。内部账户名仍为`speed`，其教学主题为`mixed_locomotion`，以保留消费账本、本域actor时钟、历史attempt和已封存模型的身份。

已部署计划为`contracts/v6_task_campaign_v10.json`，worker为`contracts/v6_task_campaign_speed_worker_v10.json`，独立设计表为`contracts/v6_task_campaign_speed_expectations_v10.json`，命名迁移为`contracts/v6_task_campaign_migration_v10.json`。准备期间SSH曾返回“No route to host”，用户恢复局域网后于04:47（UTC+8）连接成功，并再次授权“部署续训”。04:54从冻结源码`122d8db63f376f0737e89b4caf746e3f744aee8e`在原root／tmux r11接续；已核验原生混采人口和首actor实际PPO。实际迁移结果见§7与部署回执；文件生成或测试本身不代表部署成功。

2026-10-04 04:47（UTC+8）实际只读核验：speed本域actor1666／3563，全局消费18010，唯一该任务supervisor2995678／worker2995685仍在PPO。迁移时重新读取实时状态，不复用这里的PID或推算进度。实际部署完成后须追加新源码、归档、合同、最新封存和首PPO核验回执，保留旧`launch.json`历史。

## 2. 为什么重新分配

v9已经包含全部41个flat行为，并非完全没有复习；但它把129组主要摊在60%运动和20%启停池中，基础组的单组份额随扩展组增多而降低。旧平移单组约0.625%、yaw约0.625%、基础启停约0.909%。因此“组还在”不能证明其采样足够。本次直接给基础行为族保留固定份额，不随训练迭代减小，也不依赖当前成功率来回切换。

这不是确定的遗忘根因判决。当前速度域还同时存在探索增大、5%边缘初态、5%持续单侧降级、宽幅命令及全强度延迟等差异；带探索训练读数不能替代父子同条件的确定性评测。保留基础采样是明确的设计改进，能否恢复精度仍要由续训行为验证。

参考固定版本`scutrobotlab/wheeled-legged_RL@b8ff79f3df855faf9dc92f4a282bd80c42649466`：实际归档的普通连续命令同时覆盖低速和高速，显式保留静止命令；任务线区分平地、冲刺、地形等主题，不等于一个模型只训某个速度档。参考[rough_dash实际命令配置](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml#L1742)及[地形命令覆盖](https://github.com/scutrobotlab/wheeled-legged_RL/blob/b8ff79f3df855faf9dc92f4a282bd80c42649466/pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml#L2129)。下表是本项目的首轮配额，不冒称为华南虎原数值，也不保证常驻混采即可消除遗忘。

## 3. 冻结的混采表

基础成员严格从原`flat_worker_v6`的41组提取，扩展成员是原`speed_worker_v6`其余88组；不凭名字猜测速度或复制新组。129组完整保留，命令spec、驻留、正反向和跨高度定义不变。

| 行为池 | 名义份额 | 4096环境实际人数 | 内容 |
|---|---:|---:|---|
| 固定高度静止 | 12% | 495 | 九高度锚点 |
| 连续高度静止 | 8% | 328 | 连续高度、完整零速驻留 |
| 升降 | 5% | 205 | 已定义高度扫描 |
| 基础双向平移 | 15% | 624 | 固定／连续低中速、曲线与跨高度 |
| 基础双向yaw | 10% | 408 | 正反固定／连续旋转与跨高度 |
| 基础启停 | 15% | 612 | 0.5／1m/s启停与速度曲线、完整停车窗 |
| 扩展运动 | 25% | 1008 | 原有中高速平移、快速旋转、旋转平移及跨高度 |
| 扩展启停 | 10% | 416 | 原有较高速制动与曲线 |
| 合计 | 100% | 4096 | 基础名义65%、扩展35% |

实际整数人口来自现有`choose_scene_groups`最大余数分配，每组至少一个环境，单组与理想配额相差不到一个环境；并列余数按稳定组顺序处理。不能把名义份额说成精确整数人口比例，或有效PPO梯度份额。人数在场景初始化计算，本次不新增物理子步线程、每步采样循环或同卡评测进程。

基础份额常驻，从当前actor时钟接续到原终点。速度并非只取离散评测点：既有连续命令／曲线采样spec继续覆盖中间幅值；固定行为和高度锚点用于保护基础访问。方向与高度变体等份，静态预检和4096分配测试检查配对与总人口。

## 4. 本次不同时改动的学习和控制合同

稠密宽域＋精核奖励、固定LR`1e-4`、探索参数、边缘初态／降级比例、域随机化和所有193项固定评测均继承。合同迁移拒绝夹带reward、物理、网络、响应、预算、终点、父模型或其他worker修改。基础reward是否支付仍由实际有效mask和分项曲线审计，不能把旧已停用`stationary_width=0.03`的论证套到当前dense分支。

保留4096×24 rollout、4 minibatch、5 epochs、35D actor／81D critic、6动作、50Hz策略和200Hz仿真新反馈PD；当前是粗步长仿真合同，不声称等价于1kHz实车PD。腿PD为160／2.5，轮侧响应及气簧模型继承；未qualified腿侧bag仍未接入。本次也不改变RMCS或硬件MIT命令。

其他五个worker及父模型树逐字节保留：flat→recovery、flat→当前运动、flat→terrain、flat→jump、jump→stepjump。本文不把未来terrain／jump候选称为已继承当前高速能力。后续完整运动复习及累加父树仍需独立命名修订和物理／来源核验，见主计划§8；不能仅改执行顺序或模型名称就宣称已解决所有后续域的遗忘。

## 5. 一次冻结迁移与自动接续

1. 源码、v10计划／worker／设计表／迁移manifest与测试先提交，再从该commit归档。30角色物化预检通过后才准备封存旧进程；活进程不读本机文件。
2. 连接恢复后实时读取`job.json`、进程归属、账本和当前域。若已有源修订或已切到下一域，先按实际身份重新核验，不复用过期PID、回执或启动第二份PPO。
3. 向已核实归属的supervisor发SIGTERM，使其转发worker、封存完整actor／critic／Adam和最新时钟，再汇总消费。等旧worker、supervisor和外层归档wrapper都退出后，在原root以命名`task-plan-revision-v10`接续；不用新run重新领取预算。
4. 同域新采样合同SHA触发原生10次critic-only适应，计入消费，不推进actor课程时钟。首critic验证actor不变／critic改变；首actor验证真正PPO学习，同时核对恢复初态逐值一致、LR及合同哈希。
5. 不从flat或A回滚。speed actor终点仍3563，terrain仍1186，其他自动终点保持；每域20000消费上限／总121000和全部历史消费不变。网络中断不产生退款或额外训练授权。

训练按原时钟自动推进，不恢复45项阻断gate。终点为`unassessed`、`qualified=null`；完成课程不等于技能精度通过。训练中的问题使用冻结版本修订，常规检查点和正常评测不等待用户再次发令。

## 6. 验证与观察

CPU回归检查命名迁移完整性、129组覆盖、基础41组来源、正反／高度对称、4096整数分配、30独立角色、固定评测不变、禁止改reward／时钟／预算、同域剩余适应次数及归档身份。v10继承v9已封存模型的显式review入口，避免因读取review重新建立场景。

本机针对迁移、review入口、归档恢复与原生整数分配的独立回归147项通过；另外课程／恢复／缩容／200Hz相关106项通过（两组有重叠，不相加为独立总数）。原生CLI的`--preflight-only`核验全部30角色通过。实际命令、合同SHA与整数人口保存在`reports/v6_mixed_locomotion_20261004/validation.json`。这些是静态与CPU核验，不声称新策略已经学会。

部署后用远端原生TensorBoard最近完整100个actor更新窗口，核对各池人数、有效PPO转移、驻停完整窗、静站速度和10s位移、平移MAE／有符号偏置、yaw角速度MAE、跨高度表现、失败／边界截断、探索std与每更新耗时。停车位移只统计制动后的窗，运动总路程不叫静止漂移。训练曲线保持其探索和随机化条件，不直接称为确定性验收。

基础保持与扩展收益要同时观察：基础精度改善但高速全面回退，不算达到本次目的；reward上涨也不算。固定父子比较安排在独立资源或域自然结束后，不与本次PPO抢同一卡。TensorBoard仅运行于Kaiser远端6006；本机不启服务，6007属于另一实验。

## 7. 实际部署与学习核验

2026-10-04 04:54（UTC+8）从提交`122d8db63f376f0737e89b4caf746e3f744aee8e`归档部署，源码包SHA`5baeee6f818050f8ec3585a295808f742e0851305db1d57392423fffde005b21`。实际v10 plan SHA`22f76bcd5676dae800207040ce446e0ec7de1a9842380d4b2e4d5ca2e90f3587`，worker SHA`37837b286de10a66404ef155ae84ead613a90f76375d4abe6874a042eae6b406`。远端使用归档中的预检器，30角色及完整归档文件SHA均通过，未部署本机未提交的回放改动。

旧supervisor／worker于actor1684、消费18028合法封存，外层归档退出后才启动r11／speed attempt_001。父完整模型SHA`08642e05947e0133390ccd2fe7323e3de6497a997574832e5e5a96f75d4af742`，其学习谱系15092与消费18028不同，继续分开记录。实际剩余1879 actor更新，加10次计费critic适应，worker请求1889次更新；没有重新领3563次或退款。

原生场景实际4096人口与§3逐项相同，129组完整；初始actor／critic／Adam逐值一致。首18029 critic检查点的actor不变、critic改变、本域actor仍1684；首18039检查点本域actor1685、warmup余0，9个actor张量改变，所有actor／critic参数有限，optimizer LR为`1e-4`。这确认新混采已经进入真实学习，不能据此判断新配额的行为效果。

05:00:56实际核验supervisor3069343／唯一该任务worker3069350、本域actor1692／全局消费18046，保持`actor_and_critic`。当前自动节点仍3563，其后按原合同接terrain；固定评测不阻断PPO，资格仍null，后续五worker与父链没有被本次采样修订改写。TensorBoard run为`v6-task-domains-20261001/speed/attempt_001/train`，远端6006。

回执在[部署目录](../reports/v6_mixed_locomotion_20261004/deploy/)：`mixed_predeploy_011.json`、`source_revision_011.json`、`mixed_seal_011.json`、`mixed_resume_verification_011.json`、`mixed_first_actor_verification_011.json`、`mixed_live_011.json`与`verified_launch_011.json`。最新通用入口仍为[launch.json](../reports/v6_task_campaign_20261001/deploy/launch.json)。本节及结果回执的后续提交仅归档事实，不触发再次部署或重启。
