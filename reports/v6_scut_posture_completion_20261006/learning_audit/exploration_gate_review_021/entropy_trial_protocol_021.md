# source021 最小 entropy A/B 真PPO协议（尚未实施）

当前021保持原进程与冻结合同连续运行。本文件只有只读设计：没有试验部署、没有有效A/B、没有研究预算预留或PPO更新。optional entropy wiring只提供设置／读回接口，不能当strict AB继承、双seed或账本已经完成。

optional接线已更新到revision002：完成optimizer后再触发系数拒绝时保留实际收费与actor／critic分账，NaN/Inf等失败completion写安全null/status/repr且不封存坏state；独立6个真实CPU PPO/Adam故障路径通过。它仍没有四arm研究队列、strict合同或真实RNGfork准入，旧失败稿原字节保留。最新patch和审核SHA见JSON。

## 推荐规模与比较范围

A使用原生entropy_coef=0.005，B=0；两个真正训练RNGfork分别391021、391022（候选值，尚未准入）。每fork分别从同一份待认证的source021/v19最新own REC完整seal恢复actor／critic／Adam／std，各跑100actor更新，四arm共400。顺序seed1 A→B、seed2 B→A，一arm一个常驻worker，不做每40轮在线评测／反复建场景。每arm仅48s物理任务时间，因此这是机制筛查，不是收敛或实车验收。

更小的4×50=200只能作事先声明的快速筛查，每arm24s；若两fork没有足够实际连续门／释放曝光，零改善不能证伪entropy原因。推荐400上限，先把400严格计费，不因不喜欢结果临时加量。

entropy系数仅改actor目标，奖励／值目标定义不变，科学上无需新增10critic。**现generic task-change看到新合同SHA仍会执行10critic**；400总optimizer方案需新增严格的entropy-only零预热准入并测试。若保留原生10，每arm应110optimizer，共440optimizer／400actor，全部收费，不能把40critic漏记。不能直接复用旧P0的750来源、宽度修订、独立零起诊断账本和强制10critic。

按已观察的约14.675–16.148s/update，400真实更新约98–108分钟，另加四次启动、封存归档和离线0PPO回放；这不是承诺工期或新提速结果。

## 真实双seed与严格同父

四arm学习父必须同一份PT／seal／contract SHA；全状态逐值恢复后，明确记录父CPU／CUDA／env RNG恢复成功，再在原restored-clock reset之前执行**研究分支专用RNGfork**。同seed的A/B使用同一组起始随机状态，两个seed必须记录不同的实际采样state SHA。当前runner.load之后会恢复父CPU、env.generator和CUDA RNG，所以仅改命令行seed不能证明独立训练；旧两评测seed更不等于两个训练seed。

必须盘点所有实际活动随机流：Torch CPU／CUDA、env.generator、构造期ContactDomain／RigidBodyRandomization、command transport／活动USB及reset generator、RecoveryObserver目前固定190619的generator，以及其他活动模块。利用原生Generator接口，保持随机化分布和噪声参数不变；记录状态、初始domain realization与首次观测／命令／lesson／raw sampled action指纹。同seed A/B初始化一致，异seed采样流确实分叉。故障恢复从该arm最新全seal RNG续跑，禁止再次reseeding或重走父fork。

保持4096、35D/81D、50Hz策略／200Hz反馈PD160/2.5、完整95组与1449普通lane、所有随机化／延迟／轮响应／64/32求解器／气簧闭链／机械guard。恢复当前真实task/stage/global actor时钟并各自推进；origin2294和原1s／原followup不改，不用工程forced2294。除entropy及明确arm/fork/account元数据外，A/B不能有其他差异，更不能手工调低std或清Adam。native checkpoint不包含完整物理接触／延迟隐藏状态，故只声称学习全状态同父与配对fresh episodes，不声称物理快照完整fork。

## 唯一worker与账本：当前的实际阻塞

当前TaskCampaign整段run持有campaign独占flock，open_account拒绝活worker，persist会写自己掌握的完整attempt账本；它没有四arm研究队列。因此在“不动当前021连续PPO／不增加第二worker／仍统一计费”的条件下，**不能立即另外启动四arm**，不能向正在被owner维护的claim偷偷加入另一写入者，也不能用独立研究账本绕过20k／121000。

当前保持main不停。未来试验只有在独立认证的一次自然seal或受控源迁移后，由唯一scheduler在研究队列串行执行；这需要新增最小队列／branch收费准入。该串行研究槽确实会占用主线PPO时间，不能承诺主线当时无间断。不要借此文档现在停止021。

未来admission在原claim锁内回放最新attempt/progress，预留400（或明确440）recovery消费，强验recovery已收费+预留≤20000、campaign已收费+预留≤121000。四arm使用同一原账本的独立trial_id/seed/arm记录；每optimizer前持久reservation，不确定退出保留max(successful,attempted)，弃选与失败也收费。未跑完／删目录不能退款。400actor的策略转移量为39,321,600，物理interval为157,286,400；脚本／BLEND不属于actor样本的份额另记。

研究checkpoint必须位于单独research/entropy_021命名空间，不放进recovery/attempt_*/train：现latest_checkpoint按最高learning_updates自动选父，放错路径会让main静默接入未审arm。试验结束全部seal，主线默认恢复自己的已封存学习父，并保留更新后的全campaign消费债；是否采用某arm需另外明确审核，不能混同消费时钟与主线学习谱系。

## 怎样判定，避免只看σ

主要看同条件first-born、whole-actor-owned的**实际连续1s门、L0实际释放、原forward/yaw后续目标曝光与跟随**，保留完整分母、失败和删失。并看实际轮／腿／三轴gyro运动及普通静止、前后欠速、yaw角速度、真实驻停条件样本与姿态。新hold付款、六轴std和KL用原TB；std缩放只是未裁剪目标尺度，腿单位是角位置，不能和速度门直接类比；也不比较reward/s与entropy loss量纲。

每arm100actor可读100个dense std/KL/cost点。既有Behavior首个真实累计端点在actor10，终点actor100只支持90-update差分；不能伪造actor0 baseline，也不能把旧actor11–110的100窗口套给只跑100的arm。离线复用已有冻结mean/stochastic首出生观察器，对父和四endpoint同协议／固定评测seed检查；不在训练中每40轮拦停。若原父回放PT／协议／source不完全一致，就不能复用成绩。

原3mm闭链护栏、力矩／反馈／接触／几何护栏不放宽。护栏终止lane保留作竞争失败或删失，不从分母删除，更不把提前截断当接稳。B释放率好不能遮盖机械／源身份错误或变成实车资格。两实际训练fork的配对差异需方向一致，且有实际gate/释放/后续改善、没有普通技能／机械回归；只σ变小不够。不同seed判决相反或没有足够曝光就是inconclusive。

## 必须先完成的5个阻塞

1. 命名entropy-only strict recipe及独立设计／runtime表：现v19完整canonical recipe拒绝新增entropy／seed／warmup字段；现train固定seed290927。
2. 原campaign独占owner内的研究队列、400/440预留和不退款replay；旧P0独立ledger不可替代。
3. post-restore真实双训练RNGfork与活动generator实际native证据，恢复arm时不重新fork。
4. 显式0critic entropy-only规则或将40额外critic全部纳入440；不能沿用“新SHA=自动10”却只报400。
5. 真正PT／claim／预算／四arm合同／源码冻结绑定与首次PPO读回验证。当前这些父和预算binding仍为null，launch_allowed=false。

源码依据（均为冻结d0dce618，不是optional workingtree）：run_task_campaign.py:142–173活worker拒绝／账本恢复、:499–517研究父自动发现、:525–542固定worker命令、:643–713持续域、:747–810整run独占锁；task_campaign.py:53–59固定seed/预算、:709–783严格迁移和worker、:785–819非退款收费；train_chassis.py:61–72新SHA critic规则、:645–677全状态与RNG恢复、:700–723 restored-clock reset与critic；diagnostic_curriculum.py:15–29／34–82旧P0专用限制。具体绝对路径与SHA见配套JSON。
