原方案与真实两臂回执保持原字节。本补充只形成付款观测方案，尚未实现／运行／部署；工程020不包含质量改动。诊断仍绑定source712与seal2579（PT SHA `7f91d355…dbcca`），将来接续使用当时最新本域完整seal，不回滚此模型。完整SHA及源码行号见JSON。

真实均值／探索两臂各4096×600策略、2400物理、12s、0PPO：首生恢复mean放行1134、成功983；stochastic放行／成功均0，2304未完仅记右删失。初态和private采样流相同；后续reset可使环境随机流分岔。这个结果证明此保存探索在该窗口阻断连续hold与运动曝光，尚未证明reward或entropy唯一原因。两进程exit0由root实际观察；缺postclose sidecar，App.close是否返回仍未知。

旧takeover项确实 **inactive**。`recovery_prefix_revision.py:49` 删除recipe的 `recovery_takeover`，当前v18 worker无此键；`env.py:100/1590` 因settings=None不调用 `apply_recovery_takeover`。旧hold腿／轮／gyro broad+fine只存在源码，不能把其权重当真实付款。当前执行通用腿／轮速度、horizontal gyro、action/effort、姿态和可见命令dense tracking；当前mean/stoch wrapper55没有保存逐lane reward分项，不能从旧回执回推付款。

| 当前active项 | 源码参考状态 | density reward/s | 单20ms payment |
|---|---|---:|---:|
| leg_velocity | 四腿各2rad/s | −0.08 | −0.0016 |
| wheel_velocity | 两轮各0.25m/s轮缘、r=.06m | −0.00034722 | −0.000006944 |
| horizontal_angular_velocity | gyro XY norm=.75rad/s | −0.028125 | −0.0005625 |

这些是公式代入，不是实际原生测量或“reward不足”的证明。最后一项不含yaw；zero-command yaw另有dense yaw跟踪成本。完整tracking／姿态／effort／event及优势归一化都参与学习。基础1449虽然不走恢复脚本／outcome，仍接受通用v18成本并共享actor。L0放行后原motion目标仍零，stationary成本继续合法生效；目标桥接为明确教学修复，但探索0放行说明桥接单独不足。

最小观测用 **两份新artifact**：一个composition-only payment observer、一个导入冻结55 wrapper的driver；不改旧wrapper／controller／RNG／reset／reward。新增文件SHA纳入preflight和receipt。现在只提供设计，未创建或运行这两份观察程序。

1. 在原生 `begin_policy` 原调用之后保存旧lesson/profile、first-live和原hold-start，不增加调用次数。
2. 包装原 `episode_event_terms` 一次：记录实际输入与原返回的success/failure payment，原dictionary／tensor原样返回。600步必须恰600次；最后capture后、App.close前恢复hook。
3. `env.step` 后、base observer永久关闭首次done之前，消费原生 `extras.native_reward_components`、returned reward／done、pre-reset diagnostics与whole-interval `policy_learning_mask`。不读取auto-reset后的新lesson给旧付款归属。

逐lane逐步强验：`reward = .02 × Σactual_density + actual_success_event + actual_failure_event`，容差 `2e−5 + 1e−6×abs(reward)`。components为reward/s、event无dt；`extras.log`的component均值已经乘dt，无法按lesson／ownership拆账。所有分项都收录，不重新调用producer，也不只挑负项漏掉tracking peaks。inactive takeover key若出现则拒绝本合同。

按all-native／首生两个独立scope、四lesson、原stand／forward／正反yaw目标分箱。付款partition为基础actor-valid、恢复excluded、actor-owned terminal、hold中释放区间、仍holding、完整postrelease；保留terminal／hold-start／末FSM holding／命令及八门flags。whole mask决定PPO资格：中途SCRIPT→RL仍排除，actor导致FAILED的终止区间可有效，不能用末态pure_rl替代。释放所在20ms仍算hold、下一区间另记postrelease。实际环境付款与mask选择的即时付款分开，后者仍不是优势或梯度。

每箱记录policy-lane interval数、lane-seconds（n×.02）、分项sum density／sum payment／mean density、实际event与总reward、空箱null。device上有限分箱汇总，保存一次audit receipt与少量release／terminal核对示例，不新建训练曲线源。门是200Hz feedback、reward是50Hz末实际plant：如需margin分布，读取已有FSM调用参数dq／gyro，不新读PhysX；必须区分USB/control反馈和真实motor dq，不能把四门sample称四份payment。

Saved σ为direct scalar clamp，六维raw Gaussian entropy为 **1.889009 nat**。源码β=.005给解析entropy-loss `−βH=-0.009445`；直接项对σ的解析导数 `−β/σ`，只说明其方向，不能冒充真实训练std梯度。PPO使用 `surrogate + value_coef×value_loss − β×mean(entropy)`；whole-mask有效样本才进PPO。全局state-independent σ的mean entropy梯度不随hold人口份额简单放大。controller缩放／clip后物理动作不是PPO的raw Gaussian entropy。

真实100窗的逐sample V／return／GAE／normalized advantage、surrogate与entropy各自梯度、Adam／clip effects **unknown**。它们与reward/s、单20ms付款量纲不同，禁止按βH与一个cost数值比较就断定entropy主导。zeroPPO observer不造returns、不执行PPO；若既有event有source-bound loss，可按实际cadence引用，但不能虚构lesson attribution。

付款账通过后才选质量候选。默认复用现有 `dense_error_terms`：broad为 `−w_b(√(1+(e/s_b)²)−1)`，fine为 `w_f expm1(−(e/s_f)²)`，broad尾部已有非饱和方向；此前bounded ψ想法不再作为默认。只考虑目标明确的hold legs／wheels／gyro子项校准，优先处理与active项重叠，避免重复税；不整体复启旧tracking-baseline清零／重复高度套餐。具体尺度／权重待实际付款，当前不改cost、σ或entropy schedule。

观测DoD是2,457,600个lane-interval全部reconcile，600event／2400FSM原调用，source545／PT／std／learning clocks原值保留，首done仅一次且所有分箱可核总数。先用有意义CPU fixture拒绝重复dt／event、错auto-reset归属、midinterval所有权、release cutoff、excluded NaN／空箱及重复hook调用；本轮未运行这些新fixture。保留真实sensor/contact/followup连续1s、平台／script／tracking门与原fixed验收，机制观察不授予qualified。

质量候选仍在原独立恢复20k／总121k内、固定origin2294，不增加PPO、不借预算、不回滚2579。easy目标桥最早可在经冻结候选的首个自然reset生效；原3294／5294／8294难度时钟与15000终点沿用，未以此承诺达标。后续合同冻结前先认证最新own-domain fullseal与真实消费。
